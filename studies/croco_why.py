#!/usr/bin/env python3
"""Why the closed loop fails: one record per (contact mode x reach target) cell.

The S13 sweep scored each cell with four numbers -- did it fall, how far did the
hand miss, how much force went through the PLANNED brace site, what was the worst
penetration.  Those four are enough to say a cell failed and not enough to say
why, and three of them are quietly misleading:

  * `mpc_brace_N` sums contacts on the SITE'S BODY only.  A cell where the robot
    ends up holding itself up on its torso, or on the reaching arm, or on the
    gripper of the arm that is not bracing, reports 0 N -- indistinguishable
    from a cell that is genuinely unsupported.  It is not the same failure and
    it does not have the same fix.
  * `worst_penetration` is `min(contact.dist)`, and for the MESH links -- which
    is what the elbow and forearm braces actually are -- MuJoCo's convex
    narrowphase does not return a usable depth.  Measured against an exact
    vertex-vs-box SDF it under-reports by an order of magnitude.
  * `reach_err` at one instant hides whether the plant reached and then let go,
    never got there, or was thrown.

So this module recomputes the cell from the two trajectories that were saved --
the OCP's own `xs` and the replay's `qpos` trace -- and asks, per node:

  gap        exact signed distance from each brace surface to the TABLE BOX,
             for the plan and for the plant.  Surface, not site: the site is a
             point on the link axis and the thing that touches wood is the
             capsule/box/hull around it.
  contacts   every table contact MuJoCo reports, with its force, the body it is
             on, and WHERE on the table it is -- top face, near edge, or side
             rail.  A brace on the rail is not a brace.
  load       that force bucketed by role: planned brace / bracing arm elsewhere
             / reaching arm / torso / legs.
  reach      hand-to-target error for the plan and the plant, so a stall is
             distinguishable from a fall.

Everything here is measurement.  No cost, no solver, no model of what should
have happened -- just the two trajectories and the geometry they are in.

usage: croco_why.py --sweep runs/.../sweep [--out why.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import mujoco

import contact_select as cs
import croco_bridge as cb

BRACE_SITES = ("elbow", "forearm", "palm")

# Which bodies' collision geoms ARE each brace surface.  Deliberately not "the
# site's body": for the palm, the site hangs off `*_magpie_gripper` but the
# hand assembly that reaches the wood is the wrist housing and its pad, one
# link up the chain -- which is why a body-identity test reports a palm brace as
# absent while it is visibly flat on the table (croco_modes.contact_gaps says
# the same thing about the certification).
SURFACE_BODIES = {
    "elbow":   ["{a}_shoulder_yaw_link"],
    "forearm": ["{a}_elbow_link"],
    "palm":    ["{a}_wrist_roll_link", "{a}_wrist_pitch_link",
                "{a}_wrist_yaw_link", "{a}_magpie_gripper"],
}
EXTRA_GEOMS = {"forearm": ["{a}_forearm_pad"], "palm": ["{a}_wrist_pad"]}

EDGE_TOL = 0.008        # [m] how close to a table edge counts as an edge contact
TOP_TOL = 0.004         # [m] how close to the top plane counts as a top contact
F_ON = 5.0              # [N] a contact is "carrying" above this


# ------------------------------------------------------------------ geometry --
class Table:
    """The tabletop as a box, with an exact signed-distance function."""

    def __init__(self, m, d):
        self.g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                   "table_top_collision")
        mujoco.mj_forward(m, d)
        self.c = d.geom_xpos[self.g].copy()
        self.R = d.geom_xmat[self.g].reshape(3, 3).copy()
        self.h = m.geom_size[self.g].copy()
        self.top = self.c[2] + self.h[2]

    def sdf(self, P):
        """Signed distance of world points to the box (negative inside)."""
        L = (P - self.c) @ self.R
        q = np.abs(L) - self.h
        out = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
        return out + np.minimum(q.max(axis=-1), 0.0)

    def where(self, p):
        """Classify a contact point on the tabletop, and how far it is inboard.

        The distinction that matters here is not top-vs-side, it is INTERIOR
        top face vs a top-face contact sitting on an edge.  The table is only
        0.595 m wide, the reaching arm owns the middle of it, and a brace pose
        that puts the bracing arm out at |y| = 0.29 is one lateral centimetre
        from the wood ending -- which is a very different object from a brace
        in the middle of a slab, even though both report `contact.dist < 0`
        and both feed the friction cone the same way.

        Returns (label, inboard [m]) where `inboard` is the distance from the
        contact to the nearest top-face edge, negative off the face.
        """
        L = (p - self.c) @ self.R
        inb = float(min(self.h[0] - abs(L[0]), self.h[1] - abs(L[1])))
        if abs(L[2] - self.h[2]) > TOP_TOL:      # not on the top plane at all
            return ("rim" if L[2] > 0 else "under"), inb
        if inb < EDGE_TOL:
            return "edge", inb
        return "top", inb


def geom_cloud(m, g):
    """(points in geom frame, radius) -- see brace_surfaces.geom_points."""
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        a = m.mesh_vertadr[m.geom_dataid[g]]
        n = m.mesh_vertnum[m.geom_dataid[g]]
        return m.mesh_vert[a:a + n].astype(float), 0.0
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                         for k in (-1, 1)], float) * s[:3], 0.0
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        th = np.linspace(0, 2 * np.pi, 48, endpoint=False)
        return np.concatenate([
            np.stack([s[0] * np.cos(th), s[0] * np.sin(th),
                      np.full_like(th, sg * s[1])], 1) for sg in (-1, 1)]), 0.0
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return np.array([[0, 0, -s[1]], [0, 0, s[1]]]), float(s[0])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return np.zeros((1, 3)), float(s[0])
    raise ValueError(f"geom type {t}")


class Probe:
    """Model + the per-site geom sets + cached point clouds."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(cs.MODEL)
        self.d = mujoco.MjData(self.m)
        self.table = Table(self.m, self.d)
        self.tbl_body = cs.bid(self.m, "table")
        a = cs.BRACE_ARM
        other = "right" if a == "left" else "left"

        def gid(n):
            return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n)

        self.surf = {}
        for s, bodies in SURFACE_BODIES.items():
            gs = []
            for b in bodies:
                bid = cs.bid(self.m, b.format(a=a))
                gs += [g for g in range(self.m.ngeom)
                       if self.m.geom_bodyid[g] == bid
                       and self.m.geom_contype[g] == 1]
            gs += [gid(n.format(a=a)) for n in EXTRA_GEOMS.get(s, [])]
            self.surf[s] = [g for g in gs if g >= 0]
        self.cloud = {g: geom_cloud(self.m, g)
                      for gs in self.surf.values() for g in gs}

        # role of every body, for load attribution
        self.role = {}
        for b in range(self.m.nbody):
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm.startswith(a) and "ankle" not in nm and "knee" not in nm \
                    and "hip" not in nm:
                self.role[b] = "brace_arm"
            elif nm.startswith(other) and "ankle" not in nm and "knee" not in nm \
                    and "hip" not in nm:
                self.role[b] = "reach_arm"
            elif "ankle" in nm or "knee" in nm or "hip" in nm:
                self.role[b] = "legs"
            elif nm in ("torso_link", "pelvis"):
                self.role[b] = "torso"
            else:
                self.role[b] = "other"
        # the bodies each site's surface lives on, for "is this contact the brace"
        self.site_bodies = {s: {self.m.geom_bodyid[g] for g in gs}
                            for s, gs in self.surf.items()}

    # -------------------------------------------------------------- per node --
    def set_q(self, qpos):
        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)

    def gap(self, site):
        """Exact signed distance from the site's surface to the table box [m]."""
        best = np.inf
        for g in self.surf[site]:
            P, r = self.cloud[g]
            W = self.d.geom_xpos[g] + P @ self.d.geom_xmat[g].reshape(3, 3).T
            best = min(best, float(self.table.sdf(W).min()) - r)
        return best

    def site_z(self, site):
        return float(cs.point_world(self.m, self.d, *cs.SITES[site])[2]
                     - self.table.top)

    def reach(self, target):
        p = cs.point_world(self.m, self.d, cs.REACH_BODY, cs.REACH_OFF)
        return float(np.linalg.norm(p - np.asarray(target)))

    def contacts(self, subset):
        """Table contacts with force, role, and location on the table."""
        out = []
        buf = np.zeros(6)
        for c in range(self.d.ncon):
            con = self.d.contact[c]
            b1 = self.m.geom_bodyid[con.geom[0]]
            b2 = self.m.geom_bodyid[con.geom[1]]
            if self.tbl_body not in (b1, b2):
                continue
            og = con.geom[1] if b1 == self.tbl_body else con.geom[0]
            ob = self.m.geom_bodyid[og]
            nm = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, ob) or ""
            if nm == "object":
                continue
            mujoco.mj_contactForce(self.m, self.d, c, buf)
            fn = abs(float(buf[0]))
            site = next((s for s in subset if ob in self.site_bodies[s]), None)
            tg = con.geom[0] if og == con.geom[1] else con.geom[1]
            if tg == self.table.g:
                where, inb = self.table.where(con.pos)
            else:
                where, inb = "leg", 0.0     # a table LEG, not the top at all
            out.append(dict(
                geom=mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, og)
                or f"mesh#{og}",
                body=nm, role=self.role.get(ob, "other"), site=site,
                fn=fn, pos=[round(float(v), 4) for v in con.pos],
                where=where, inboard=round(inb * 1e3, 1)))
        return out


# ------------------------------------------------------------------- driving --
def cell(probe, xd, mode, target, n_approach=120):
    """Everything measurable about one grid cell, node by node."""
    tag = mode.replace("+", "_")
    fq = os.path.join(xd, f"replay_{tag}_mpc_q.npy")
    fx = os.path.join(xd, f"xs_{tag}.npy")
    frep = os.path.join(xd, f"replay_{tag}_mpc.json")
    if not (os.path.exists(fq) and os.path.exists(frep)):
        return None
    q = np.load(fq)
    xs = np.load(fx) if os.path.exists(fx) else None
    rep = json.load(open(frep))
    subset = [s for s in mode.split("+") if s in BRACE_SITES]

    rows = []
    for k in range(len(q)):
        probe.set_q(q[k])
        r = dict(k=k,
                 gap={s: round(probe.gap(s) * 1e3, 3) for s in subset},
                 site_dz={s: round(probe.site_z(s) * 1e3, 3) for s in subset},
                 reach=round(probe.reach(target) * 1e3, 2),
                 com=[round(float(v), 4) for v in probe.d.subtree_com[0]],
                 pelvis_z=round(float(probe.d.xpos[cs.bid(probe.m, "pelvis")][2]), 4),
                 con=probe.contacts(subset))
        if xs is not None and k < len(xs):
            probe.set_q(cb.pin_to_mj(xs[k][:34], q[k].copy()))
            r["gap_plan"] = {s: round(probe.gap(s) * 1e3, 3) for s in subset}
            r["reach_plan"] = round(probe.reach(target) * 1e3, 2)
            r["com_plan"] = [round(float(v), 4) for v in probe.d.subtree_com[0]]
        rows.append(r)

    br = rows[n_approach:]

    def load_by_role(rs):
        acc = {}
        for r in rs:
            for c in r["con"]:
                key = f"brace:{c['site']}" if c["site"] else c["role"]
                acc[key] = acc.get(key, 0.0) + c["fn"]
        return {k: round(v / max(len(rs), 1), 2) for k, v in
                sorted(acc.items(), key=lambda kv: -kv[1])}

    def where_hist(rs):
        acc = {}
        for r in rs:
            for c in r["con"]:
                if c["fn"] > F_ON:
                    acc[c["where"]] = acc.get(c["where"], 0.0) + c["fn"]
        tot = sum(acc.values()) or 1.0
        return {k: round(100 * v / tot, 1) for k, v in
                sorted(acc.items(), key=lambda kv: -kv[1])}

    # How far inboard the load actually sits: force-weighted mean over the
    # braced phase.  A brace at 10 mm of inboard margin on a 595 mm slab is
    # doing the same job as one at 150 mm right up until it isn't.
    wf = [(c["inboard"], c["fn"]) for r in br for c in r["con"]
          if c["fn"] > F_ON and c["where"] in ("top", "edge")]
    inboard = (round(sum(i * f for i, f in wf) / sum(f for _, f in wf), 1)
               if wf else None)
    wb = [(c["inboard"], c["fn"]) for r in br for c in r["con"]
          if c["fn"] > F_ON and c["site"]]
    inboard_brace = (round(sum(i * f for i, f in wb) / sum(f for _, f in wb), 1)
                     if wb else None)

    # WHEN it fell matters more than THAT it fell.  A cell whose pelvis is below
    # 0.55 m at node 90 never got to the brace at all -- the contacts do not
    # switch on until 120 -- so it is a failure of the APPROACH, of getting from
    # a stand into the pre-brace pose on the legs alone, and no amount of contact
    # modelling touches it.  Reported together, the two look like one failure
    # mode and the fix aims at the wrong stage.
    fell = bool(rep["summary"]["fell"])
    below = [r["k"] for r in rows if r["pelvis_z"] < 0.55]
    fall_node = below[0] if below else None
    if not fell:
        outcome = "reached" if rows[-1]["reach"] <= 60 else "stalled"
    elif fall_node is not None and fall_node < n_approach:
        outcome = "fell_approach"
    else:
        outcome = "fell_braced"
    duty = {s: round(100.0 * np.mean([any(c["site"] == s and c["fn"] > F_ON
                                          for c in r["con"]) for r in br]), 1)
            for s in subset}
    return dict(
        mode=mode, target=list(target), fell=fell,
        fall_node=fall_node, outcome=outcome,
        n_approach=n_approach,
        reach_end=rows[-1]["reach"], reach_braced_min=min(r["reach"] for r in br),
        reach_plan_braced=(round(float(np.mean([r["reach_plan"] for r in br
                                                if "reach_plan" in r])), 2)
                           if "reach_plan" in br[0] else None),
        gap_braced={s: round(float(np.mean([r["gap"][s] for r in br])), 2)
                    for s in subset},
        gap_plan_braced=({s: round(float(np.mean([r["gap_plan"][s] for r in br])), 2)
                          for s in subset} if "gap_plan" in br[0] else None),
        duty=duty, load=load_by_role(br), where=where_hist(br),
        inboard=inboard, inboard_brace=inboard_brace,
        tau=rep["summary"]["max_tau_ratio"],
        rows=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n-approach", type=int, default=120)
    args = ap.parse_args()

    sw = json.load(open(os.path.join(args.sweep, "sweep.json")))
    probe = Probe()
    # STAMP THE GEOMETRY.  Everything here is measured by re-forwarding saved
    # qpos through the CURRENT model, so a run analysed against a model whose
    # collision proxies have since changed is silently wrong -- and this study
    # changed exactly those proxies mid-session.  The fingerprint is the brace
    # surfaces' own dimensions, which is what the measurement depends on.
    fp = {mujoco.mj_id2name(probe.m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"#{g}":
          [int(probe.m.geom_type[g])] + [round(float(v), 5) for v in probe.m.geom_size[g]]
          for gs in probe.surf.values() for g in gs}
    out = {"targets": sw["targets"], "modes": sw["modes"],
           "target_yz": sw["target_yz"], "ngeom": int(probe.m.ngeom),
           "brace_surfaces": fp, "cells": {}}
    print(f"model {os.path.basename(cs.MODEL)}  ngeom {probe.m.ngeom}  "
          f"wrist pad {fp.get('%s_wrist_pad' % cs.BRACE_ARM)}")

    for row in sw["rows"]:
        if row["mode"] == "legs_only" or not row.get("planned"):
            continue
        xd = os.path.join(args.sweep,
                          "x%04d" % int(round(row["target_x"] * 1000)))
        tgt = (row["target_x"], sw["target_yz"][0], sw["target_yz"][1])
        c = cell(probe, xd, row["mode"], tgt, args.n_approach)
        if c is None:
            continue
        key = f"{row['target_x']:.3f}|{row['mode']}"
        out["cells"][key] = c
        top = list(c["load"].items())[:3]
        print(f"x{row['target_x']:.3f} {row['mode']:22s} "
              f"{c['outcome']:14s}"
              f"reach {c['reach_end']:7.1f} mm (plan {c['reach_plan_braced']})  "
              f"duty {c['duty']}  "
              f"load " + " ".join(f"{k}={v:.0f}N" for k, v in top) +
              "  on " + ",".join(f"{k}:{v:.0f}%" for k, v in
                                 list(c["where"].items())[:3]) +
              (f"  inboard {c['inboard']:.0f}mm" if c["inboard"] is not None
               else ""))

    if args.out:
        json.dump(out, open(args.out, "w"))
        print(f"\nwrote {args.out}  ({len(out['cells'])} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
