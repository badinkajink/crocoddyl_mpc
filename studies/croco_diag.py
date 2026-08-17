#!/usr/bin/env python3
"""Geometry diagnostic on a planned trajectory, read out of MuJoCo's narrowphase.

The S12 write-up reported one number for the plan's geometry -- "deepest
robot-table penetration -67.8 mm, and it is the gripper" -- and one for its
physics -- "no replay establishes brace contact".  Those are the same question
asked twice and neither answer says WHERE in the maneuver it goes wrong, so this
walks the plan node by node and reports, per node:

  * every robot body MuJoCo's narrowphase finds against the table, with depth
  * the height of each bracing site above the tabletop plane
  * how far the site is OUTSIDE the table's x/y footprint -- a site at the right
    height off the SIDE of a 0.595 m table reads as placed on a z residual and is
    not touching anything (this is the failure contact_select.table_y_range was
    added for, and a z-only barrier cannot see it either)

so "the braced phase is leaning on thin air" can be distinguished from "the brace
is down but the arm is through the slab".
"""

import argparse
import json
import os

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import contact_select as cs
import mujoco


def _under(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def diag(xs, subset, run_dir, start_key="stand", n_approach=None):
    m, d = cs.load(ik_margin=0.0)   # no margin: real contacts only
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, start_key)
    qpos_full = m.key_qpos[kid].copy()
    d.qpos[:] = qpos_full
    mujoco.mj_forward(m, d)
    tz = cs.table_top_z(m, d)
    x0, x1 = cs.table_x_range(m, d)
    y0, y1 = cs.table_y_range(m, d)
    tbl = cs.bid(m, "table")

    rows = []
    for k, x in enumerate(xs):
        d.qpos[:] = cb.pin_to_mj(x[:cb.NQ_ROBOT], qpos_full)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        pen = {}
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
            if tbl not in (b1, b2):
                continue
            rb = b2 if b1 == tbl else b1
            # ROBOT bodies only.  `object` sits 15 mm inside the tabletop in the
            # model as shipped; counting it reports a 15 mm penetration on every
            # node of every plan, including the ones where the robot is standing
            # still, which is how it read as a floor on the plan's own error.
            if not _under(m, rb, 1):
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb)
            pen[name] = min(pen.get(name, 0.0), float(con.dist))

        rec = {"k": k, "pen": pen,
               "worst_pen": min(pen.values()) if pen else 0.0,
               "worst_body": min(pen, key=pen.get) if pen else ""}
        for s in list(subset) + [t for t in cs.ARM_SITES if t not in subset]:
            body, off = cs.SITES[s]
            p = cs.point_world(m, d, body, off)
            rec[f"z_{s}"] = float(p[2] - tz)
            # How far the site is OUTSIDE the tabletop footprint, horizontally.
            # A boolean "inside?" is the wrong instrument here: the certified
            # brace lands on the table's EDGES (elbow at x = 0.499 against a near
            # edge at 0.500, forearm at y = 0.300 against a side edge at 0.2975),
            # so an axis point on the link reads as outside by 1-3 mm while the
            # link's SURFACE is very much on the table.
            rec[f"out_{s}"] = float(np.hypot(max(x0 - p[0], 0, p[0] - x1),
                                             max(y0 - p[1], 0, p[1] - y1)))
        p = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
        rec["reach_xyz"] = [float(v) for v in p]
        rec["com"] = [float(v) for v in d.subtree_com[1]]
        rows.append(rec)
    return rows, dict(table_z=tz, x_range=[x0, x1], y_range=[y0, y1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="elbow+forearm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--dir", default="runs/2026-08-05_session12/croco")
    args = ap.parse_args()
    tag = args.tag or args.mode.replace("+", "_")
    xs = np.load(os.path.join(args.dir, f"xs_{tag}.npy"))
    with open(os.path.join(args.dir, f"plan_{tag}.json")) as fh:
        plan = json.load(fh)
    rows, geo = diag(xs, plan["subset"], args.dir, plan["start"])

    na = plan["n_approach"]
    # The braced phase ENDS where the return phase starts; without this the
    # summary averages the brace height over nodes where the arm is deliberately
    # in the air on its way back to stand, and reports a 120 mm hover as a defect.
    nb_end = len(rows) - plan.get("n_return", 0)
    print(f"table top z={geo['table_z']:.4f}  x{geo['x_range']}  y{geo['y_range']}")
    print(f"{'k':>4} {'worst pen':>10} {'body':26}", end="")
    for s in plan["subset"]:
        print(f" {'z_'+s:>10} {'out':>5}", end="")
    print()
    for r in rows:
        if r["k"] % 5 and r["k"] not in (na - 1, na, len(rows) - 1):
            continue
        mark = " <- brace phase starts" if r["k"] == na else ""
        print(f"{r['k']:>4} {r['worst_pen']*1000:10.1f} {r['worst_body'][:26]:26}", end="")
        for s in plan["subset"]:
            print(f" {r['z_'+s]*1000:10.1f} {r['out_'+s]*1000:5.1f}", end="")
        print(mark)

    worst = min(rows, key=lambda r: r["worst_pen"])
    print(f"\ndeepest penetration {worst['worst_pen']*1000:.1f} mm "
          f"({worst['worst_body']}) at k={worst['k']}")
    for s in plan["subset"]:
        zs = [r[f"z_{s}"] for r in rows[na:nb_end]]
        outs = [r[f"out_{s}"] for r in rows[na:nb_end]]
        print(f"brace {s:9s} over the braced phase: z {min(zs)*1000:+.1f} .. "
              f"{max(zs)*1000:+.1f} mm above table, "
              f"{max(outs)*1000:.1f} mm outside the footprint at worst")
    out = os.path.join(args.dir, f"diag_{tag}.json")
    with open(out, "w") as fh:
        json.dump({"geometry": geo, "rows": rows}, fh, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
