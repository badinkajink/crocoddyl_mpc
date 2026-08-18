#!/usr/bin/env python3
"""Enumerate contact modes at one reach target and SAVE the pose for each.

sweep5_full.py already enumerates 2^5 contact subsets and scores them with the
static QP, but it keeps only the scalars -- the pose that earned the score is
thrown away when `load()` is called for the next subset.  crocoddyl needs the
pose: q* is the terminal target of the OCP and the contact subset is its contact
schedule, and they have to be the SAME solve or the plan is aiming at a pose that
was never certified.

So this is sweep5_full's inner loop with the qpos retained, written one file per
admissible mode, plus a manifest recording which solve produced which pose.

Admissibility is sweep5_full's, unchanged:
    IK ok (reach < 3 cm, feet pinned, no penetration, every site actually
    touching)  AND  base residual < 1 N  AND  max |tau|/tau_limit <= 1.
Ranking is sweep5_full's too: least normalized actuator effort, ties broken
toward FEWER contacts.
"""

import argparse
import itertools
import json
import os
import subprocess

import numpy as np

import contact_select as cs
import mujoco

SITES5 = ("elbow", "forearm", "palm", "hip", "torso")
NO_CONTACT = 9999.0      # sentinel gap for "MuJoCo reports no contact here"
# How far from a site MuJoCo's contact point may be and still count as that
# site's contact [m].  Set from the study's own measured site-to-contact
# drifts (S11: elbow 47 mm, forearm 41-56, palm 32, and up to 89 mm for the
# far corner of the wrist housing) and kept below the 57 mm elbow-forearm
# site separation being a problem by assigning each contact to its NEAREST
# site only.
SITE_RADIUS = 0.10
# Deepest reported gap that still counts as "touching" [mm].  Not zero: several
# geom PAIRS in this model declare their own collision margin (the wrist brace
# pad against the table is the one that matters here), so MuJoCo reports those
# contacts at a small positive distance even on a model loaded with no inflation.
PLACE_TOL = 2.0


def enumerate_modes(target, sites=SITES5, verbose=True, subsets=None,
                    press=True):
    """Certify contact modes at one target.

    `subsets`, if given, replaces the power-set enumeration with an explicit
    list -- which is what croco_sweep.py hands in.  A (mode x target) matrix
    wants a CURATED ladder (see that module for the physical argument), and
    running 32 IK+QP solves per target to keep 5 of them is 27 solves of nothing.
    """
    if subsets is None:
        subsets = [s for k in range(len(sites) + 1)
                   for s in itertools.combinations(sites, k)]
    out = []
    for sub in subsets:
        m, d = cs.load()
        P = cs.solve_ik(m, d, np.asarray(target, float), sub)
        if press:
            m, d, P = press_contacts(m, d, target, sub, P)
        r = cs.equilibrium_qp(m, d, sub)
        gaps = contact_gaps(m, d, sub)
        # Admissibility, with the placement test REPLACED by the narrowphase one
        # (see contact_gaps).  Everything else is sweep5_full's, unchanged: the
        # hand is on target, the feet held, nothing through the slab, the base
        # balances, and the torques are inside the clamp basis.
        placed = all(v <= PLACE_TOL for v in gaps.values())
        adm = bool(P["reach"] < 0.03 and P["foot"] < 0.02
                   and P["penetration"] < 0.01 and placed
                   and r["base_residual"] < 1.0 and r["max_ratio"] <= 1.0)
        rec = dict(subset=list(sub), admissible=adm,
                   reach=float(P["reach"]), penetration=float(P["penetration"]),
                   placed=bool(placed), placed_by_body=bool(P["all_placed"]),
                   base_res=float(r["base_residual"]),
                   max_ratio=float(r["max_ratio"]), effort=float(r["effort"]),
                   brace_force=float(r["brace_force"]),
                   n_arm=len([t for t in sub if t in cs.ARM_SITES]),
                   pressed=bool(press),
                   contact_gap_mm=gaps,
                   qpos=d.qpos.copy())
        out.append(rec)
        if verbose:
            tag = "OK " if adm else "-- "
            gaps = " ".join(f"{s}:" + ("none" if v == NO_CONTACT else f"{v:+.1f}")
                            for s, v in rec["contact_gap_mm"].items())
            print(f"  {tag}{str(sub):32s} effort={r['effort']:7.3f} "
                  f"max|tau|/lim={r['max_ratio']:5.2f} reach={P['reach']*1000:6.1f}mm "
                  f"brace={r['brace_force']:6.1f}N  gap[mm] {gaps}", flush=True)
    return out


def contact_gaps(m, d, subset, radius=SITE_RADIUS):
    """Depth of the real table contact each requested site is actually making.

    NOT a body-identity test, and that is the correction.  `solve_ik`'s own
    `sites_placed` asks whether the SITE'S BODY is in the table's contact list,
    and for two of the three arm sites that body is the wrong one:

      palm     the site hangs off `left_magpie_gripper`, but the part of the hand
               assembly that reaches the wood first is the WRIST housing and its
               brace pad on `left_wrist_yaw_link`.  Measured over the sweep, a
               palm brace lands 31-44 mm from the palm site, on the parent link,
               every single time -- so a body test reports "palm: not touching"
               for a hand that is visibly flat on the table.
      elbow    the site is the elbow JOINT on the upper-arm link; the upper arm
               touches on its capsule surface, ~47 mm away (S11 measured this).

    So a site counts as placed when MuJoCo reports a table contact within
    `radius` of it, and each contact is assigned to its NEAREST site first, so a
    single contact cannot certify two sites at once -- which matters here because
    the elbow and forearm sites are only 57 mm apart.

    Returns {site: depth in mm}, negative for touching, NO_CONTACT for none.
    """
    mujoco.mj_forward(m, d)
    tbl = cs.bid(m, "table")
    sp = {s: cs.point_world(m, d, *cs.SITES[s]) for s in subset}
    out = {s: NO_CONTACT for s in subset}
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if tbl not in (b1, b2):
            continue
        rb = b2 if b1 == tbl else b1
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb) == "object":
            continue
        near, dist = None, radius
        for s, p in sp.items():
            dd = float(np.linalg.norm(con.pos - p))
            if dd < dist:
                near, dist = s, dd
        if near is None:
            continue
        depth = round(1000.0 * float(con.dist), 2)
        if out[near] == NO_CONTACT or depth < out[near]:
            out[near] = depth
    return out


def press_contacts(m, d, target, subset, P, iters=60, chunk=2):
    """Re-solve the IK against MuJoCo's REAL narrowphase, warm-started from P.

    WHY THIS IS NEEDED.  `contact_select.load()` inflates every geom's collision
    margin by IK_MARGIN = 25 mm so the IK can see a collision before it enters
    one, and `collision_rows` then drives every contact inside that band to
    dist = 0.  Those two together are supposed to snap a brace site onto the
    table -- but they are a soft term among several, and the compromise the step
    QP settles on routinely leaves the site a few millimetres clear.  MuJoCo at
    the inflated margin still calls that a contact, so `all_placed` is satisfied
    and the pose is certified.  MuJoCo at the REAL margin does not, and neither
    does the replay.

    Measured over the (mode x target) sweep before this existed: of 16 certified
    poses, 9 had at least one requested site with no real contact, and two -- both
    elbow+forearm+palm -- had none of the three touching.  Those are exactly the
    cells whose closed-loop replays collapsed with 0 N of brace force, which is
    the S12 "leaning on thin air" failure reappearing one stage upstream, in the
    certification rather than in the plan.

    The fix is not a new objective: it is the same IK on a model whose margin is
    ZERO, warm-started from the pose the first pass found.  The site z-rows keep
    pulling the site down; the collision rows now fire only on real contact and
    stop it at dist = 0.

    IT IS RUN IN SHORT CHUNKS AND STOPPED AT THE FIRST GOOD POSE, and that is not
    a performance tweak.  The site sits on the link AXIS, so "site z = table z"
    asks for the link half a diameter INSIDE the slab, and the collision rows are
    the only thing opposing it.  Pressed to convergence, the three-contact mode
    buries the forearm 15-17 mm in the wood and the single-elbow mode buries the
    upper arm 47 mm -- trading a missing contact for an inadmissible penetration.
    So the pose is scored BEFORE any pressing (one that already touches is left
    alone, which is why the S13 reference pose is unchanged by this) and after
    every 2-iteration chunk, on the criteria that actually matter: every site
    touching, nothing deeper than 10 mm.  The first pose that satisfies them wins;
    failing that the shallowest placed pose does; failing that the last iterate is
    returned and the caller's admissibility test rejects it.
    """
    m0, d0 = cs.load(ik_margin=0.0)
    d0.qpos[:] = d.qpos
    mujoco.mj_forward(m0, d0)
    if not subset:
        return m0, d0, P

    def score():
        pr = dict(P)
        pr["reach"] = float(np.linalg.norm(
            np.asarray(target, float)
            - cs.point_world(m0, d0, cs.REACH_BODY, cs.REACH_OFF)))
        g = contact_gaps(m0, d0, subset)
        placed = all(v <= PLACE_TOL for v in g.values())
        deepest = -min([v for v in g.values() if v != NO_CONTACT] or [0.0]) / 1000.0
        return pr, placed, max(deepest, 0.0)

    # Score the pose we were handed FIRST: pressing a pose that already touches
    # can only push it into the table.
    P0, placed0, deep0 = score()
    P0["penetration"] = max(P.get("penetration", 0.0), deep0)
    if placed0 and P0["penetration"] < 0.01:
        return m0, d0, P0
    best = (d0.qpos.copy(), P0, P0["penetration"]) if placed0 else None
    Pk = P0
    for _ in range(max(1, iters // chunk)):
        Pk = cs.solve_ik(m0, d0, np.asarray(target, float), subset, iters=chunk)
        pr, placed, deepest = score()
        Pk["penetration"] = max(Pk["penetration"], deepest)
        if placed and Pk["penetration"] < 0.01:
            return m0, d0, Pk
        if placed and (best is None or Pk["penetration"] < best[2]):
            best = (d0.qpos.copy(), Pk, Pk["penetration"])
    if best is not None:
        d0.qpos[:] = best[0]
        mujoco.mj_forward(m0, d0)
        return m0, d0, best[1]
    return m0, d0, Pk


def rank(records):
    adm = [r for r in records if r["admissible"]]
    return sorted(adm, key=lambda r: (round(r["effort"], 4), len(r["subset"])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, nargs=3,
                    default=[0.9047, -0.2348, 1.0982],
                    help="reach target; default is the session-12 target")
    ap.add_argument("--out", default="runs/2026-08-05_session12/croco")
    ap.add_argument("--stance-dy", type=float, default=None,
                    help="rigid y offset of the whole robot [m] before the IK")
    ap.add_argument("--stance-dx", type=float, default=None)
    args = ap.parse_args()
    if args.stance_dx is not None:
        cs.STANCE_DX = args.stance_dx
    if args.stance_dy is not None:
        cs.STANCE_DY = args.stance_dy

    os.makedirs(args.out, exist_ok=True)
    print(f"target {args.target}   sites {SITES5}   ({2**len(SITES5)} subsets)\n")
    recs = enumerate_modes(args.target)
    order = rank(recs)

    print(f"\n{len(order)} admissible of {len(recs)}")
    for i, r in enumerate(order[:8]):
        print(f"  {i}. {'+'.join(r['subset']) or 'legs_only':28s} "
              f"effort={r['effort']:.3f}  max_ratio={r['max_ratio']:.2f}")

    manifest = {"target": list(map(float, args.target)),
                "stance_dx": cs.STANCE_DX, "stance_dy": cs.STANCE_DY,
                "sites": list(SITES5),
                "model": os.path.basename(cs.MODEL),
                "brace_arm": cs.BRACE_ARM, "site_set": cs.SITE_SET,
                "seed_key": cs.SEED_KEY, "tau_basis": cs.TAU_BASIS,
                "commit": _git_head(), "pressed": True, "modes": []}
    for r in recs:
        name = "+".join(r["subset"]) or "legs_only"
        entry = {k: v for k, v in r.items() if k != "qpos"}
        entry["name"] = name
        if r["admissible"]:
            f = f"q_{name}.txt"
            np.savetxt(os.path.join(args.out, f), r["qpos"])
            entry["qpos_file"] = f
        manifest["modes"].append(entry)
    manifest["ranked"] = ["+".join(r["subset"]) or "legs_only" for r in order]
    with open(os.path.join(args.out, "modes.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    print(f"\nwrote {args.out}/modes.json")


def _git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
