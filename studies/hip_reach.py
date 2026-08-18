#!/usr/bin/env python3
"""Reach as a function of HIP ACTUATION, thresholded by CoM margin.

This replaces the target-grid reach envelope with the mechanism the user
identified, which is a better model of the task:

    contacts  ->  larger equilibrium region
              ->  more admissible hip flexion
              ->  CoM further forward, hand further out

Extra contacts do NOT lengthen the arm.  They raise the stability ceiling, and
hip flexion is what converts that ceiling into reach.  So reach is not the
independent variable -- hip pitch is, and reach is what you get, subject to the
margin you are willing to run at.

Method.  Hip pitch is COMMANDED (heavily weighted posture target on both hips,
not a hard clamp -- clamping it removes a DOF the pinned-foot task needs and the
feet drift 40 mm) and swept.  At each
angle the IK is given a deliberately unreachable target far out along +x, so the
reach task saturates and the hand goes as far as that hip angle allows; the
achieved hand x IS the max reach at that flexion.  Everything else -- feet
pinned, brace sites on the table, non-penetration -- is unchanged.  Then per
angle we record:

    reach_x   how far forward the hand actually got
    margin    actuated equilibrium-region margin at that pose
    peak      max |tau| / tau_max
    hip_tau   the hip-pitch torque itself, as a fraction of its clamped limit
    load      normal force through the brace contacts (what the flexion costs
              the arm, which is why load-balanced contact selection matters)

The deliverable is reach AT A CHOSEN MARGIN: for each contact set, the largest
hip flexion whose margin is still >= m*, and the reach that buys.  That is the
number the contact set should be judged on, and unlike the grid envelope it
cannot be censored by arm length -- if the arm runs out, reach_x simply stops
increasing and the curve says so.

usage: hip_reach.py [set,set,...]
       STANCE_DY=... FOREARM_HEADING=... TAU_BASIS=...
"""
import json
import os
import sys

import numpy as np
import mujoco

import contact_select as cs
import stability as st

SETS = [(), ("palm",), ("elbow", "forearm"), ("elbow", "forearm", "palm"),
        ("elbow", "forearm", "hip")]
# hip pitch is negative-forward on this robot; the seed sits at -1.14 rad
HIPS = np.round(np.arange(-0.70, -1.86, -0.10), 3)
THRESHOLDS = [0.0, 0.02, 0.05, 0.10]     # CoM margin [m] to report reach at
FAR_TARGET = 2.10                        # deliberately unreachable


def one(subset, hip, y, z):
    m, d = cs.load()
    soft = {"left_hip_pitch_joint": float(hip), "right_hip_pitch_joint": float(hip)}
    ik = cs.solve_ik(m, d, np.array([FAR_TARGET, y, z]), subset, soft_joints=soft)
    hand = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)

    placed = bool(ik.get("all_placed", True))
    pen_ok = ik["penetration"] < 0.01
    qp = cs.equilibrium_qp(m, d, subset)
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)

    tmax = cs.torque_limits(m)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    hi = [i for i, n in enumerate(names) if n and "hip_pitch" in n]
    hip_r = float(np.max(np.abs(qp["tau"][hi]) / tmax[hi]))

    nfoot = 8
    brace_fn = float(sum(qp["lam"][3 * (nfoot + k)] for k in range(len(subset))))
    ach = ik.get("achieved", {})
    hip_ach = float(np.mean(list(ach.values()))) if ach else float(hip)
    return dict(hip=float(hip), hip_achieved=hip_ach,
                reach_x=float(hand[0]), margin=float(marg),
                peak=float(qp["max_ratio"]), hip_tau_ratio=hip_r,
                brace_fn=brace_fn, feasible=bool(qp["feasible"]),
                placed=placed, pen_ok=bool(pen_ok), foot=float(ik["foot"]),
                valid=bool(placed and pen_ok and ik["foot"] < 0.02))


def main():
    y, z = -0.2348, 1.0982
    sets = ([tuple(s.split("+")) if s != "legs" else () for s in sys.argv[1].split(",")]
            if len(sys.argv) > 1 else SETS)

    print("hip-pitch sweep  stance_dy=%.2f heading=%s basis=%s site_set=%s"
          % (cs.STANCE_DY, cs.FOREARM_HEADING if cs.FOREARM_ALIGN else "off",
             cs.TAU_BASIS, cs.SITE_SET))
    out = {}
    for s in sets:
        name = "+".join(s) or "legs-only"
        print("\n== %s" % name)
        print("   %6s %8s %9s %7s %8s %8s %6s"
              % ("hip", "reach_x", "margin", "peak", "hip_tau", "brace_N", "valid"))
        rows = []
        for hip in HIPS:
            r = one(s, hip, y, z)
            rows.append(r)
            print("   %6.2f %8.3f %+9.4f %7.3f %8.3f %8.0f %6s"
                  % (r["hip_achieved"], r["reach_x"], r["margin"], r["peak"],
                     r["hip_tau_ratio"], r["brace_fn"],
                     "y" if (r["valid"] and r["feasible"]) else "-"), flush=True)

        good = [r for r in rows if r["valid"] and r["feasible"]]
        rep = {}
        for m_star in THRESHOLDS:
            elig = [r for r in good if r["margin"] >= m_star]
            best = max(elig, key=lambda r: r["reach_x"]) if elig else None
            rep["%.2f" % m_star] = (dict(reach_x=best["reach_x"], hip=best["hip"],
                                         margin=best["margin"], peak=best["peak"])
                                    if best else None)
            print("   reach at margin >= %.2f m : %s"
                  % (m_star, ("%.3f m (hip %.2f rad, margin %.3f)"
                              % (best["reach_x"], best["hip"], best["margin"]))
                     if best else "no admissible flexion"))
        out[name] = dict(rows=rows, at_threshold=rep)

    print("\n== reach [m] at each CoM-margin threshold ==")
    print("   %-24s %9s %9s %9s %9s" % ("set", "m>=0.00", "m>=0.02", "m>=0.05", "m>=0.10"))
    for s in sets:
        name = "+".join(s) or "legs-only"
        cells = []
        for t in THRESHOLDS:
            v = out[name]["at_threshold"]["%.2f" % t]
            cells.append("%.3f" % v["reach_x"] if v else "  --  ")
        print("   %-24s %9s %9s %9s %9s" % (name, *cells))

    dst = os.environ.get("OUT", "hip_reach.json")
    json.dump(dict(y=y, z=z, stance_dy=cs.STANCE_DY, basis=cs.TAU_BASIS,
                   thresholds=THRESHOLDS, sets=out), open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
