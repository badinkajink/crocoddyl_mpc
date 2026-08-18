#!/usr/bin/env python3
"""Does the torso ceiling actually bind?

The node's TAU_ESTOP carries 40 Nm for the torso and that traces to no safety
config; default_safety_full's estop is 1.0 x URDF = 200.  Before adopting 200
it is worth knowing what it BUYS, because a limit that never binds changes no
result no matter which number is right.

Re-solves the braced pose under both torso ceilings and prints the loaded joints
in rank order, so "the torso is the joint that carries a lean" is checked rather
than assumed.

usage: torso_basis.py [--target x,y,z]
"""
import argparse
import json
import os

import numpy as np
import mujoco

import contact_select as cs
import stability as st

SUBSETS = [("elbow", "forearm"), ("elbow", "forearm", "palm"), ("palm",)]


def solve(target, subset, torso_tau):
    cs.TAU_ESTOP["torso"] = torso_tau
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.asarray(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    _, _, marg = st.equilibrium_region(m, d, subset, actuated=True)
    tau_max = cs.torque_limits(m)
    tau = np.asarray(qp["tau"])
    ratio = np.abs(tau) / tau_max
    names = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "?")
             .replace("_joint", "") for i in range(m.nu)]
    order = np.argsort(-ratio)[:6]
    ti = names.index("torso")
    return dict(reach=float(ik["reach"]), feasible=bool(qp["feasible"]),
                peak=float(ratio.max()), margin=float(marg),
                torso_tau_nm=float(abs(tau[ti])),
                torso_ratio=float(ratio[ti]),
                torso_ceiling=float(tau_max[ti]),
                top=[(names[i], float(abs(tau[i])), float(tau_max[i]),
                      float(ratio[i])) for i in order])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="")
    ap.add_argument("--out", default="runs/torso_basis.json")
    a = ap.parse_args()

    m0, _ = cs.load(ik_margin=0)
    if a.target:
        target = [float(v) for v in a.target.split(",")]
    else:
        nid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
        target = [float(v) for v in
                  m0.numeric_data[m0.numeric_adr[nid]:m0.numeric_adr[nid] + 3]]

    out = {}
    for subset in SUBSETS:
        key = "+".join(subset)
        out[key] = {}
        print("\n=== %s ===" % key)
        for label, estop in (("node 40 Nm", 40.), ("default_safety_full 200 Nm", 200.)):
            r = solve(target, subset, estop)
            out[key][label] = r
            print("  %-28s ceiling %5.1f Nm | peak %.3f  margin %.4f  "
                  "torso %.1f Nm (%.0f%% of its own ceiling)"
                  % (label, r["torso_ceiling"], r["peak"], r["margin"],
                     r["torso_tau_nm"], 100 * r["torso_ratio"]))
            print("       most-loaded: " + ", ".join(
                "%s %.0f/%.0f=%.2f" % (n, t, tm, rr) for n, t, tm, rr in r["top"]))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(target=target, results=out), open(a.out, "w"),
              indent=1, default=float)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
