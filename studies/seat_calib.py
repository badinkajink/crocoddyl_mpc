#!/usr/bin/env python3
"""Calibrate the analytic seat measure used by the Lean Simple task.

The Lean Simple residual asks "how far is link L above the table top", one term
per candidate brace link, and saturates the term once the link is seated.  That
needs a distance function that is (a) cheap, (b) continuous while the link is
still in the air, and (c) ZERO exactly when MuJoCo starts reporting contact --
not before (the planner would stop short) and not after (it would demand
penetration it cannot get, which is the defect the S12 page diagnosed in
`Brace Pos`).

Two candidate distance functions were considered and one is disqualified here:

  mj_geomDistance      exact for primitives, WRONG for the arm's mesh geoms.
                       Measured at the shipped `forearm_brace_reach` keyframe,
                       lifting the whole robot vertically:

                           dz [mm]    upper-arm mesh   forearm mesh   pad capsule
                              0           -333.4          -419.6         -35.0
                             50           -290.9          -372.8          14.7
                            100           -233.0          -305.6          64.7
                            400           +170.2          +183.1         364.7

                       The capsule tracks the lift exactly (49.7 / 50 / 50 mm per
                       50 mm).  The two MESH columns are off by a third of a metre
                       at contact and do not even move linearly with the lift.
                       Since `left_shoulder_yaw_link` and `left_elbow_link` carry
                       NO primitive collision geom, a seat cost built on
                       mj_geomDistance over "the link's collidable geoms" reads
                       the mesh -- i.e. reads garbage -- for two of the three
                       candidate links.

  analytic capsule     min over a body-frame segment's endpoints of (z - r),
                       minus the slab top plane.  Orientation-correct for a
                       horizontal plane, exact for the pads that exist, and the
                       column above shows it is the one that tracks reality.

This script fits the per-link saturation offset for the analytic version by
translating a braced pose vertically and finding the height at which MuJoCo's
own narrowphase starts reporting link-vs-slab contact.

usage: seat_calib.py [--model PATH]
"""
import argparse
import os

import numpy as np
import mujoco

DEFAULT_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "build/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml")

# Candidate brace links, and the capsule that stands in for each link's brace
# surface.  elbow/forearm segments are the link axes; radii are the link's own
# collision proxy where the model ships one (`left_forearm_pad`, r=35 mm) and the
# mesh bounding-box half-width where it does not (upper arm, 42-49 mm -> 45).
# The palm is the one link with a real primitive: use its box corners.
LINKS = {
    "elbow": dict(body="left_shoulder_yaw_link",
                  seg=[(0.002, -0.007, -0.030), (0.002, -0.007, -0.182)],
                  r=0.045),
    "forearm": dict(body="left_elbow_link",
                    seg=[(0.020, -0.010, -0.015), (0.110, -0.030, -0.015)],
                    r=0.035),
    "palm": dict(body="left_magpie_gripper", box="left_gripper_collision"),
}


def slab(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    c, s = d.geom_xpos[g], m.geom_size[g]
    return c[2] + s[2], (c[0] - s[0], c[0] + s[0]), (c[1] - s[1], c[1] + s[1])


def measure(m, d, spec):
    """Height of the link's lowest brace-surface point above the slab top."""
    top, _, _ = slab(m, d)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
    R = d.xmat[b].reshape(3, 3)
    p = d.xpos[b]
    if "box" in spec:
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, spec["box"])
        gp, gR, gs = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
        zs = [(gp + gR @ (np.array(sx) * gs))[2]
              for sx in [(i, j, k) for i in (-1, 1) for j in (-1, 1)
                         for k in (-1, 1)]]
        return min(zs) - top
    zs = [(p + R @ np.array(o))[2] - spec["r"] for o in spec["seg"]]
    return min(zs) - top


def link_geoms(m, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]


def contacts_with_slab(m, d, body):
    gt = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    gs = set(link_geoms(m, body))
    n = 0
    for i in range(d.ncon):
        c = d.contact[i]
        if (c.geom1 == gt and c.geom2 in gs) or (c.geom2 == gt and c.geom1 in gs):
            if c.dist <= 0:
                n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--key", default="forearm_brace_reach")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(a.model)
    d = mujoco.MjData(m)
    kf = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, a.key)

    print(f"model {os.path.basename(a.model)}  key {a.key}")
    mujoco.mj_resetDataKeyframe(m, d, kf)
    mujoco.mj_forward(m, d)
    top, xr, yr = slab(m, d)
    print(f"slab top z = {top:.3f}  x {xr[0]:.3f}..{xr[1]:.3f}  "
          f"y {yr[0]:.3f}..{yr[1]:.3f}\n")

    print("  dz[mm]  " + "".join(f"{k:>22s}" for k in LINKS))
    print("          " + "".join(f"{'meas[mm]  ncon':>22s}" for k in LINKS))
    onset = {k: None for k in LINKS}
    for dz_mm in range(-40, 121, 5):
        mujoco.mj_resetDataKeyframe(m, d, kf)
        d.qpos[2] += dz_mm / 1000.0
        mujoco.mj_forward(m, d)
        row = f"  {dz_mm:6d}  "
        for k, spec in LINKS.items():
            mm = measure(m, d, spec) * 1000.0
            n = contacts_with_slab(m, d, spec["body"])
            if n == 0 and onset[k] is None:
                onset[k] = mm       # first height at which contact is LOST
            row += f"{mm:14.1f}{n:8d}"
        print(row)

    print("\nseat saturation (measure at the first height with NO contact,")
    print("i.e. the reading that corresponds to 'just touching'):")
    for k, v in onset.items():
        print(f"  {k:9s} d_sat = {v:7.1f} mm" if v is not None
              else f"  {k:9s} d_sat = (always in contact over the sweep)")


if __name__ == "__main__":
    main()
