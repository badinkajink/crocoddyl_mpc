#!/usr/bin/env python3
"""Parity audit of the built H1-2 task models.

Run after any asset change (CL_Assets bump, magpie patch edit, merge). Checks the
three things that the build itself cannot check, because the magpie patch applies
with FUZZ and `copy_directory` never removes a retired file:

  1. every H1-2 task model still LOADS;
  2. no keyframe sits outside its joint's range -- the failure mode the CL_Assets
     limit import creates, since it tightens ranges under keyframes tuned against
     the retired envelope;
  3. the EFFECTIVE actuator torque bound, per model, against the three competing
     authorities. MuJoCo enforces min(<position forcerange>, jnt_actfrcrange), so
     the imported CL number is NOT necessarily the one the planner sees.

Torque authorities, all real, all different (see h12_control_node.cc):
  TAU_LIMIT  = URDF actuatorfrcrange = CL_Assets = what the limit import writes
  TAU_ESTOP  = safety-layer trip threshold, BELOW the URDF limit
  0.9*ESTOP  = the H2 command clamp the deployed node actually enforces
The planner should never be able to command torque the robot's clamp will refuse,
so 0.9*TAU_ESTOP is the honest planning basis. Allen applied exactly that to the
ankle (48.6 = 0.9 x 54); the arms have not had it applied.

usage: parity_audit.py [build/mjpc/tasks/humanoid_bench]
"""
import os
import sys
import glob

import numpy as np
import mujoco

ROOT = (sys.argv[1] if len(sys.argv) > 1 else
        "/home/humanoid/Programs/mjpc_icra2026/build/mjpc/tasks/humanoid_bench")

# h12_control_node.cc, motor order qpos[7..33]. Legs then torso then arms x2.
JOINTS = ["hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
       "wrist_roll", "wrist_pitch", "wrist_yaw"]
TAU_ESTOP = dict(zip(JOINTS, [60, 130, 200, 300, 54, 36]))
TAU_ESTOP["torso"] = 40
TAU_ESTOP.update(dict(zip(ARM, [32, 32, 14.4, 14.4, 9.5, 9.5, 9.5])))
TAU_LIMIT = dict(zip(JOINTS, [200, 200, 200, 300, 60, 40]))
TAU_LIMIT["torso"] = 200
TAU_LIMIT.update(dict(zip(ARM, [40, 40, 18, 18, 19, 19, 19])))
CLAMP = 0.9  # deploy_common.h kClampRatio


def base_joint(name):
    """'right_wrist_pitch_joint' -> 'wrist_pitch'; 'torso_joint' -> 'torso'."""
    n = name.replace("_joint", "")
    for side in ("left_", "right_"):
        if n.startswith(side):
            n = n[len(side):]
    return n


def audit(path):
    try:
        m = mujoco.MjModel.from_xml_path(path)
    except Exception as exc:
        return {"path": path, "load_error": str(exc).split("\n")[0]}
    d = mujoco.MjData(m)

    # ---- keyframes vs joint ranges ------------------------------------------
    bad = []
    for k in range(m.nkey):
        kname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_KEY, k) or str(k)
        q = m.key_qpos[k]
        for j in range(m.njnt):
            if not m.jnt_limited[j] or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            v = q[m.jnt_qposadr[j]]
            lo, hi = m.jnt_range[j]
            if v < lo - 1e-9 or v > hi + 1e-9:
                jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or str(j)
                bad.append((kname, jn, float(v), float(lo), float(hi)))

    # ---- effective torque bound vs the three authorities --------------------
    tau = {}
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        j = m.actuator_trnid[i, 0]
        fr = m.actuator_forcerange[i, 1] if m.actuator_forcelimited[i] else np.inf
        ja = m.jnt_actfrcrange[j, 1] if m.jnt_actfrclimited[j] else np.inf
        tau[base_joint(n)] = min(fr, ja)

    over = []
    for jn, eff in sorted(tau.items()):
        if jn not in TAU_ESTOP:
            continue
        clamp = CLAMP * TAU_ESTOP[jn]
        if eff > clamp + 1e-6:
            over.append((jn, eff, clamp, TAU_LIMIT[jn]))

    # ---- table, if this model has one ---------------------------------------
    table = None
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    if g >= 0:
        mujoco.mj_forward(m, d)
        table = (float(d.geom_xpos[g][2] + m.geom_size[g][2]),
                 float(d.geom_xpos[g][0] - m.geom_size[g][0]),
                 float(m.geom_size[g][2]))
    return {"path": path, "nq": m.nq, "nu": m.nu, "bad_keys": bad,
            "over_clamp": over, "table": table}


def main():
    models = sorted(glob.glob(os.path.join(ROOT, "*", "*_H12*.xml")))
    models = [p for p in models if "h1_2/" not in p]
    if not models:
        sys.exit("no H12 task models under %s -- build copy_model_resources first" % ROOT)

    n_bad = n_load = 0
    print("== load + keyframe audit ==")
    for p in models:
        r = audit(p)
        rel = os.path.relpath(p, ROOT)
        if "load_error" in r:
            n_load += 1
            print("  FAIL  %-44s %s" % (rel, r["load_error"]))
            continue
        if r["bad_keys"]:
            n_bad += len(r["bad_keys"])
            print("  BAD   %-44s %d out-of-range keyframe entries" % (rel, len(r["bad_keys"])))
            for kn, jn, v, lo, hi in r["bad_keys"][:6]:
                print("          %-16s %-28s %+.4f not in [%+.4f, %+.4f]" % (kn, jn, v, lo, hi))
        else:
            t = ""
            if r["table"]:
                t = "  table surf z=%.3f near_x=%.3f halfthick=%.3f" % r["table"]
            print("  ok    %-44s nq=%-3d nu=%-3d%s" % (rel, r["nq"], r["nu"], t))

    print("\n== effective torque bound vs the deployed 0.9 x TAU_ESTOP clamp ==")
    print("   (a joint listed here can be commanded torque the real node will refuse)")
    seen = {}
    for p in models:
        r = audit(p)
        if "load_error" in r:
            continue
        for jn, eff, clamp, urdf in r["over_clamp"]:
            seen.setdefault(jn, (eff, clamp, urdf, []))[3].append(os.path.relpath(p, ROOT))
    if not seen:
        print("   none -- every model is inside the clamp")
    else:
        print("   %-16s %9s %9s %9s   %s" % ("joint", "model", "0.9xestop", "urdf", "n_models"))
        for jn, (eff, clamp, urdf, ps) in sorted(seen.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
            print("   %-16s %9.2f %9.2f %9.2f   %d  (%.0f%% over)"
                  % (jn, eff, clamp, urdf, len(ps), 100 * (eff / clamp - 1)))

    print("\n%d load failures, %d out-of-range keyframe entries, %d joints over clamp"
          % (n_load, n_bad, len(seen)))
    return 1 if (n_load or n_bad) else 0


if __name__ == "__main__":
    sys.exit(main())
