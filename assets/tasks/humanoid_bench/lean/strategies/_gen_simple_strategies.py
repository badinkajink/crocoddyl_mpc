#!/usr/bin/env python3
"""Generate the standalone "simple task" strategies for the Lean H12 task.

Each simple task is one indefinite-hold phase selectable from the MJPC
Strategy slider (see GetStrategyNames in lean.h). Pose tasks name a model
<key> keyframe that the Posture/Control costs track (see the pose-library
block in lean.cc). This script:

  1. validates every pose target against the real H1-2 joint limits +
     self-collision + CoM-over-feet sanity (loads the vendored model),
  2. emits the <key .../> XML lines to paste into Lean_H12.xml and
     Lean_H12_Hands.xml,
  3. writes all strategy JSON files (both variants) into this directory.

Re-run after editing POSES to regenerate. Author: simple-task bring-up.
"""
import json, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.normpath(os.path.join(
    HERE, "../../../../../build/_deps/h12-src/unitree_robots/h1_2"))

# ----- qpos layout (verified against the compiled model) -------------------
# index of each actuated joint within qpos, per variant.
LEG_TORSO = {
    "left_hip_yaw": 7, "left_hip_pitch": 8, "left_hip_roll": 9,
    "left_knee": 10, "left_ankle_pitch": 11, "left_ankle_roll": 12,
    "right_hip_yaw": 13, "right_hip_pitch": 14, "right_hip_roll": 15,
    "right_knee": 16, "right_ankle_pitch": 17, "right_ankle_roll": 18,
    "torso": 19,
}
LEFT_ARM = {"left_shoulder_pitch": 20, "left_shoulder_roll": 21,
            "left_shoulder_yaw": 22, "left_elbow": 23, "left_wrist_roll": 24,
            "left_wrist_pitch": 25, "left_wrist_yaw": 26}
RIGHT_ARM_HANDLESS = {"right_shoulder_pitch": 27, "right_shoulder_roll": 28,
                      "right_shoulder_yaw": 29, "right_elbow": 30,
                      "right_wrist_roll": 31, "right_wrist_pitch": 32,
                      "right_wrist_yaw": 33}
RIGHT_ARM_HANDS = {k: v + 12 for k, v in RIGHT_ARM_HANDLESS.items()}  # +12 DOF L-hand
IDX_HANDLESS = {**LEG_TORSO, **LEFT_ARM, **RIGHT_ARM_HANDLESS}
IDX_HANDS = {**LEG_TORSO, **LEFT_ARM, **RIGHT_ARM_HANDS}

# home qpos for each variant (base, joints..., object free joint)
def home_handless():
    q = [0.0] * 41
    q[0:7] = [0.19, 0, 1.028, 1, 0, 0, 0]
    q[34:41] = [1.5, 0, 0.73, 1, 0, 0, 0]
    return q
def home_hands():
    q = [0.0] * 65
    q[0:7] = [0.19, 0, 1.028, 1, 0, 0, 0]
    q[58:65] = [1.5, 0, 0.73, 1, 0, 0, 0]
    return q

# ----- the validated pose targets (joint -> radians) -----------------------
POSES = {
    "crouch": {"left_hip_pitch": -0.6, "right_hip_pitch": -0.6,
               "left_knee": 1.0, "right_knee": 1.0,
               "left_ankle_pitch": -0.4, "right_ankle_pitch": -0.4},
    "arms_sideways": {"left_shoulder_roll": 1.5, "right_shoulder_roll": -1.5},
    "arms_forward": {"left_shoulder_pitch": -1.5, "right_shoulder_pitch": -1.5},
    # right_shoulder_pitch range is mirrored ([-1.57,3.14]); +2.8 raises the
    # right arm overhead (via the back), mirroring left's -2.8 (via the front).
    "arms_overhead": {"left_shoulder_pitch": -2.8, "right_shoulder_pitch": 2.8,
                      "left_shoulder_roll": 0.25, "right_shoulder_roll": -0.25},
    "single_arm_raise": {"right_shoulder_roll": -1.5},
    "lean_left": {"left_hip_roll": -0.30, "right_hip_roll": -0.30},
    "lean_right": {"left_hip_roll": 0.30, "right_hip_roll": 0.30},
    "torso_twist": {"torso": 0.6},
}

# ----- weight maps (keys MUST match the <user> cost-sensor names) -----------
POSE_HOLD_BASE = {
    "humanoid_bench": 0.0, "Object Dist": 0.0, "Reaching Hand Dist": 0.0,
    "Brace Pos": 0.0, "Brace Force": 0.0, "Contact": 0.0,
    "Left Leg Anchor": 0.0, "Right Foot Lift": 0.0, "Pelvis Forward": 0.0,
    "Support Polygon": 0.0, "Body Yaw": 0.0, "Body Table Clearance": 0.0,
    "Hip Clearance": 0.0, "Leg Clearance": 0.0, "Torso Forward Tilt": 0.0,
    "Posture": 6.0,
}
def merged(**ovr):
    w = dict(POSE_HOLD_BASE); w.update(ovr); return w

POSE_WEIGHTS = {
    "crouch":          merged(**{"Height": 0.0, "Knees Straight": 0.0, "Posture": 8.0}),
    "arms_sideways":   merged(Posture=8.0),
    "arms_forward":    merged(Posture=8.0),
    "arms_overhead":   merged(Posture=8.0),
    "single_arm_raise": merged(Posture=8.0),
    "torso_twist":     merged(**{"Waist Yaw": 0.0, "Posture": 8.0}),
    # Balance cost centres the CoM, which directly opposes a weight-shift, so
    # drop it low (1.5) and boost Posture so the hip-roll lean develops. Feet
    # stay flat (no ankle roll) so Foot Up is left high and doesn't fight.
    "lean_left":       merged(**{"Hip Roll L": 0.0, "Hip Roll R": 0.0, "Balance": 1.5, "Posture": 12.0}),
    "lean_right":      merged(**{"Hip Roll L": 0.0, "Hip Roll R": 0.0, "Balance": 1.5, "Posture": 12.0}),
}
# non-pose tasks (track home key 0 / use existing reach machinery)
SPECIAL = {
    # stand = pure upright hold: zero ALL lean/table/brace costs (else it inherits
    # the lean task defaults incl. Object Dist -> drags the robot toward a table
    # that isn't in the twin scene). Same cost map as arms_forward, home pose.
    "stand": {"name": "stand_up", "weight": merged(Posture=8.0)},
    "reach_forward": {"name": "arm_extend_standing",
                      "weight": {"Pelvis Tilt": 300.0, "Torso Forward Tilt": 0.0,
                                 "Foot Stability": 60.0, "Object Dist": 80.0,
                                 "Reaching Hand Dist": 40.0}},
}

EMPTY_CONTACTS = [{"body1": -1, "body2": -1, "geom1": -1, "geom2": -1,
                   "local_pos1": [0.0, 0.0, 0.0], "local_pos2": [0.0, 0.0, 0.0]}
                  for _ in range(5)]

def strategy_json(phase_name, weight):
    return [{"contacts": EMPTY_CONTACTS, "facing_target": [], "name": phase_name,
             "success_sustain_time": 9999.0, "target_distance_tolerance": 0.3,
             "time_limit": 9999.0, "weight": weight, "brace_force_target": 0.0}]

# ----- validate poses against the real model -------------------------------
def validate():
    m = mujoco.MjModel.from_xml_path(os.path.join(VENDOR, "h1_2_pos.xml"))
    d = mujoco.MjData(m)
    def qadr(j):
        return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j + "_joint")]
    def jr(j):
        return m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j + "_joint")]
    def bp(b):
        return d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)].copy()
    ok = True
    for name, jd in POSES.items():
        mujoco.mj_resetDataKeyframe(m, d, 0)
        for j, a in jd.items():
            lo, hi = jr(j)
            if not (lo - 1e-6 <= a <= hi + 1e-6):
                print(f"  !! {name}: {j}={a} out of [{lo:.2f},{hi:.2f}]"); ok = False
            d.qpos[qadr(j)] = a
        mujoco.mj_forward(m, d)
        com = d.subtree_com[0]
        Lf, Rf = bp("left_ankle_roll_link"), bp("right_ankle_roll_link")
        dL = np.hypot(com[0]-Lf[0], com[1]-Lf[1])
        dR = np.hypot(com[0]-Rf[0], com[1]-Rf[1])
        flag = ""
        if name == "lean_left" and not dL < dR - 0.03: flag = " !! expected weight LEFT"; ok = False
        if name == "lean_right" and not dR < dL - 0.03: flag = " !! expected weight RIGHT"; ok = False
        if d.ncon > 4: flag += f" !! ncon={d.ncon}"; ok = False
        print(f"  {name:16s} ncon={d.ncon} CoM=({com[0]:+.2f},{com[1]:+.2f},{com[2]:.2f}) "
              f"dL={dL:.2f} dR={dR:.2f}{flag}")
    return ok

# ----- emit keyframes + JSON ----------------------------------------------
def keyline(name, idxmap, home):
    q = list(home)
    for j, a in POSES[name].items():
        q[idxmap[j]] = a
    vals = " ".join((f"{v:g}") for v in q)
    return f'        <key name="{name}" qpos="{vals}"/>'

def main():
    print("== validating poses =="); ok = validate()
    if not ok:
        print("VALIDATION FAILED"); sys.exit(1)
    print("== all poses valid ==\n")

    # keyframe XML
    kh = "\n".join(keyline(n, IDX_HANDLESS, home_handless()) for n in POSES)
    kn = "\n".join(keyline(n, IDX_HANDS, home_hands()) for n in POSES)
    open(os.path.join(HERE, "_keys_handless.xml.txt"), "w").write(kh + "\n")
    open(os.path.join(HERE, "_keys_hands.xml.txt"), "w").write(kn + "\n")
    print("wrote _keys_handless.xml.txt + _keys_hands.xml.txt")

    # JSON strategies, both variants
    written = []
    for prefix in ("h12_simple", "h12_hands_simple"):
        for task, spec in SPECIAL.items():
            fn = f"{prefix}_{task}.json"
            json.dump(strategy_json(spec["name"], spec["weight"]),
                      open(os.path.join(HERE, fn), "w"), indent=4)
            written.append(fn)
        for task in POSES:
            fn = f"{prefix}_{task}.json"
            json.dump(strategy_json(task, POSE_WEIGHTS[task]),
                      open(os.path.join(HERE, fn), "w"), indent=4)
            written.append(fn)
    print(f"wrote {len(written)} strategy JSONs")

if __name__ == "__main__":
    main()
