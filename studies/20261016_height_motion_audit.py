#!/usr/bin/env python3
"""Measure archived height-rollout speeds without advancing the simulator.

Run one height per process, with the same TABLE_H / LEAN_TASK_DIR as its video.
Joint acceleration is an adjacent-sample finite difference of logged velocity;
point velocity is J(q) v, evaluated on the replayed model. Neither is a new run.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--face", type=float, required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.environ["TABLE_H"] = str(args.face)
    import mujoco
    import contact_select as cs

    source = Path(args.csv)
    with source.open() as f:
        reader = csv.reader(line for line in f if not line.startswith("#"))
        names = next(reader)
        data = np.array(list(reader), dtype=float)
    col = {name: i for i, name in enumerate(names)}
    t = data[:, col["time"]]
    model, state = cs.load(ik_margin=0)
    q = data[:, [col[f"qpos{i}"] for i in range(model.nq)]]
    v = data[:, [col[f"qvel{i}"] for i in range(model.nv)]]
    motor_joints = model.actuator_trnid[:, 0]
    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(j))
                   for j in motor_joints]
    dofs = model.jnt_dofadr[motor_joints]
    motor_v = v[:, dofs]

    def peak(values, stamps, units):
        k, j = np.unravel_index(np.abs(values).argmax(), values.shape)
        return dict(value=float(values[k, j]), absolute=float(abs(values[k, j])),
                    time_s=float(stamps[k]), joint=joint_names[j], units=units)

    speed = peak(motor_v, t, "rad/s")
    accel = peak(np.diff(motor_v, axis=0) / np.diff(t)[:, None],
                 (t[1:] + t[:-1]) / 2, "rad/s^2 over a 10 ms sample interval")
    arms = np.array([("shoulder" in n or "elbow" in n or "wrist" in n)
                     for n in joint_names])
    arm_speed = peak(motor_v * arms[None, :], t, "rad/s")
    points = {"reaching_hand": (cs.REACH_BODY, cs.REACH_OFF),
              "brace_wrist": ("left_wrist_yaw_link", np.zeros(3))}
    positions = {key: [] for key in points}
    velocities = {key: [] for key in points}
    table = cs.bid(model, "table")
    first_touch = None
    jp = np.zeros((3, model.nv))
    jr = np.zeros((3, model.nv))
    for k in range(len(t)):
        state.qpos[:] = q[k]
        mujoco.mj_kinematics(model, state)
        mujoco.mj_comPos(model, state)
        for key, (body, offset) in points.items():
            body_id = cs.bid(model, body)
            point = cs.point_world(model, state, body, offset)
            mujoco.mj_jac(model, state, jp, jr, point, body_id)
            positions[key].append(point.copy())
            velocities[key].append(jp @ v[k])
        if first_touch is None:
            mujoco.mj_collision(model, state)
            for contact in state.contact:
                bodies = model.geom_bodyid[contact.geom]
                if table not in bodies or contact.dist > 0:
                    continue
                other = int(bodies[1] if bodies[0] == table else bodies[0])
                body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other)
                if body_name and (body_name.startswith("left_") or body_name == "torso_link"):
                    first_touch = dict(time_s=float(t[k]), body=body_name,
                                       separation_m=float(contact.dist))
                    break

    point_metrics = {}
    for key in points:
        vel = np.asarray(velocities[key])
        norm = np.linalg.norm(vel, axis=1)
        k = int(norm.argmax())
        # Independent pose finite differences catch velocity-frame/index mistakes.
        position_fd = np.diff(positions[key], axis=0) / np.diff(t)[:, None]
        midpoint_v = (vel[1:] + vel[:-1]) / 2
        point_metrics[key] = dict(peak_speed_m_s=float(norm[k]), time_s=float(t[k]),
            velocity_pose_difference_rms_m_s=float(np.sqrt(np.mean((position_fd - midpoint_v)**2))))
    plan_path = source.parent / "cell" / ("plan_stand.json" if "stand" in source.stem
                                           else "plan_elbow_forearm.json")
    plan = json.loads(plan_path.read_text())
    video = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "stream=duration,nb_frames,avg_frame_rate",
        "-of", "json", args.video], text=True))["streams"][0]
    result = dict(source_csv=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_plan=str(plan_path), model=str(cs.MODEL),
        model_xml_sha256=hashlib.sha256(Path(cs.MODEL).read_bytes()).hexdigest(),
        table_face_m=args.face, samples=len(t), simulation_end_s=float(t[-1]),
        sample_interval_s=float(np.median(np.diff(t))), old_video=args.video, old_video_probe=video,
        old_frame_stride=10, old_playback_speed=3,
        approach_s=plan["n_approach"] * plan["dt"],
        nominal_plan_s=(plan["n_approach"] + plan["n_braced"] + plan["n_return"]) * plan["dt"],
        additional_velocity_cost_weight=plan.get("w_vel"),
        peak_motor_velocity=speed, peak_arm_velocity=arm_speed, peak_motor_acceleration=accel,
        peak_base_translation_speed_m_s=float(np.linalg.norm(v[:, :3], axis=1).max()),
        points=point_metrics, first_reconstructed_table_overlap=first_touch,
        limitations="100 Hz archived samples may miss faster peaks. Point kinematics and first overlap use the current staged model, not archived collision telemetry. Acceleration is a finite difference; no actuator bandwidth or hardware tracking validation.")
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"csv": source.name, "motor_speed": speed, "arm_speed": arm_speed,
                      "acceleration": accel, "points": point_metrics, "first_touch": first_touch}))


if __name__ == "__main__":
    main()
