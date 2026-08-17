#!/usr/bin/env python3
"""Which wrist ROLL puts the gripper's flat SIDE face down, and what does the
wrist YAW motor do once it is there?

The user's proposal: stop eliciting a "palm" contact with wrist PITCH (range
+/-0.4625 rad, and it drives the gripper's jaw-separation axis at the table),
and instead roll the wrist ~90 deg so the gripper's flat side faces the table,
then drive it down with wrist YAW (range +/-1.27 rad).

This measures, over a roll sweep at the brace seed pose:
  - the world-frame direction of the gripper body's local axes
  - which local face is lowest (the one that will touch)
  - the world-frame direction the YAW joint axis points (lateral = it pitches
    the hand up/down, i.e. it can drive the gripper into the table)
  - the lowest point of the gripper collision box and of the jaws
"""
import numpy as np
import mujoco

import contact_select as cs

ARM = cs.BRACE_ARM
GRIP = "%s_magpie_gripper" % ARM
BOX_POS = np.array([0.0965, 0.0, 0.0])
BOX_HALF = np.array([0.0425, 0.0315, 0.0667])


def jnt(m, name):
    j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    return j, m.jnt_qposadr[j], m.jnt_dofadr[j]


def main():
    m, d = cs.load(ik_margin=0)
    jr, ar, dr = jnt(m, "%s_wrist_roll_joint" % ARM)
    jy, ay, dy = jnt(m, "%s_wrist_yaw_joint" % ARM)
    gb = cs.bid(m, GRIP)
    z_tab = cs.table_top_z(m, d)

    print("brace arm = %s   gripper body = %s   table top z = %.4f" % (ARM, GRIP, z_tab))
    print("gripper collision box: pos %s half-extents %s" % (BOX_POS, BOX_HALF))
    print("  local +-x = finger length, +-y = FLAT SIDE (thin, 63 mm), +-z = jaw separation\n")

    print("%7s | %-22s %-22s | %-16s | %8s %8s"
          % ("roll", "local +y in world", "local +z in world",
             "yaw axis (world)", "box low", "jaw low"))
    for roll in np.deg2rad([0, -30, -60, -90, -120, 90]):
        d.qpos[ar] = roll
        mujoco.mj_forward(m, d)
        R = d.xmat[gb].reshape(3, 3)          # columns: local axes in world
        ey, ez = R[:, 1], R[:, 2]
        # yaw joint axis in world
        yaxis = d.xaxis[jy] if hasattr(d, "xaxis") else None
        if yaxis is None:
            yaxis = R @ np.array([0., 0., 1.])
        # lowest point of the box: centre - sum |half_i * axis_i . zhat|
        c = d.xpos[gb] + R @ BOX_POS
        drop = sum(abs(BOX_HALF[i] * R[2, i]) for i in range(3))
        box_low = c[2] - drop
        jaw_low = 1e9
        for s in (+1, -1):
            jc = d.xpos[gb] + R @ np.array([0.1795, -0.0038, s * 0.0801])
            jd = sum(abs(np.array([0.0459, 0.008, 0.0261])[i] * R[2, i]) for i in range(3))
            jaw_low = min(jaw_low, jc[2] - jd)
        print("%6.0f deg | %-22s %-22s | %-16s | %8.4f %8.4f"
              % (np.rad2deg(roll), np.round(ey, 3), np.round(ez, 3),
                 np.round(yaxis, 3), box_low, jaw_low))

    print("\nlateral-ness of the YAW axis (|axis . world y|): 1.0 = yaw pitches the")
    print("hand straight up/down, i.e. the yaw motor drives the gripper into the table.")
    for roll in np.deg2rad([0, -45, -90, 90]):
        d.qpos[ar] = roll
        mujoco.mj_forward(m, d)
        ax = d.xaxis[jy]
        print("  roll %6.0f deg -> yaw axis %s  |.y| = %.3f  |.z| = %.3f"
              % (np.rad2deg(roll), np.round(ax, 3), abs(ax[1]), abs(ax[2])))

    # what the PITCH joint can do by comparison
    jp, ap, dp = jnt(m, "%s_wrist_pitch_joint" % ARM)
    print("\njoint travel and torque, the two candidates for driving the palm down:")
    for nm, j in (("wrist_pitch", jp), ("wrist_yaw", jy)):
        print("  %-12s range %s rad (%.0f deg span)  actfrcrange %s"
              % (nm, np.round(m.jnt_range[j], 3),
                 np.rad2deg(m.jnt_range[j][1] - m.jnt_range[j][0]),
                 np.round(m.jnt_actfrcrange[j], 1)))


if __name__ == "__main__":
    main()
