#!/usr/bin/env python3
"""Replay a crocoddyl plan in MuJoCo under five controllers, score it, render it.

crocoddyl solved the plan against RIGID, PRESCRIBED contacts on a Pinocchio
model.  MuJoCo has soft contacts, a real table it can collide with anywhere, and
position-servo actuators.  Whether the plan survives that is the only question
that matters, and it is not answerable from the solver's own cost.

THE LADDER.  S12 compared two things -- "position tracking" and "torque
inversion" -- found both failed, and concluded that "the missing piece is a
tracking controller".  Neither of those two closes a loop around the BASE, which
is the only loop that matters here, so this runs the whole ladder and lets "does
closing the loop help" be a measurement.  Every mode below sends the command the
real robot takes, (q_des, kp, kd, tau_ff); they differ only in tau_ff and q_des:

  kinematic    no dynamics; drive MuJoCo onto q_plan.  What crocoddyl BELIEVES
               it planned.  Contact forces here are meaningless by construction.
  hold         q_des = the START pose, forever.  The control experiment: it says
               how long this robot stays up doing NOTHING, which turns out to be
               the number every other row has to be read against.
  position     q_des = q_plan, tau_ff = 0.  The servo is the whole controller --
               a naive WBC handoff, with no knowledge of the planned torques.
  ff           q_des = q_plan, tau_ff = tau_plan.  The deployment-faithful
               handoff, and what S12 called "torque inversion" -- note it is NOT
               open-loop torque: the servo's PD term is still underneath it.
               (`torque` is kept as an alias for S12's name.)
  torque_meas  invert the servo at the MEASURED state, which cancels it exactly
               and leaves pure open-loop tau_plan.  The ablation that says how
               much of a replay's survival is the plan and how much is the servo.
  riccati      tau_ff = tau_plan - K_k (x_meas (-) x_plan).  The gains are the
               DDP's OWN Riccati gains, computed on the way to its step and
               otherwise thrown away: a time-varying LQR about the optimal
               trajectory, for free, and the first row that feeds back the base.
  mpc          re-solve a short-horizon OCP from the measured state every control
               step, warm-started from the last solution; apply its first control
               AND its first predicted state as the servo's setpoint.  Refinement
               and stabilisation together.

TWO THINGS THIS GETS RIGHT, both of which have burned this study before:

 1. THE ACTUATORS ARE POSITION SERVOS, not torque sources: tau = kp (ctrl - q)
    - kd qvel, with kp/kd read off the model rather than assumed.  Writing the
    plan's torques into d.ctrl commands a joint angle of a few hundred radians.
 2. THE PLAN IS AT dt = 0.02 AND MuJoCo RUNS AT 0.002.  Each plan node is held
    for the intervening substeps rather than one MuJoCo step per node, which
    would replay a 2 s trajectory in 0.2 s and call the resulting flailing a
    tracking failure.

usage: croco_replay.py --tag s13 [--ctrl riccati] [--video] [--push 120]
"""

import argparse
import json
import sys
import os
import time

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL (see that module)
import pinocchio as pin
import contact_select as cs
import croco_plan as cp
import mujoco

W, H = 960, 720
CTRLS = ["kinematic", "hold", "position", "ff", "torque", "torque_meas",
         "riccati", "mpc"]

# Per-site marker colours for the rendered contacts.  Same hues as the docpage's
# --s1..--s5 series so a video and a plot of the same run agree about which dot
# is the elbow.
SITE_RGBA = {
    "elbow":   (0.15, 0.45, 0.85, 0.95),
    "forearm": (0.90, 0.55, 0.10, 0.95),
    "palm":    (0.20, 0.65, 0.35, 0.95),
    "hip":     (0.60, 0.35, 0.75, 0.95),
    "torso":   (0.85, 0.25, 0.35, 0.95),
}
FOOT_RGBA = (0.45, 0.45, 0.50, 0.75)
TARGET_RGBA = (0.95, 0.15, 0.45, 0.85)
FORCE_SCALE = 0.0015          # metres of arrow per newton (100 N -> 15 cm)

# UNPLANNED table contacts (2026-08-07).  The first version of this drew only
# contacts that walked up to a known brace SITE and dropped everything else,
# which meant the one thing the grid replays most needed to show was invisible:
# in most cells the largest force on the table does not go through the brace at
# all -- it goes through the torso, or the reaching arm, or the gripper of the
# arm that is not bracing (croco_why quantifies this).  A video that hides those
# contacts shows a robot leaning on nothing and gives no clue why it stays up.
# They are drawn in MAGENTA, one colour for all of them, because the message is
# not "which unplanned part" but "this is not the brace".
UNPLANNED_RGBA = (0.95, 0.25, 0.85, 0.95)

# Camera presets.  `wide` is the S13 framing and is kept so the new videos are
# comparable to the ones already on the docpage; `brace` is a close-up on the
# tabletop, which is the only framing in which a 12 mm contact marker is legible.
CAMERAS = {
    "wide":  dict(lookat=(0.75, 0.00, 0.90), distance=3.0, azimuth=120,
                  elevation=-14),
    "brace": dict(lookat=(0.72, 0.00, 1.05), distance=1.4, azimuth=70,
                  elevation=-20),
    "table": dict(lookat=(0.70, 0.10, 1.02), distance=1.2, azimuth=160,
                  elevation=-15),
}


def show_gripper(m):
    """Make the magpie's collision jaws VISIBLE without giving them dynamics.

    Answering a question this study kept asking of the videos: the gripper's only
    rendered geom is the `h12_mount` mesh (group 1), while the parts that
    actually touch the table -- the 171 mm jaw boxes and the wrist brace pad --
    are class="collision", i.e. group 3, which the default renderer hides.  So
    every replay video showed the arm ending in a stub roughly 100 mm short of
    the hardware that was bracing on the wood.

    Promoting those geoms to group 2 is a RENDER-ONLY change to the MjModel: it
    touches `geom_group` and `geom_rgba`, not contype/conaffinity, mass, or
    anything the integrator reads.  And it is worth saying plainly that there are
    no gripper DOFs to remove -- `left_magpie_gripper` is a rigidly welded body
    with `jntnum = 0`, so it already costs the planner nothing but the frames its
    keep-out points hang off.  Nothing is being traded here for the picture.
    """
    for arm in ("left", "right"):
        for suffix in ("gripper_collision", "gripper_jaw_a", "gripper_jaw_b"):
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                                  f"{arm}_{suffix}")
            if g >= 0:
                m.geom_group[g] = 2
                m.geom_rgba[g] = (0.25, 0.25, 0.28, 1.0)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{arm}_wrist_pad")
        if g >= 0:
            m.geom_group[g] = 2
            m.geom_rgba[g] = (0.25, 0.25, 0.28, 1.0)


def ghost_arm(m, arm=None, alpha=0.45):
    """Make the BRACING arm translucent in the render.

    The contact points this session cares about are on the UNDERSIDE of the
    elbow, forearm and gripper, i.e. between the link and the tabletop, so a
    marker drawn where the contact actually is gets hidden by the link that is
    making it -- a video of a brace shows an arm on a table and nothing about
    which part of it is loaded.  Dropping the bracing arm's visual alpha lets the
    markers read without moving them somewhere they are not.

    Render-only: `geom_rgba` alone, and only on group 1/2 (visual) geoms, so the
    collision geometry, the mass and the contact model are untouched.
    """
    arm = arm or cs.BRACE_ARM
    links = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
             "wrist_roll", "wrist_pitch", "wrist_yaw"]
    bodies = {cs.bid(m, f"{arm}_{l}_link") for l in links}
    bodies.add(cs.bid(m, f"{arm}_magpie_gripper"))
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in bodies and m.geom_group[g] in (1, 2):
            m.geom_rgba[g][3] = alpha


def _marker(scene, pos, rgba, size=0.02, gtype=mujoco.mjtGeom.mjGEOM_SPHERE):
    if scene.ngeom >= scene.maxgeom:
        return 0
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype,
                        np.full(3, size), np.asarray(pos, float),
                        np.eye(3).ravel(), np.array(rgba, np.float32))
    scene.ngeom += 1
    return 1


def draw_refs(scene, target, site_ref):
    """The reach target and the certified brace landing spots, as ghost markers.

    Without these a viewer can see the hand move but not whether it arrived, and
    "the brace is 10 mm above where it was certified" -- the S12 defect this
    whole session turns on -- is invisible in a render that draws only what the
    robot is doing and not what it was aiming at.
    """
    n = _marker(scene, target, TARGET_RGBA, 0.022)
    for s, p in site_ref.items():
        rgba = list(SITE_RGBA.get(s, FOOT_RGBA))
        rgba[3] = 0.55                       # ghost: this is a reference, not a fact
        n += _marker(scene, p, rgba, 0.012)
    return n


def draw_contacts(scene, m, d, site_body, tbl, feet):
    """Mark the live table contacts on the rendered scene, coloured by identity.

    MuJoCo's own mjVIS_CONTACTPOINT/CONTACTFORCE flags draw every contact in one
    colour, which in this maneuver is 8 foot corners plus whatever the arm is
    doing -- and the whole question the videos are being asked is WHICH of the
    elbow, forearm and palm is carrying load at any moment.  So the markers are
    drawn per contact instead: a sphere at the contact point in the site's own
    colour, and an arrow along the contact normal whose length is the normal
    force.  Foot contacts get the same treatment in grey, because "is the robot
    still on its heels" is the other thing a viewer wants from the picture.

    Returns the number of geoms added, so the caller can size max_geom.
    """
    buf = np.zeros(6)
    added = 0
    # `hip` and `torso` are both sites on torso_link, so a naive inversion loses
    # one of them.  The ARM sites are the ones this render exists for, so they
    # are written last and win the collision.
    body_site = {}
    for s in ("hip", "torso", "elbow", "forearm", "palm"):
        if s in site_body:
            body_site[site_body[s]] = s
    for c in range(d.ncon):
        if scene.ngeom + 2 > scene.maxgeom:
            break
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if tbl in (b1, b2):
            rb = b2 if b1 == tbl else b1
            rgba = None
            while rb > 0 and rgba is None:            # walk up to a known site
                rgba = SITE_RGBA.get(body_site.get(rb))
                rb = m.body_parentid[rb]
            if rgba is None:
                rgba = UNPLANNED_RGBA
        elif b1 in feet or b2 in feet:
            rgba = FOOT_RGBA
        else:
            continue
        mujoco.mj_contactForce(m, d, c, buf)
        fn = abs(float(buf[0]))
        p = con.pos.copy()
        added += _marker(scene, p, rgba, 0.018)
        if fn > 1.0:
            # frame[0] is the contact NORMAL, pointing from geom[0] into geom[1]
            # (checked against a resting contact: table/object reads +z).  The
            # arrow shows the force ON THE ROBOT, so it runs along +normal when
            # the robot is the second geom and -normal when it is the first.
            # Getting this from "is geom[0] the table" instead does not work for
            # the FEET, whose partner is `world`, not `table` -- their arrows
            # then point down through the floor.
            n = con.frame[:3].copy()
            if not _under(m, b2, 1):
                n = -n
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW,
                                np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                                np.array(rgba, np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.008,
                                 p, p + n * (FORCE_SCALE * fn))
            scene.ngeom += 1
            added += 1
    return added


SEAT_JOINTS = ["left_ankle_roll_joint", "left_ankle_pitch_joint",
               "right_ankle_roll_joint", "right_ankle_pitch_joint"]


def seat_stance(m, d, q, iters=40, tol=1e-5):
    """Level the soles on the floor: the start pose, seated.

    THE DEFECT.  At the shipped `stand` keyframe the feet are not flat.  Each
    sole is rolled about 6.4 deg (9 mm of corner-height spread across an 80 mm
    sole), so the robot stands on its two INNER foot edges -- a knife-edge
    stance whose support polygon is a pair of lines, not a pair of rectangles.
    Every static hold from that pose topples backwards within 1.2 s no matter
    how the joints are held: with integral action driving the joint error to
    0.005 rad the robot still leaves, because the joints were never the problem.
    The maneuver survived it only because the MPC is actively correcting from
    the first control period, which is also why the failure looked like a
    controller problem and was not.

    THE FIX.  Gauss-Newton on five numbers -- base height and the two ankle
    roll/pitch pairs -- driving all eight sole corners onto the floor plane.
    Ankles only: the stance width, the hip angles and everything above the knee
    are what the study certified its poses against, and this must not move them.
    """
    adr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
           for j in SEAT_JOINTS]
    idx = [2] + adr                      # base z, then the four ankle joints
    q = q.copy()

    def corners():
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        return np.array([cs.point_world(m, d, b, o)[2]
                         for f in cs.FEET
                         for b, o in cs.foot_corners(m, d, f)])

    for _ in range(iters):
        r = corners()
        if np.max(np.abs(r)) < tol:
            break
        J = np.zeros((len(r), len(idx)))
        for c, i in enumerate(idx):
            q[i] += 1e-5
            J[:, c] = (corners() - r) / 1e-5
            q[i] -= 1e-5
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        q[idx] += np.clip(step, -0.05, 0.05)
    corners()
    return q


def hold_torque(m, d, q0, seconds=3.0, ki=8.0, tol=2e-3):
    """The feedforward that actually holds `q0`, found by integral action.

    Not `mj_inverse`: with contacts in the loop the inverse dynamics at
    qacc = 0 returns a generalised force including six rows no actuator can
    supply, and driving the servo with its actuated part folds the robot up
    (measured: pelvis 1.00 m -> 0.39 m).  Integral action asks the plant the
    question directly -- raise the feedforward until the joints stop sagging --
    and what it converges to IS the static torque, contacts and all.

    Returns (tau, settled_qpos, joint error).  The error is the interesting
    output: it says whether the pose is holdable inside the clamp basis at all.
    """
    kp, _ = servo_gains(m)
    nq = cb.NQ_ROBOT
    lim = cs.torque_limits(m)
    d.qpos[:] = q0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    tau = np.zeros(m.nu)
    qd = q0[7:nq].copy()
    for _ in range(int(round(seconds / m.opt.timestep))):
        e = qd - d.qpos[7:nq]
        tau = np.clip(tau + ki * kp * e * m.opt.timestep, -lim, lim)
        d.ctrl[:] = qd + tau / kp
        mujoco.mj_step(m, d)
        if np.max(np.abs(e)) < tol and np.max(np.abs(d.qvel[:nq - 1])) < 1e-2:
            break
    q = d.qpos.copy()
    err = float(np.max(np.abs(qd - q[7:nq])))
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return tau, q, err


def settled_start(m, d, q0, seconds=0.5):
    """Where the plant is after `seconds` of the static hold that precedes the
    maneuver -- which is where a plan should START, not where the keyframe is.

    The controller's settle (see `replay`) holds the start pose under the
    static-equilibrium QP's torque.  That holds it, but not exactly: the servo
    still needs a position error to make up whatever the QP's min-effort
    solution does not, and the pose ends about 0.09 rad away from the keyframe
    over 27 joints.  The plan meanwhile begins at the keyframe.  Closing that
    last gap is what this is for -- and unlike everything else in the launch
    story it is deterministic, so it can be baked into the plan offline.
    """
    kp, _ = servo_gains(m)
    nq = cb.NQ_ROBOT
    lim = cs.torque_limits(m)
    d.qpos[:] = q0
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    tau = np.clip(np.array(cs.equilibrium_qp(m, d, ())["tau"], float),
                  -lim, lim)
    qd = q0[7:nq].copy()
    for _ in range(int(round(seconds / m.opt.timestep))):
        d.ctrl[:] = qd + tau / kp
        mujoco.mj_step(m, d)
    q = d.qpos.copy()
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    return q


def corrupt(x, sense, rng, bias, nq):
    """A state ESTIMATE from the true state: per-run bias plus white noise.

    Split that way on purpose.  A state estimator's error is dominated by a
    slowly-varying term -- IMU bias, an accumulated yaw, a base position that
    has drifted since the last contact reset -- and a controller that re-solves
    every 20 ms rejects white noise far more easily than a constant offset.
    Reporting a single "state noise" number would therefore flatter the loop.
    """
    x = x.copy()
    x[0:3] += bias["base_p"] + rng.normal(0, sense.get("base_p", 0.0) / 3, 3)
    if np.any(bias["base_r"]):
        rp = sense.get("base_rp", sense.get("base_r", 0.0))
        yaw = sense.get("base_yaw", sense.get("base_r", 0.0))
        dr = bias["base_r"] + np.array([rng.normal(0, rp / 3),
                                        rng.normal(0, rp / 3),
                                        rng.normal(0, yaw / 3)])
        # `@`, not `*`.  numpy's `*` on two 3x3 arrays is element-wise, which
        # silently returns a non-orthonormal matrix -- and pin.Quaternion
        # accepts it, so the run does not fail, it just gets a corrupted
        # attitude of a magnitude that has nothing to do with dr.  It looked
        # like a 0.001-degree attitude error toppling the robot, at exactly the
        # same reach error for every magnitude tried, which is what gave it
        # away: a physical sensitivity has a slope.
        q = pin.Quaternion(x[6], x[3], x[4], x[5])
        q = pin.Quaternion(pin.exp3(dr) @ q.matrix())
        x[3:7] = [q.x, q.y, q.z, q.w]
    x[7:nq] += bias["q"] + rng.normal(0, sense.get("q", 0.0), nq - 7)
    x[nq:nq + 3] += bias["base_v"] + rng.normal(0, sense.get("base_v", 0.0) / 3, 3)
    x[nq + 3:nq + 6] += rng.normal(0, sense.get("base_w", 0.0), 3)
    x[nq + 6:] += rng.normal(0, sense.get("v", 0.0), nq - 7)
    return x


def servo_gains(m):
    """(kp, kd) per actuator, read off the MJCF position actuators."""
    kp = m.actuator_gainprm[:m.nu, 0].copy()
    kd = -m.actuator_biasprm[:m.nu, 2].copy()
    if not np.allclose(-m.actuator_biasprm[:m.nu, 1], kp):
        raise RuntimeError("actuators are not the expected position servos "
                           "(biasprm[1] != -gainprm[0]); check the model")
    return kp, kd


def support_margin(m, d, contacts_only=True):
    """Signed distance from the CoM to the edge of the FOOT support polygon [m].

    Deliberately the foot polygon and not stability.equilibrium_region: this is
    evaluated on every one of ~200 frames of every replay, and the region is 36
    LP ray-shoots.  The region is the certificate and croco_score computes it on
    the frames that matter; this is the cheap running signal, and on the
    feet-only phases the two agree about what "leaving support" means.
    """
    pts = []
    for f in cs.FEET:
        for body, off in cs.foot_corners(m, d, f):
            pts.append(cs.point_world(m, d, body, off)[:2])
    pts = np.array(pts)
    # convex hull of 8 coplanar points -> axis-aligned is exact for two rectangles
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = d.subtree_com[1][:2]
    inside = np.all((c > lo) & (c < hi))
    dist = min(np.min(c - lo), np.min(hi - c))
    return float(dist if inside else -abs(dist))


# The MPC controller now lives in croco.control.mpc -- it is the only part of
# this file that would run on a robot, and it was unreachable at line 434 of a
# replay/scoring/rendering script. Imported rather than duplicated so there is
# one copy to fix. `croco_replay.MPC` keeps working for every caller and every
# docpage that names it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from croco.control.mpc import MPC          # noqa: E402,F401


def certified_sites(plan, run_dir, m, d):
    """World positions of the bracing sites AT q*, for the render's ghost markers.

    Evaluated in MuJoCo off the saved q*, then the state is restored -- so this
    is the same number the static QP certified and the same one the OCP's
    `land_`/`hold_` costs are written against, read out of the model the replay
    is about to run rather than re-derived.
    """
    if not plan["subset"]:
        return {}
    modes = json.load(open(os.path.join(run_dir, "modes.json")))
    entry = next(e for e in modes["modes"] if e["name"] == plan["mode"])
    q_star = np.loadtxt(os.path.join(run_dir, entry["qpos_file"]))
    keep_q, keep_v = d.qpos.copy(), d.qvel.copy()
    d.qpos[:len(q_star)] = q_star
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    ref = {s: cs.point_world(m, d, *cs.SITES[s]).copy() for s in plan["subset"]}
    d.qpos[:], d.qvel[:] = keep_q, keep_v
    mujoco.mj_forward(m, d)
    return ref


def build_ocp(plan, run_dir):
    """Rebuild the OCP the plan came from, for the MPC and the Riccati state map."""
    modes = json.load(open(os.path.join(run_dir, "modes.json")))
    entry = next(e for e in modes["modes"] if e["name"] == plan["mode"])
    q_star = np.loadtxt(os.path.join(run_dir, entry["qpos_file"]))
    m, d = cs.load()
    q0 = cs.start_qpos(m, plan["start"])
    d.qpos[:] = q0
    mujoco.mj_forward(m, d)
    ocp = cp.LeanOCP(plan["subset"], q_star, q0, mu=plan["mu"],
                     table_z=cs.table_top_z(m, d),
                     reach_target=np.array(plan["target"]),
                     cones=plan["cones"], legacy=plan.get("legacy", False),
                     keepout=bool(plan.get("keepout", 0)),
                     com_margin=plan.get("com_margin", 0.03),
                     w_land=plan.get("w_land", 5e3), w_hold=plan.get("w_hold", 1e3),
                     w_com=plan.get("w_com", 1e3),
                     w_cone=plan.get("w_cone", 1e1),
                     w_reach=plan.get("w_reach", 1e2),
                     cop_shrink=plan.get("cop_shrink", 1.0),
                     min_nforce=plan.get("min_nforce", 1.0),
                     w_com_track=plan.get("w_com_track", 0.0),
                     w_com_damp=plan.get("w_com_damp", 0.0),
                     w_ctrl=plan.get("w_ctrl", 1e-3),
                     n_hold=plan.get("n_hold", 0),
                     w_hold_state=plan.get("w_hold_state", 1e2),
                     # 1e1 is the pre-S17 default, so a plan solved before this
                     # session still gets the weight it was solved with.
                     w_jlim=plan.get("w_jlim", 1e1),
                     w_vel=plan.get("w_vel", 0.0),
                     reach_rot=plan.get("reach_rot"),
                     w_reach_rot=plan.get("w_reach_rot", 0.0),
                     drop=[v for v in plan.get("drop", "").split(",") if v])
    return ocp, q_star


def replay(tag, ctrl_mode="riccati", dt_plan=0.02, run_dir=".", video=None,
           fps=30, speed=1.0, push=0.0, push_at=0.5, push_dir=(1, 0, 0),
           push_hold=0.2, mpc_horizon=40, mpc_iters=2, seed=None, q_noise=0.0,
           cam="wide", ghost=True, qpos_out=None, mpc_cones=True,
           mpc_alphas=0, mpc_threads=0, tau_clamp=False, mpc_dt_scale=1,
           base_xy=0.0, base_z=0.0, base_yaw=0.0, base_rp=0.0, settle=0,
           sense=None, mu_scale=1.0, table_shift=(0.0, 0.0, 0.0),
           schedule_shift=0):
    xs = np.load(os.path.join(run_dir, f"xs_{tag}.npy"))
    us = np.load(os.path.join(run_dir, f"us_{tag}.npy"))
    with open(os.path.join(run_dir, f"plan_{tag}.json")) as fh:
        plan = json.load(fh)
    # STANCE FIRST, before any cs.load() in this call: the offset is part of the
    # model the plan was solved against, and a replay that loads the unshifted
    # stance is replaying a plan for a robot standing somewhere else.
    cs.STANCE_DX = plan.get("stance_dx", cs.STANCE_DX)
    cs.STANCE_DY = plan.get("stance_dy", cs.STANCE_DY)
    subset = plan["subset"]
    Kpath = os.path.join(run_dir, f"K_{tag}.npy")
    Ks = np.load(Kpath) if os.path.exists(Kpath) else None
    if ctrl_mode == "riccati" and Ks is None:
        raise SystemExit(f"{Kpath} missing -- re-run croco_run.py to save gains")

    ocp = mpc = state = None
    if ctrl_mode in ("riccati", "mpc"):
        ocp, _ = build_ocp(plan, run_dir)
        state = ocp.state
        if ctrl_mode == "mpc":
            # The ONLINE problem may drop the cone costs while the OFFLINE plan
            # keeps them.  That is a real change to what the MPC optimises and it
            # is opt-in for that reason -- but it is the largest remaining item
            # in the step: the five cone terms are 36 us of a 131 us braced node,
            # and they are also the only costs that read the contact-force
            # derivatives, so dropping them lets `enable_force` go off too, for
            # 50 us total.  The plan the horizon slides over was solved WITH the
            # cones, so the trajectory being tracked is still cone-feasible; the
            # question this makes measurable is whether re-solving needs to
            # re-check that every 20 ms.  `--mpc-no-cones` and the (H, iters,
            # cones) grid in croco_speed.py sweep report what it costs.
            problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                                n_braced=plan["n_braced"],
                                n_return=plan.get("n_return", 0),
                                dwell=plan.get("dwell", 0),
                                cones=plan["cones"] and mpc_cones)
            problem = ocp.build(dt=plan["dt"] * max(int(mpc_dt_scale), 1),
                                n_approach=plan["n_approach"],
                                n_braced=plan["n_braced"],
                                n_return=plan.get("n_return", 0),
                                dwell=plan.get("dwell", 0),
                                cones=plan["cones"] and mpc_cones) \
                if mpc_dt_scale > 1 else problem
            mpc = MPC(ocp, list(problem.runningModels), problem.terminalModel,
                      horizon=mpc_horizon, iters=mpc_iters,
                      xs_plan=xs, us_plan=us, n_alphas=mpc_alphas,
                      nthreads=mpc_threads, dt_scale=mpc_dt_scale)

    # NO inflated collision margin.  cs.load defaults to margin = 25 mm so the
    # IK can see collisions before it enters them; in a DYNAMICS replay that
    # same inflation makes every geom generate contact forces 25 mm before it
    # touches anything, which is not the contact model the plan is being tested
    # against.  S12's replays ran with it on.
    m, d = cs.load(ik_margin=0.0)
    kp, kd = servo_gains(m)
    ctrl_lo, ctrl_hi = m.actuator_ctrlrange[:m.nu].T.copy()
    if tau_clamp:
        # The PLANT clamped to the same basis the plan is solved against.
        # Without this the servo term can push a joint past the clamp limit --
        # measured at 1.11x on left_shoulder_pitch in the nominal replay -- so a
        # run can be "successful" while asking the hardware for torque its
        # safety layer would refuse.  With it, the limit is enforced by the
        # actuator rather than checked afterwards, and the controller has to
        # cope with saturation instead of being credited for exceeding it.
        lim = cs.torque_limits(m)
        m.actuator_forcerange[:m.nu, 0] = -lim
        m.actuator_forcerange[:m.nu, 1] = lim
        m.actuator_forcelimited[:m.nu] = 1
    nsub = max(1, int(round(dt_plan / m.opt.timestep)))

    qpos_full = cs.start_qpos(m, plan["start"])
    nq = cb.NQ_ROBOT
    d.qpos[:] = cb.pin_to_mj(xs[0][:nq], qpos_full)
    d.qvel[:] = 0
    # HOW THE ROBOT ARRIVES.  S15's noise model perturbed the 27 joint angles and
    # left the floating base exactly where the plan put it, which is not how this
    # robot is instantiated: it is lowered on a winch, so the pose it starts from
    # varies in the BASE -- where it lands on the floor, which way it is facing,
    # and how level it is -- as much as in the joints.  Those are different
    # disturbances.  Joint noise alone moves the feet relative to the floor and
    # is really a drop test (measured: 12-25 mm of foot-corner spread at
    # 0.02 rad); base noise moves the whole robot relative to the TABLE, which
    # is what decides whether the certified landing spots are still where the
    # brace is going.
    if q_noise or base_xy or base_z or base_yaw or base_rp:
        rng = np.random.default_rng(seed)
        if q_noise:
            d.qpos[7:nq] += rng.normal(0.0, q_noise, nq - 7)
        if base_xy:
            d.qpos[0:2] += rng.normal(0.0, base_xy, 2)
        if base_z:
            d.qpos[2] += rng.normal(0.0, base_z)
        if base_yaw or base_rp:
            rpy = np.array([rng.normal(0.0, base_rp), rng.normal(0.0, base_rp),
                            rng.normal(0.0, base_yaw)])
            dq, out = np.zeros(4), np.zeros(4)
            mujoco.mju_euler2Quat(dq, rpy, "xyz")
            mujoco.mju_mulQuat(out, dq, d.qpos[3:7])
            d.qpos[3:7] = out
    mujoco.mj_forward(m, d)

    # WHAT THE CONTROLLER IS ALLOWED TO KNOW.  Everything above hands the MPC
    # MuJoCo's exact state.  On hardware the floating base is an ESTIMATE -- IMU
    # plus leg kinematics plus a contact assumption -- and its error is
    # slowly-varying, not white, so it is modelled as a per-run bias with a
    # smaller white part on top.  Joint angles are encoders (good) and joint
    # velocities are differentiated encoders (not).  Note what is NOT here: the
    # cone costs never read a force sensor.  Their residual is A*lambda - b with
    # lambda the wrench the OCP's own KKT solve produces for (x, u), so the
    # privileged quantity in this loop is the STATE, not the contact force.
    srng = np.random.default_rng((seed or 0) + 9973)
    sbias = {}
    if sense:
        # Attitude error is split into roll/pitch and yaw on purpose.  They come
        # from different places on a real robot -- gravity pins roll and pitch
        # to a tenth of a degree and nothing pins yaw at all without a
        # magnetometer or vision -- so a single "orientation error" number
        # cannot say whether this loop is deployable.
        rp = sense.get("base_rp", sense.get("base_r", 0.0))
        yaw = sense.get("base_yaw", sense.get("base_r", 0.0))
        sbias = dict(
            base_p=srng.normal(0, sense.get("base_p", 0.0), 3),
            base_r=np.array([srng.normal(0, rp), srng.normal(0, rp),
                             srng.normal(0, yaw)]),
            base_v=srng.normal(0, sense.get("base_v", 0.0), 3),
            q=srng.normal(0, sense.get("q_bias", 0.0), nq - 7))
    if mu_scale != 1.0:
        # The plan assumes mu = 0.6 at the table and the floor and nobody has
        # measured it.  This scales the PLANT's tangential friction only.
        m.geom_friction[:, 0] *= mu_scale
    if any(table_shift):
        # The table where the plan thinks it is, versus where it is.  The
        # landing spots, the keep-out box and the reach target are all written
        # in world coordinates off the nominal table, so moving the real one is
        # exactly the perception error a deployment would carry.
        tj = m.body_jntadr[cs.bid(m, "table")]
        adr = m.jnt_qposadr[tj]
        d.qpos[adr:adr + 3] += np.asarray(table_shift, float)
        mujoco.mj_forward(m, d)

    renderer = frames = None
    if video:
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
        m.vis.global_.offheight = max(m.vis.global_.offheight, H)
        show_gripper(m)
        if ghost:
            ghost_arm(m)
        # +256 geoms of headroom for the contact markers (2 per contact).  The
        # default max_geom is sized from the model, so appending to the scene
        # without this silently drops the markers once the arm is down and there
        # are a dozen live contacts -- exactly the frames they exist for.
        renderer = mujoco.Renderer(m, H, W, max_geom=m.ngeom + 256)
        frames = []
        frame_every = max(1, int(round(speed / (fps * dt_plan))))
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        preset = CAMERAS[cam]
        camera.lookat[:] = preset["lookat"]
        camera.distance = preset["distance"]
        camera.azimuth = preset["azimuth"]
        camera.elevation = preset["elevation"]

    tbl = cs.bid(m, "table")
    brace_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in subset}
    # For the render: EVERY arm site, not just the ones in the contact schedule.
    # A video that only marks the planned contacts cannot show the failure this
    # study spent S12 chasing -- a link that is touching the table while the plan
    # says it is not.
    all_site_bodies = {s: cs.bid(m, cs.SITES[s][0]) for s in SITE_RGBA
                       if s in cs.SITES}
    feet = [cs.bid(m, f) for f in cs.FEET]
    tau_lim = cs.torque_limits(m)
    target = np.array(plan["target"])
    site_ref = certified_sites(plan, run_dir, m, d) if video else {}
    push_k = int(round(push_at * len(us)))
    log = []
    # Full qpos trace, saved alongside the log as .npy.  The JSON log carries
    # scalars only, which is enough to say a replay reached the target and not
    # enough to ask anything else of the pose it reached -- and the CoM-region
    # comparison (croco_region.py) needs the POSE, not a summary of it.  41
    # doubles x ~200 nodes is 66 kB of binary, so it is saved for every replay
    # rather than for the ones that turn out to be interesting.
    qtrace = []

    # SETTLE.  `settle` control periods spent on the FIRST node of the plan
    # before the maneuver is allowed to advance: the controller holds the start
    # pose while the winch lets go.  It is not cosmetic -- the initial state is
    # off the plan by construction now (base pose included), and a receding
    # horizon that starts advancing its schedule from a state it has not yet
    # stabilised is solving for a maneuver the robot is not standing in.  The
    # settle steps are scored separately (plan["settle"]) and kept OUT of `log`
    # so every metric downstream still indexes the plan's own node numbering.
    settle_log = []
    tau_hold = np.zeros(m.nu)
    if settle:
        keep_q, keep_v = d.qpos.copy(), d.qvel.copy()
        d.qpos[:] = cb.pin_to_mj(xs[0][:nq], qpos_full)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        tau_hold = np.array(cs.equilibrium_qp(m, d, ())["tau"], float)
        d.qpos[:], d.qvel[:] = keep_q, keep_v
        mujoco.mj_forward(m, d)
    for k in [0] * settle + list(range(len(us))):
        in_settle = len(settle_log) < settle
        q_plan = xs[k][:nq]
        v_plan = xs[k][nq:]
        # Pinocchio joint order == MJCF joint order for the 27 actuated joints,
        # and both put them after the free joint, so the slice is direct.
        qj, vj = q_plan[7:], v_plan[6:]

        clip_max, clip_n = 0.0, 0
        if ctrl_mode == "kinematic":
            d.qpos[:] = cb.pin_to_mj(q_plan, qpos_full)
            d.qvel[:] = 0
            mujoco.mj_forward(m, d)
            tau_cmd = us[k]
        else:
            # THE COMMAND THE ROBOT ACTUALLY TAKES.  h12_control_node sends
            # (q_des, kp, kd, tau_ff) and the motor driver forms
            #     tau = kp (q_des - q) + kd (v_des - v) + tau_ff
            # downstream of it.  A MuJoCo <position> actuator emits
            #     tau = kp (ctrl - q) - kd v,
            # so the two are identical under
            #     ctrl = q_des + (tau_ff + kd v_des) / kp,
            # which is the inversion S12 called "torque" -- and the important
            # consequence is that it is NOT open-loop torque: the servo's own
            # PD term is still there, doing the stabilising.  Every mode below
            # except `torque_meas` is that command with a different tau_ff, so
            # the ladder isolates the feedforward and leaves the interface fixed.
            q_des, v_des = qj, vj
            if in_settle:
                # A STATIC HOLD, and the feedforward is the thing that makes
                # it one.  Three candidates were measured from the `stand` pose,
                # all with the joints commanded to the keyframe:
                #   tau_ff = 0            sags 0.09 rad in 0.2 s and topples
                #                         backwards by 1.2 s
                #   tau_ff = plan u[0]    holds, but 112 N.m away from static
                #                         (knees -67 N.m where +10 is wanted),
                #                         so it settles 0.59 rad off the pose
                #   tau_ff = static QP    holds the pose: pelvis 0.995 m and CoM
                #                         inside the polygon for 3 s
                # The third is the study's own `equilibrium_qp`, i.e. the same
                # object that certified q*, evaluated at the START pose with
                # feet-only support.  It is a constant of the keyframe, not of
                # the plan, so it is exactly what a stand controller would have
                # loaded before the maneuver was ever commanded.
                q_des, v_des = xs[0][7:nq], np.zeros(nq - 7)
                tau_ff = np.clip(tau_hold, -tau_lim, tau_lim)
                tau_cmd = tau_ff
                raw_ctrl = q_des + (tau_ff + kd * v_des) / kp
                d.ctrl[:] = raw_ctrl
                clip = np.maximum(ctrl_lo - raw_ctrl, raw_ctrl - ctrl_hi)
                clip_max = float(max(clip.max(), 0.0))
                clip_n = int(np.sum(clip > 1e-9))
                for _ in range(nsub):
                    mujoco.mj_step(m, d)
                settle_log.append(dict(
                    k=-1, t=0.0, pelvis_z=float(d.qpos[2]),
                    tracking_err=float(np.linalg.norm(d.qpos[7:nq] - q_des)),
                    tau_ratio=float(np.max(np.abs(d.actuator_force[:m.nu])
                                           / tau_lim)),
                    support_margin=support_margin(m, d),
                    com=[float(v) for v in d.subtree_com[1]]))
                continue
            if ctrl_mode == "hold":
                q_des, v_des = xs[0][7:nq], np.zeros(nq - 7)
                tau_ff = np.zeros(m.nu)
            elif ctrl_mode == "position":
                tau_ff = np.zeros(m.nu)
            elif ctrl_mode in ("ff", "torque"):
                tau_ff = us[k]
            else:
                x_meas = np.concatenate([
                    cb.mj_to_pin(d.qpos),
                    cb.mj_to_pin_v(d.qvel, d.xmat[1].reshape(3, 3))])
                if sense:
                    x_meas = corrupt(x_meas, sense, srng, sbias, nq)
                if ctrl_mode == "torque_meas":
                    tau_ff = us[k]
                elif ctrl_mode == "riccati":
                    # u = u* - K (x (-) x*).  crocoddyl's K is defined with that
                    # sign; state.diff(x*, x) is the tangent step FROM the plan TO
                    # the measurement, which is the argument the gain expects.
                    dx = state.diff(xs[k], x_meas)
                    tau_ff = us[k] - Ks[min(k, len(Ks) - 1)] @ dx
                else:                                    # mpc
                    # SCHEDULE SHIFT.  The contact schedule is prescribed by
                    # clock; on hardware the touchdown instant comes from a
                    # detector with latency.  A positive shift is the controller
                    # believing the brace is down before it is.
                    u0, xs1 = mpc(min(max(k + schedule_shift, 0),
                                      len(us) - 1), x_meas)
                    if u0 is None:
                        tau_ff = us[k]
                    else:
                        tau_ff = u0
                        # The MPC re-plans the reference too: its own next state
                        # is a better setpoint for the servo than the offline
                        # plan's, and using it is what "refinement" means here.
                        q_des, v_des = xs1[:nq][7:], xs1[nq:][6:]
                tau_ff = np.clip(tau_ff, -tau_lim, tau_lim)
                if ctrl_mode == "torque_meas":
                    # ABLATION: invert at the MEASURED state instead, which
                    # cancels the servo exactly and leaves pure open-loop torque.
                    # Kept because it is the only way to see how much of a
                    # replay's survival is the plan and how much is the servo.
                    d.ctrl[:] = (d.qpos[7:nq]
                                 + (tau_ff + kd * d.qvel[6:nq - 1]) / kp)
                    q_des = None
            tau_cmd = tau_ff
            if q_des is not None:
                # RAW before MuJoCo clamps it.  <position> actuators declare a
                # ctrlrange equal to the joint range, and the servo inversion
                # puts the feedforward INTO the setpoint -- so a command that
                # needs tau_ff near the limit asks for a setpoint tau_ff/kp
                # outside the joint range and is silently truncated.  Recorded
                # rather than assumed: `ctrl_clip` is how much of the command
                # the plant never saw.
                raw_ctrl = q_des + (tau_ff + kd * v_des) / kp
                d.ctrl[:] = raw_ctrl
                clip = np.maximum(ctrl_lo - raw_ctrl, raw_ctrl - ctrl_hi)
                clip_max = float(max(clip.max(), 0.0))
                clip_n = int(np.sum(clip > 1e-9))
            else:
                clip_max, clip_n = 0.0, 0
            # DISTURBANCE.  A sustained horizontal force on the pelvis, held for
            # `push_hold` seconds, rather than a velocity impulse on the base.
            # The impulse version is nearly free to reject -- both feet are on the
            # ground, so the added base velocity is absorbed within a step, and
            # 0.5 m/s changed the reach error by 1 mm -- while a force in newtons
            # is directly comparable to stability.max_push, the study's own
            # "how hard can something shove this pose" metric.
            if push:
                on = push_k <= k < push_k + max(1, int(round(push_hold / dt_plan)))
                d.xfrc_applied[1, 0:3] = np.asarray(push_dir, float) * push if on \
                    else 0.0
            for _ in range(nsub):
                mujoco.mj_step(m, d)

        reach_p = cs.point_world(m, d, cs.REACH_BODY, cs.REACH_OFF)
        rec = dict(k=k, t=k * dt_plan,
                   pelvis_z=float(d.qpos[2]),
                   tracking_err=float(np.linalg.norm(d.qpos[7:nq] - qj)),
                   reach_err=float(np.linalg.norm(reach_p - target)),
                   com=[float(v) for v in d.subtree_com[1]],
                   support_margin=support_margin(m, d),
                   tau_ratio=float(np.max(np.abs(d.actuator_force[:m.nu]) / tau_lim)),
                   tau_argmax=int(np.argmax(np.abs(d.actuator_force[:m.nu]) / tau_lim)),
                   ctrl_clip=clip_max, ctrl_clip_n=clip_n,
                   cmd_ratio=float(np.max(np.abs(tau_cmd) / tau_lim)))
        f_brace = {s: 0.0 for s in subset}
        f_feet = 0.0
        # EVERY robot-table normal force gets counted SOMEWHERE. Until now
        # `f_brace` was keyed on the plan's subset, so any contact on a body
        # that is not a named site's body and not a foot was summed nowhere and
        # the total silently under-reported the brace. It is not hypothetical:
        # in `elbow+palm` the arm rests on `left_wrist_pad`, which lives on
        # `left_wrist_yaw_link` -- a body no site names -- while the "palm"
        # site lives on the gripper, which a 90 mm keepout holds clear of the
        # table. So the plan reported 175 N through the elbow and 0 N through
        # the palm and neither number was the whole brace.
        f_other = 0.0
        other_bodies = set()
        deepest = 0.0
        buf = np.zeros(6)
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
            mujoco.mj_contactForce(m, d, c, buf)
            fn = abs(float(buf[0]))
            if tbl in (b1, b2):
                rb = b2 if b1 == tbl else b1
                if _under(m, rb, 1):
                    deepest = min(deepest, float(con.dist))
            if tbl in (b1, b2):
                rb = b2 if b1 == tbl else b1
                hit = [s for s, bid_ in brace_bodies.items() if bid_ == rb]
                if hit:
                    for s in hit:
                        f_brace[s] += fn
                elif rb not in feet and _under(m, rb, 1):
                    f_other += fn
                    other_bodies.add(rb)
            if b1 in feet or b2 in feet:
                f_feet += fn
        rec.update({f"F_{s}": f_brace[s] for s in subset})
        rec["F_feet"] = f_feet
        rec["F_other"] = f_other
        rec["F_other_bodies"] = sorted(
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
            for b in other_bodies)
        rec["F_brace_total"] = float(sum(f_brace.values()) + f_other)
        rec["penetration"] = deepest
        log.append(rec)
        qtrace.append(d.qpos.copy())

        if renderer is not None and k % frame_every == 0:
            renderer.update_scene(d, camera=camera)
            draw_refs(renderer.scene, target, site_ref)
            draw_contacts(renderer.scene, m, d, all_site_bodies, tbl, feet)
            frames.append(renderer.render())

    if renderer is not None:
        import imageio.v2 as imageio
        imageio.mimsave(video, frames, fps=fps, quality=8, macro_block_size=1)
        print(f"wrote {video}  ({len(frames)} frames)")

    if qpos_out:
        np.save(qpos_out, np.array(qtrace))

    if settle_log:
        plan["settle"] = dict(
            steps=len(settle_log),
            fell=bool(min(r["pelvis_z"] for r in settle_log) < 0.55),
            min_pelvis_z=min(r["pelvis_z"] for r in settle_log),
            max_tau_ratio=max(r["tau_ratio"] for r in settle_log),
            min_support_margin=min(r["support_margin"] for r in settle_log),
            drift_rad=settle_log[-1]["tracking_err"],
            com_x=[r["com"][0] for r in settle_log[::5]])
    if mpc is not None:
        sl = np.array(mpc.step_lengths)
        plan["mpc_solve_ms"] = dict(
            mean=float(1000 * np.mean(mpc.solve_times)),
            p95=float(1000 * np.percentile(mpc.solve_times, 95)),
            horizon=mpc.H, iters=mpc.iters,
            nthreads_requested=int(mpc.nthreads),
            nthreads_effective=int(mpc.problem.nthreads),
            step_length_median=float(np.median(sl)),
            step_length_min=float(sl.min()),
            # A rung of crocoddyl's alpha ladder is a rollout, so log2(1/alpha)
            # rollouts were thrown away before one was accepted.
            trials_median=float(np.median(np.log2(1.0 / sl)) + 1),
            frac_full_step=float(np.mean(sl >= 1.0)))
    return log, plan


def _under(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def summarise(log, plan, ctrl_mode, verbose=True):
    last = log[-1]
    n_ret = plan.get("n_return", 0)
    k_score = len(log) - 1 - n_ret          # end of the braced phase
    braced = log[plan["n_approach"]:len(log) - n_ret]
    fell = min(r["pelvis_z"] for r in log) < 0.55
    res = dict(
        ctrl=ctrl_mode, fell=bool(fell),
        final_pelvis_z=last["pelvis_z"],
        min_pelvis_z=min(r["pelvis_z"] for r in log),
        max_tracking_err=max(r["tracking_err"] for r in log),
        final_tracking_err=last["tracking_err"],
        reach_err_at_brace_end=log[k_score]["reach_err"],
        reach_err_best=min(r["reach_err"] for r in braced) if braced else None,
        reach_err_braced_mean=float(np.mean([r["reach_err"] for r in braced]))
        if braced else None,
        reach_err_braced_std=float(np.std([r["reach_err"] for r in braced]))
        if braced else None,
        brace_force_peak={s: max(r[f"F_{s}"] for r in log) for s in plan["subset"]},
        brace_force_braced_mean={
            s: float(np.mean([r[f"F_{s}"] for r in braced])) if braced else 0.0
            for s in plan["subset"]},
        brace_total_braced_mean=float(np.mean([r["F_brace_total"] for r in braced]))
        if braced else 0.0,
        brace_force_other_mean=float(np.mean([r["F_other"] for r in braced]))
        if braced else 0.0,
        brace_force_other_peak=max(r["F_other"] for r in log),
        brace_other_bodies=sorted({b for r in log for b in r["F_other_bodies"]}),
        feet_force_final=last["F_feet"],
        min_support_margin=min(r["support_margin"] for r in log),
        support_margin_at_brace_end=log[k_score]["support_margin"],
        max_tau_ratio=max(r["tau_ratio"] for r in log),
        worst_penetration=min(r["penetration"] for r in log),
    )
    if "mpc_solve_ms" in plan:
        res["mpc_solve_ms"] = plan["mpc_solve_ms"]
    if not verbose:
        return res
    print(f"\n--- MuJoCo replay ({ctrl_mode}) ---")
    print(f"  steps                {len(log)}   ({last['t']:.2f} s)")
    print(f"  pelvis z  final      {res['final_pelvis_z']:.3f} m   "
          f"min {res['min_pelvis_z']:.3f} m   "
          f"{'TOPPLED' if fell else 'upright'}")
    print(f"  joint tracking error final {res['final_tracking_err']:.4f} rad, "
          f"max {res['max_tracking_err']:.4f} rad")
    print(f"  reach error at brace end   {res['reach_err_at_brace_end']*1000:7.1f} mm"
          f"   (best over braced phase {res['reach_err_best']*1000:.1f} mm)")
    for s in plan["subset"]:
        print(f"  brace {s:9s} force  braced-mean "
              f"{res['brace_force_braced_mean'][s]:7.1f} N   "
              f"peak {res['brace_force_peak'][s]:7.1f} N")
    if res["brace_force_other_peak"] > 0.0:
        print(f"  brace UNATTRIBUTED    braced-mean "
              f"{res['brace_force_other_mean']:7.1f} N   "
              f"peak {res['brace_force_other_peak']:7.1f} N")
        print(f"    carried by {', '.join(res['brace_other_bodies'])} -- bodies "
              f"no site in the plan's subset names")
    print(f"  feet normal force    final {res['feet_force_final']:7.1f} N   "
          f"(robot weight 673.6 N)")
    print(f"  CoM support margin   min {res['min_support_margin']*1000:+7.1f} mm  "
          f"at brace end {res['support_margin_at_brace_end']*1000:+7.1f} mm")
    print(f"  max |tau|/limit      {res['max_tau_ratio']:.3f} (measured)")
    print(f"  worst table penetration {res['worst_penetration']*1000:7.1f} mm")
    if "mpc_solve_ms" in res:
        s = res["mpc_solve_ms"]
        print(f"  MPC solve            mean {s['mean']:.1f} ms, p95 {s['p95']:.1f} ms "
              f"(H={s['horizon']}, {s['iters']} iters, control period "
              f"{plan['dt']*1000:.0f} ms)")
        if "step_length_median" in s:
            print(f"  MPC line search      median step {s['step_length_median']:.4f}"
                  f" ({s['trials_median']:.0f} rollouts), "
                  f"{100*s['frac_full_step']:.0f}% took the full step")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="elbow+forearm")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--ctrl", default="riccati", choices=CTRLS)
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--push", type=float, default=0.0,
                    help="horizontal force [N] on the pelvis, mid-trajectory")
    ap.add_argument("--push-at", type=float, default=0.5,
                    help="fraction of the trajectory at which the push starts")
    ap.add_argument("--push-hold", type=float, default=0.2,
                    help="how long the push is held [s]")
    ap.add_argument("--push-dir", default="-1,0,0",
                    help="unit direction of the push; default -x, i.e. AWAY from "
                         "the table, which is the direction the plant already "
                         "falls (the planned CoP rides the heel edge)")
    ap.add_argument("--q-noise", type=float, default=0.0,
                    help="std [rad] of joint noise on the initial state")
    ap.add_argument("--base-xy", type=float, default=0.0,
                    help="std [m] of horizontal error in where the winch sets "
                         "the robot down")
    ap.add_argument("--base-z", type=float, default=0.0,
                    help="std [m] of vertical error in where the winch lets go")
    ap.add_argument("--base-yaw", type=float, default=0.0,
                    help="std [rad] of heading error at release")
    ap.add_argument("--base-rp", type=float, default=0.0,
                    help="std [rad] of roll/pitch error at release")
    ap.add_argument("--sense", default=None,
                    help="state-estimate error, 'k=v,k=v' over "
                         "base_p,base_r,base_v,base_w,q,q_bias,v.  The base "
                         "terms get a per-run bias plus white noise at a third "
                         "of it; the joint terms are white")
    ap.add_argument("--mu-scale", type=float, default=1.0,
                    help="scale the PLANT's friction; the plan still assumes mu")
    ap.add_argument("--table-shift", default="0,0,0",
                    help="move the real table [m] while the plan keeps its "
                         "nominal one -- i.e. a perception error")
    ap.add_argument("--schedule-shift", type=int, default=0,
                    help="advance (+) or retard (-) the prescribed contact "
                         "schedule by this many control periods")
    ap.add_argument("--settle", type=int, default=0,
                    help="control periods held on the plan's first node before "
                         "the maneuver advances (the winch letting go)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--mpc-horizon", type=int, default=40)
    ap.add_argument("--mpc-iters", type=int, default=2)
    ap.add_argument("--mpc-threads", type=int, default=0,
                    help="ShootingProblem.nthreads for the online problem "
                         "(0 = crocoddyl's default).  Inert unless libcrocoddyl "
                         "was built with OpenMP -- see the S16 docpage.")
    ap.add_argument("--mpc-dt-scale", type=int, default=1,
                    help="integrate the ONLINE problem at this multiple of the "
                         "control period.  The window then advances one coarse "
                         "node every n periods -- same preview, n times fewer "
                         "nodes.  See MPC.COARSE for why a fine head plus a "
                         "coarse tail is not the cheaper option it looks like")
    ap.add_argument("--mpc-alphas", type=int, default=0,
                    help="truncate the FDDP line-search ladder to this many "
                         "rungs (0 = crocoddyl's default 10).  Bounds the "
                         "rollouts per iteration, and so the worst-case step.")
    ap.add_argument("--tau-clamp", action="store_true",
                    help="clamp the PLANT's actuator force to the clamp basis, "
                         "so the position servo cannot push a joint past the "
                         "safety limit the plan is solved against")
    ap.add_argument("--mpc-no-cones", action="store_true",
                    help="drop the friction/wrench cone costs from the ONLINE "
                         "problem only (the plan keeps them).  Also switches "
                         "enable_force off, since the cones are the only costs "
                         "that read the contact-force derivatives.")
    ap.add_argument("--cam", default="wide", choices=sorted(CAMERAS),
                    help="camera preset for --video")
    ap.add_argument("--no-ghost", action="store_true",
                    help="keep the bracing arm opaque in the video (the default "
                         "makes it translucent so the contact markers under it "
                         "are visible)")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    tag = args.tag or args.mode.replace("+", "_")
    name = f"replay_{tag}_{args.ctrl}{args.suffix}"
    vid = os.path.join(args.dir, name + ".mp4") if args.video else None
    log, plan = replay(tag, ctrl_mode=args.ctrl, run_dir=args.dir, video=vid,
                       qpos_out=os.path.join(args.dir, name + "_q.npy"),
                       fps=args.fps, speed=args.speed, push=args.push,
                       push_at=args.push_at, push_hold=args.push_hold,
                       push_dir=[float(v) for v in args.push_dir.split(",")],
                       seed=args.seed, q_noise=args.q_noise,
                       base_xy=args.base_xy, base_z=args.base_z,
                       base_yaw=args.base_yaw, base_rp=args.base_rp,
                       settle=args.settle,
                       sense=(dict((k, float(v)) for k, v in
                                   (kv.split("=") for kv in
                                    args.sense.split(",")))
                              if args.sense else None),
                       mu_scale=args.mu_scale,
                       table_shift=[float(v) for v in
                                    args.table_shift.split(",")],
                       schedule_shift=args.schedule_shift,
                       mpc_horizon=args.mpc_horizon, mpc_iters=args.mpc_iters,
                       mpc_cones=not args.mpc_no_cones,
                       mpc_alphas=args.mpc_alphas,
                       mpc_threads=args.mpc_threads,
                       mpc_dt_scale=args.mpc_dt_scale,
                       tau_clamp=args.tau_clamp,
                       cam=args.cam, ghost=not args.no_ghost,
                       dt_plan=plan_dt(args.dir, tag))
    res = summarise(log, plan, args.ctrl)
    out = os.path.join(args.dir, name + ".json")
    with open(out, "w") as fh:
        json.dump({"mode": plan["mode"], "tag": tag, "ctrl": args.ctrl,
                   "push": args.push, "push_hold": args.push_hold,
                   "q_noise": args.q_noise, "seed": args.seed,
                   "summary": res, "log": log}, fh, indent=1)
    print(f"\nwrote {out}")


def plan_dt(run_dir, tag):
    with open(os.path.join(run_dir, f"plan_{tag}.json")) as fh:
        return json.load(fh)["dt"]


if __name__ == "__main__":
    main()
