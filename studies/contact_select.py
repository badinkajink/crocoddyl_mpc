#!/usr/bin/env python3
"""Static contact selection for the H12 braced lean.

For a reach target (x, y, z) and a candidate set of bracing contacts on the right
arm, this:

  1. solves IK for a lean pose that puts the LEFT hand on the target, keeps both
     feet pinned, and places each selected bracing site on the table top;
  2. solves a static-equilibrium QP at that pose for the contact forces and joint
     torques, subject to linearized friction cones, unilateral normal forces, and
     the real actuator torque limits;
  3. reports feasibility and cost for every non-empty subset of the candidate
     contacts, plus the legs-only case.

Formulation
-----------
Static equilibrium of the floating-base robot (qvel = qacc = 0):

    S' tau  +  sum_i J_i' lambda_i  =  g(q)

with g(q) = d.qfrc_bias at rest (pure gravity).  NOT mj_inverse, which folds in
qfrc_constraint -- whatever the pose is penetrating.  Splitting the
6 unactuated floating-base rows from the 27 actuated rows:

    base   :  sum_i J_i[:, :6]'  lambda_i  =  g[:6]        (hard constraint)
    joints :  tau = g[6:] - sum_i J_i[:, 6:]' lambda_i     (bounded by tau_max)

Decision variables are the contact forces lambda_i in R^3 (world frame), one per
contact point: 4 corners per foot (8) plus one per selected bracing site.

Objective:  w_tau * ||tau / tau_max||^2  +  w_lam * ||lambda||^2
The first term is the thing we actually care about -- normalized actuator effort,
so a knee at 129/300 counts the same as a wrist at 4/9.5.  The second is the
min-norm regularizer that makes load distribution across redundant contacts
well-posed (and is what should make triple contact emerge).
"""

import itertools
import json
import sys

import numpy as np
import mujoco
from scipy.optimize import lsq_linear

import os
# Where the STAGED lean task lives.  Defaults to this checkout's own build tree
# (../build/... relative to this file) so the study runs on any machine that has
# run `studies/stage_assets.sh` or a real MJPC build; LEAN_TASK_DIR
# overrides it.  Was hardcoded to one developer's absolute path, which is why
# nothing in here ran anywhere else.
_LEAN_DIR = os.environ.get(
    "LEAN_TASK_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "build/mjpc/tasks/humanoid_bench/lean")) + "/"
# LEAN_MODEL=magpie selects the variant that carries the real magpie grippers
# (0.506 kg each, with box collision proxies + jaws).  The base model is the
# handless robot: its "palm" is a wrist capsule + sphere, NOT a gripper.
_VARIANTS = {
    "handless": "Lean_H12.xml",              # no grippers at all
    "magpie":   "Lean_H12_Magpie.xml",       # box proxies + jaw boxes (current)
    "sphere":   "Lean_H12_Magpie_sphere.xml",# single r=32mm sphere per gripper
    "mesh":     "Lean_H12_Magpie_mesh.xml",  # visual mount mesh IS the collision mesh
}
# Default moved handless -> magpie (2026-08-04): the real machine wears the
# grippers, and only the Magpie model carries the brace keyframes the IK now
# seeds from (see SEED_KEY).  LEAN_MODEL=handless reproduces the S1-S10 runs.
MODEL = _LEAN_DIR + _VARIANTS.get(os.environ.get("LEAN_MODEL", "magpie"),
                                  "Lean_H12_Magpie.xml")

# WHICH ARM BRACES.  The study started with the right arm bracing and the left
# reaching, matching the old model's reach_hand=2.  Allen's 2026-07-31 setup is
# the OTHER handedness -- reach_target y = -0.2348 puts the object on the robot's
# right, so the RIGHT hand reaches and the LEFT arm braces -- and his
# forearm_brace_lean keyframe (now the IK seed) is posed that way.
#
# Getting this backwards is not a cosmetic mismatch: seeding from his pose while
# bracing the right arm asks the solver to lift the already-braced left arm off
# the table and put the right one down across the body. Under that crossed
# configuration NO elbow/forearm set placed at ANY target on the 0.80-1.40 sweep,
# which reads exactly like "the forearm brace is geometrically impossible on the
# new table" and is not.  BRACE_ARM=right restores the original convention.
BRACE_ARM = os.environ.get("BRACE_ARM", "left")
_OTHER = "right" if BRACE_ARM == "left" else "left"
# The H1-2 arms are mirror-symmetric, so the right-arm offsets below carry over
# by negating y.  Sign lives here rather than in two hand-written tables.
_MY = 1.0 if BRACE_ARM == "right" else -1.0

# SITE GEOMETRY REVISION (2026-08-04, S11).  v1 is the S1-S10 geometry; v2 is
# the user's rework.  SITE_SET=v1 reproduces every earlier run.
#
# Two complaints, both correct, both about the same thing -- the three arm sites
# were not three DISTINGUISHABLE places on the arm:
#
#  (1) forearm sat 50 mm along the forearm link, i.e. ~57 mm from the elbow site
#      at the seed pose.  Geometrically valid (a rigid link laid on a table does
#      touch at both) but it does not read as a separate contact, and the two
#      share almost the same moment arm, so the QP has nothing to choose between.
#      v2 moves it to 110 mm -- the DISTAL END of `left_forearm_pad`, the capsule
#      that is the forearm's actual brace surface (fromto 0.02 -> 0.11).  That is
#      as far distal as the modelled pad goes; past it is the wrist roll joint at
#      0.121.  Separation at the seed: 57 mm -> 115 mm.
#
#  (2) palm was a point on the wrist_yaw_link AXIS, 130 mm out -- which is the
#      wrist housing, not the hand, and it is driven down by wrist PITCH.  The
#      real machine is meant to brace on the SIDE OF THE GRIPPER.  v2 puts the
#      site on the `*_magpie_gripper` body at the centre of the flat side face
#      of `*_gripper_collision` (box half-extents .0425/.0315/.0667 at x=.0965;
#      local +-y IS the flat side, the thin 63 mm axis; local +-z is the jaw
#      separation axis).  Getting that face onto the table needs the wrist ROLLED
#      ~90 deg, which is what PALM_ALIGN in solve_ik does -- and once rolled, the
#      joint that drives the face into the table is wrist YAW, not wrist pitch.
#      Measured (wrist_probe.py): yaw travel 146 deg vs pitch 53 deg for the same
#      19 Nm limit, and rolling takes the yaw axis from |axis.y| = 0.29 (yaw
#      swings the hand sideways) to 0.73-0.88 (yaw pitches it into the table).
SITE_SET = os.environ.get("SITE_SET", "v2")

SITES = {
    # NOTE: body-frame z of the elbow JOINT on the upper-arm link is -0.182.
    # The first version of this used -0.250, taken from the collision geom's
    # `geom_size`, which for a MESH is the BOUNDING-BOX half-extent, not the
    # shape -- so the point floated ~68 mm past the joint in empty space and the
    # site was placeable only 1/12.  At the joint it places 12/12.
    "elbow":   ("%s_shoulder_yaw_link" % BRACE_ARM,
                np.array([0.002,  _MY * 0.007, -0.182])),
    "forearm": ("%s_elbow_link" % BRACE_ARM,
                np.array([0.110,  _MY * 0.030, -0.015]) if SITE_SET == "v2"
                else np.array([0.050,  _MY * 0.021, -0.007])),
    "palm":    (("%s_magpie_gripper" % BRACE_ARM,
                 np.array([0.0965, _MY * 0.0315, 0.000])) if SITE_SET == "v2"
                else ("%s_wrist_yaw_link" % BRACE_ARM,
                      np.array([0.130,  0.000,  0.000]))),
    # Trunk sites (added S10).  The pose audit found the hips touching the table
    # edge in ~80% of solved poses and the torso in ~60%, all unmodelled -- so
    # they are promoted to first-class candidates.  Both anchors are taken from
    # PRIMITIVE geoms on torso_link, where `size` is the true shape (capsule
    # radius / box half-extent), NOT a mesh bounding box -- see the S6.1 elbow bug.
    #   "hip"   : front surface of the `hip` capsule  (local [0,0,-0.05], r=0.05)
    #   "torso" : lower front edge of the `torso` box (local [0,0,0.25], hx=0.07)
    "hip":     ("torso_link",              np.array([0.050,  0.000, -0.050])),
    "torso":   ("torso_link",              np.array([0.070,  0.000,  0.050])),
}
# THE PALM SITE AND ITS BRACE FACE ARE DERIVED, NOT DECLARED (2026-08-16).
#
# The two values above for "palm" were (0.0965, +-0.0315, 0) with the aligner
# below driving the gripper's local +-Y face at the wood. Both were read off the
# HAND-AUTHORED gripper proxy, and that proxy turned out to be transposed 90 deg
# and 25 mm short (see _gen_magpie_gripper.py). Against the corrected CAD:
#
#   * the site lands 26 mm short in x and 18.1 mm INSIDE the gripper body, so
#     the static QP was certifying a contact point that lives in solid
#     geometry. It certified happily -- a point has no thickness -- and then the
#     dynamic replay had to drive the box 18 mm through the table to put that
#     point on it. Measured: elbow+palm cells certify, then fall in the NOMINAL
#     replay (0% survival, no disturbance at all) with `penetration` failures.
#   * the widest face flips. Old half-extents (0.0415, 0.0170, 0.0667) put the
#     83x133 mm face normal to Y; corrected (0.0415, 0.0667, 0.0170) puts it
#     normal to Z. So the aligner was laying the 83x34 mm EDGE on the table and
#     calling it a palm brace. This is S15's "133 mm across against 34", with
#     the two faces swapped.
#
# Hard-coding either number again would just re-arm the same trap, so both are
# now read off `*_gripper_collision` at load(): the site is the centre of the
# widest face, on its outward side, and PALM_AXIS is that face's normal.
PALM_AXIS = np.array([0.0, 0.0, -1.0])      # set by _resolve_palm(); local frame


def _resolve_palm(m):
    """Point the palm site and the align axis at the gripper's widest face.

    Returns (offset, axis) in the gripper BODY frame and updates the module
    globals, so every consumer of SITES["palm"] / PALM_AXIS follows the model.
    """
    global PALM_AXIS
    body = "%s_magpie_gripper" % BRACE_ARM
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                          "%s_gripper_collision" % BRACE_ARM)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if g < 0 or b < 0 or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
        return SITES["palm"][1], PALM_AXIS      # handless / sphere variants
    # geom pose in the body frame (the geom carries its own pos/quat)
    d0 = mujoco.MjData(m)
    mujoco.mj_forward(m, d0)
    R = d0.xmat[b].reshape(3, 3)
    c = R.T @ (d0.geom_xpos[g] - d0.xpos[b])
    h = m.geom_size[g][:3].copy()
    # widest face: the axis whose PERPENDICULAR pair has the largest area
    areas = [4 * h[(i + 1) % 3] * h[(i + 2) % 3] for i in range(3)]
    k = int(np.argmax(areas))
    # brace on the face pointing DOWN in the gripper's own frame, i.e. -k
    off = c.copy()
    off[k] -= h[k]
    axis = np.zeros(3)
    axis[k] = -1.0
    PALM_AXIS = axis
    SITES["palm"] = (body, off)
    return off, axis


ARM_SITES = ("elbow", "forearm", "palm")
TRUNK_SITES = ("hip", "torso")
REACH_BODY = "%s_wrist_yaw_link" % _OTHER
REACH_OFF = np.array([0.13, 0.0, 0.0])
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]

IK_MARGIN = 0.025   # collision look-ahead for the IK [m]

# SEED POSE.  solve_ik pins the feet AT THE SEED and pulls the nullspace back
# toward it, so the seed is not a warm start -- it selects which basin is being
# searched, and on the 2026-07-31 table it decides whether a brace is findable
# at all.  That table sits 12 cm higher and its near edge 20 cm further out than
# the slab the study started on, and from `stand_up` the IK simply cannot get an
# elbow or a forearm down onto it: sites land 40-55 mm short at every target
# tried, so the QP scores contacts that are not there.
#
# Allen's brace poses are retuned for exactly this table (pelvis dropped to 0.884,
# torso pitched ~32 deg, hips -1.14, knees +1.10) and exist ONLY in
# Lean_H12_Magpie.xml -- which is also the deploy path, so the model default moves
# with the seed.  LEAN_MODEL=handless SEED_KEY=stand_up recovers the S1-S10 runs.
#
# Of his three, `forearm_brace_reach` is the one to seed from, and the difference
# is not subtle.  All three have the brace down (elbow/forearm 16-29 mm above the
# table), but in `..._lean` and `..._release` the REACHING hand hangs 23-27 cm
# BELOW the table surface and behind its near edge.  From there the straight-line
# path to a target on the tabletop runs through the slab, and a local Gauss-Newton
# IK with hard non-penetration cannot route around it -- it has to move AWAY from
# the target (up, over the edge) before it can move toward it.  Measured: 250
# iterations, zero QP fallbacks, reach residual pinned at its seed value of 0.52 m.
# That looks exactly like "the target is unreachable" and is not.
# `..._reach` starts the hand 15 cm ABOVE the table with the brace already
# established, i.e. the phase this study is actually about.
SEED_KEY = os.environ.get("SEED_KEY", "forearm_brace_reach")

# STANCE OFFSET (2026-08-04, S11).  A rigid translation of the WHOLE robot in the
# table frame, applied to the free joint after the seed keyframe is loaded.  The
# feet ride along with it and solve_ik reads its pinned foot targets from the
# shifted pose, so this is exactly "stand somewhere else", not "move the feet".
#
# Why it is a knob at all: the study's stance is square-on and centred on the
# table's centreline (feet y = +-0.2346, table y = +-0.2975), so the LEFT bracing
# arm comes in across the table's near corner and its distal half runs out of
# table sideways before it runs out of table forwards -- the table is 0.595 m wide
# and 1.18 m deep, and the arm is using the narrow axis.  Shifting the robot in -y
# swings the bracing shoulder toward the table centreline and lets the arm lie
# along the DEEP axis, which is what "decenter the stance for whole-table contact"
# buys.  It also moves the kinematic reach wall, which is the thing that actually
# caps the braced reach envelope (see reach_diag.py).
STANCE_DX = float(os.environ.get("STANCE_DX", "0.0"))
STANCE_DY = float(os.environ.get("STANCE_DY", "0.0"))

# --------------------------------------------------------------------------- #
# TORQUE BASIS -- which limit the QP is allowed to spend.  Three real, different
# authorities, so this has to be a named choice rather than "whatever the model
# carries".  Set with TAU_BASIS=model|urdf|clamp (default clamp).
#
#   model  min(<position forcerange>, jnt_actfrcrange) as built.  MuJoCo enforces
#          this, and for the arms the forcerange (32/14.4/9.5, an MJPC default
#          class) binds BELOW the imported CL actuatorfrcrange (40/18/19).
#          NOTE this corrects S10.2/S10.3 of the plan file, which read the low
#          numbers as the CL import having failed to propagate actuatorfrcrange.
#          It propagates correctly -- verified generated == CL on all 27 joints,
#          including the right_shoulder_yaw mirror.  The forcerange is the binder.
#   urdf   URDF/CL_Assets actuatorfrcrange == h12_safety_layer URDF_TORQUE_LIMITS
#          == what the full-body deploy node patches the ARMS to (FRC_LIMIT).
#   clamp  0.9 x TAU_ESTOP, the torque-budget clamp h12_control_node actually
#          enforces on every commanded joint (deploy_common.h kClampRatio).
#
# `clamp` is the default because it is the only one the robot cannot exceed: the
# node refuses anything above it, so a plan that needs more is not executable.
# Allen applied exactly this to the ankle (48.6 = 0.9 x 54).
#
# TORSO, resolved 2026-08-05 (user decision).  The node's TAU_ESTOP table carries
# 40 Nm for the torso and that number matches NO safety-layer config: the header
# calls the table "estop torque_ratio x URDF from default_safety_full.yaml", but
# that config's estop torque_ratio is 1.0 and joint_limits.py's URDF torque limit
# for torso_joint is 200, so the traceable value is 200.  (Its clip ratio 0.25
# gives 50, tight_safety_full gives its own, and none of them give 40.)  Untraced
# numbers do not get to bound a result, so the torso entry is now the
# default_safety_full ESTOP -- 200 Nm, i.e. 180 under the 0.9 clamp.
#
# It changes nothing, and that is the useful part.  torso_basis.py re-solves the
# braced pose under both ceilings: the torso carries 1.0-1.9 Nm, ~1% of even the
# OLD 36 Nm clamp, and peak ratio / region margin agree to four decimals across
# the two bases.  The lean is carried by the reaching arm's shoulder_pitch
# (0.46-0.47, the binding joint at every subset tested) and the hips (0.29-0.32),
# not by the torso.  So the S11 flag that a tight-config deployment would e-stop
# the torso at 56% of budget was aimed at a joint that is nearly idle; the joint
# that actually binds is right_shoulder_pitch, at a 29 Nm clamp.
#
# The 50 Nm clip is a real but DIFFERENT constraint: it clips `motor_cmd.tau`,
# the feedforward channel only.  h12_control_node commands position + kp/kd, and
# the PD term is formed on the motor driver downstream of the safety layer, so
# the joint can make 180 Nm without ever putting 180 in the clipped field.  What
# would actually stop it is the estop watching measured tau_est at 200.
_LEG = ["hip_yaw", "hip_pitch", "hip_roll", "knee", "ankle_pitch", "ankle_roll"]
_ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
        "wrist_roll", "wrist_pitch", "wrist_yaw"]
TAU_ESTOP = dict(zip(_LEG, [60., 130., 200., 300., 54., 36.]))
TAU_ESTOP["torso"] = 200.
TAU_ESTOP.update(dict(zip(_ARM, [32., 32., 14.4, 14.4, 9.5, 9.5, 9.5])))
TAU_URDF = dict(zip(_LEG, [200., 200., 200., 300., 60., 40.]))
TAU_URDF["torso"] = 200.
TAU_URDF.update(dict(zip(_ARM, [40., 40., 18., 18., 19., 19., 19.])))
CLAMP_RATIO = 0.9
TAU_BASIS = os.environ.get("TAU_BASIS", "clamp")


def _base_joint(name):
    """'right_wrist_pitch_joint' -> 'wrist_pitch'."""
    n = name.replace("_joint", "")
    for side in ("left_", "right_"):
        if n.startswith(side):
            n = n[len(side):]
    return n


def torque_limits(m, basis=None):
    """Per-actuator torque ceiling under the selected basis.  Always min'd with
    what the model can physically emit, so a basis can only ever TIGHTEN."""
    basis = basis or TAU_BASIS
    model = np.array([
        min(m.actuator_forcerange[i, 1] if m.actuator_forcelimited[i] else np.inf,
            m.jnt_actfrcrange[m.actuator_trnid[i, 0], 1]
            if m.jnt_actfrclimited[m.actuator_trnid[i, 0]] else np.inf)
        for i in range(m.nu)])
    if basis == "model":
        return model
    table = TAU_URDF if basis == "urdf" else TAU_ESTOP
    scale = 1.0 if basis == "urdf" else CLAMP_RATIO
    out = model.copy()
    for i in range(m.nu):
        jn = _base_joint(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
        if jn in table:
            out[i] = min(model[i], scale * table[jn]) if basis != "urdf" \
                     else scale * table[jn]
    return out


# Friction coefficient at the table and the floor.  0.6 was a placeholder nobody
# measured, and every slip number in S1-S10 rests on it.  It is now settable so
# mu can be SOLVED for rather than assumed: friction_sweep.py finds mu*, the
# smallest coefficient at which friction stops being the binding constraint and
# the joint-torque envelope takes over.  Above mu* the results characterise the
# actuators; below it they characterise the tape on the table.
MU = float(os.environ.get("MU", "0.6"))
N_ROBOT_DOF = 33    # 6 floating base + 27 joints; DOFs 33+ are the object anchor
N_ACT = 27


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def load(ik_margin=IK_MARGIN):
    """Load the model.  `ik_margin` inflates every geom's collision margin so
    MuJoCo reports near-contacts BEFORE they overlap.  Without it the IK only
    learns about a collision once it is already inside the table, and repairing
    that within one trust-region step is often infeasible -- measured 245 of 250
    iterations infeasible, i.e. non-penetration silently dropped.  With a margin
    the constraint is predictive and the solver never enters the hole."""
    m = mujoco.MjModel.from_xml_path(MODEL)
    if ik_margin:
        m.geom_margin[:] = np.maximum(m.geom_margin, ik_margin)
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, SEED_KEY)
    if key < 0:
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand_up")
    if key >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key)
    if STANCE_DX or STANCE_DY:
        d.qpos[0] += STANCE_DX      # free-joint translation == move the whole
        d.qpos[1] += STANCE_DY      # robot, feet included (see STANCE_DX above)
    mujoco.mj_forward(m, d)
    _resolve_palm(m)          # site + brace face follow the model, never a constant
    return m, d


def start_qpos(m, key):
    """A named keyframe with the STANCE OFFSET applied, as `load` would apply it.

    `load` shifts the robot after resetting to SEED_KEY, but the crocoddyl track
    reads its START pose out of a DIFFERENT keyframe (`stand`) and did so by
    indexing `m.key_qpos` directly -- so every plan silently ignored the stance
    offset while the IK that produced its q* honoured it.  That combination
    plans a maneuver from a stance the robot is not in.  Both paths go through
    here now.
    """
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        raise KeyError(f"no keyframe {key!r}")
    q = m.key_qpos[kid].copy()
    q[0] += STANCE_DX          # free-joint translation: the whole robot,
    q[1] += STANCE_DY          # feet included.  The table does not move.
    return q


def bid(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def point_world(m, d, body, offset):
    b = bid(m, body)
    return d.xpos[b] + d.xmat[b].reshape(3, 3) @ offset


def point_jac(m, d, body, offset):
    """3 x nv positional Jacobian of a body-fixed point."""
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, jacp, None, point_world(m, d, body, offset), bid(m, body))
    return jacp


def foot_corners(m, d, foot):
    """4 world-frame corner points of a foot, on its sole."""
    b = bid(m, foot)
    # foot collision proxy is centred at (0.033, 0, -0.019) in the ankle frame;
    # sole plane sits at the lowest z the body reaches.
    half_x, half_y = 0.10, 0.04
    z = -0.055
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            out.append((foot, np.array([0.033 + sx * half_x, sy * half_y, z])))
    return out


def resolved_contacts(m, d, subset):
    """The contacts the QP should actually be solving for.

    Until 2026-08-04 the QP placed a point force at the nominal SITE, e.g. the
    elbow joint's position on the upper-arm link.  That is not where the contact
    is: the site sits on the link AXIS and the link touches the table on its
    collision capsule's SURFACE.  Measured offsets between the site and MuJoCo's
    own contact point at solved brace poses: elbow 47 mm, forearm 41-56 mm,
    palm 32 mm, hip 104-129 mm.  Every force, torque, slip margin and support
    region computed at the site therefore carried a moment-arm error of that size,
    worst exactly on the trunk contacts the trunk-vs-arm argument turns on.

    It also assumed every normal is world-vertical.  On the 2026-07-31 table the
    brace lands on the near EDGE (contacts cluster at x = 0.50, the near edge, and
    y = 0.297, the side edge), where the normal is not vertical at all -- so the
    friction cone was being built around the wrong axis on the very contacts whose
    slip margin is in question.

    So: take the point AND the frame from MuJoCo's narrowphase where a contact
    exists, and fall back to the site with a vertical normal where it does not
    (which is what the feet always use -- flat floor, and the corner model is a
    deliberate 4-point approximation of a sole, not a narrowphase result).

    Returns a list of (body_id, world_point, R) where R's ROWS are (n, t1, t2)
    and n points into the robot, so lam_n >= 0 is "push, do not pull".
    """
    tbl = bid(m, "table")
    found = {}
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl:
            robot_b, sign = b2, +1.0     # frame normal points geom1 -> geom2
        elif b2 == tbl:
            robot_b, sign = b1, -1.0
        else:
            continue
        prev = found.get(robot_b)
        if prev is None or con.dist < prev[0]:
            R = con.frame.reshape(3, 3).copy()
            R[0] *= sign                 # normal into the robot
            found[robot_b] = (con.dist, con.pos.copy(), R)

    VERT = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    out = []
    for f in FEET:
        for body, off in foot_corners(m, d, f):
            out.append((bid(m, body), point_world(m, d, body, off), VERT.copy()))
    for s in subset:
        body, off = SITES[s]
        b = bid(m, body)
        if b in found:
            _, p, R = found[b]
            out.append((b, p, R))
        else:
            out.append((b, point_world(m, d, body, off), VERT.copy()))
    return out


def table_top_z(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    return d.geom_xpos[g][2] + m.geom_size[g][2]


def table_x_range(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    cx = d.geom_xpos[g][0]
    return cx - m.geom_size[g][0], cx + m.geom_size[g][0]


def table_y_range(m, d):
    """Lateral extent of the table top.

    This was not constrained until 2026-08-04 and did not need to be: the old
    slab was 2.12 m wide, so a bracing site was inside it in y no matter what.
    Allen's real adjustable table is 0.595 m wide -- the SHORT side faces the
    robot -- and the IK promptly started parking sites at exactly the right
    height just off the side of it. They then read as "placed" on the z residual
    (1.4 mm!) while MuJoCo's narrowphase correctly reported no contact at all,
    which is what the sites_placed / site-residual disagreement was."""
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    cy = d.geom_xpos[g][1]
    return cy - m.geom_size[g][1], cy + m.geom_size[g][1]


# --------------------------------------------------------------------------- #
# IK: lean pose for a reach target + a set of bracing sites on the table
# --------------------------------------------------------------------------- #
TORSO_DOF = 18          # dofadr of torso_joint
W_FOOT, W_REACH, W_SITE = 300.0, 4.0, 6.0   # feet are pinned: they must not slide
W_COLL = 40.0           # collision rows dominate: non-penetration is not optional
W_POST = 0.6            # nullspace pull back toward the seed pose
# Palm-face alignment.  Same order as W_SITE: getting the face flat matters as
# much as getting it to the right height, because a face that is 30 deg off is
# an edge contact, and an edge contact is a different (and worse) contact.
W_PALM_ALIGN = 6.0
PALM_ALIGN = os.environ.get("PALM_ALIGN", "1") == "1"
# Forearm heading (see forearm_heading_rows).  Weight is deliberately BELOW
# W_SITE: getting the brace onto the table still outranks pointing it prettily,
# so on targets where the two conflict the contact wins and the heading degrades
# gracefully instead of the pose failing to place.
W_FOREARM = 3.0
FOREARM_ALIGN = os.environ.get("FOREARM_ALIGN", "1") == "1"
# Desired forearm compass heading [deg] in the table plane. 0 = along the table's
# LONG axis (world +x).  Negative skews the forearm inward, toward the table's
# far side, which is the "or even skewed in" case.
FOREARM_HEADING = float(os.environ.get("FOREARM_HEADING", "0.0"))
PEN_TOL = 0.002         # allowed contact depth before a collision row fires
TRUST = 0.05            # per-step trust region [rad/m], enforced INSIDE the QP
LEVENBERG = 1e-6        # keeps the QP Hessian positive definite


def collision_rows(m, d, allow_bodies):
    """One row per penetrating robot contact, pushing along the contact normal.

    Uses MuJoCo's own narrowphase (mj_forward has already run), so it sees exactly
    the collision proxies the planner sees.

    NOTE: robot bodies are never exempted, not even the feet or the selected
    bracing site.  Exempting a site's body lets the whole link sink through the
    table (measured 47 mm).  Instead a 2 mm tolerance lets a resting contact sit
    at dist ~ 0 without fighting the site-on-plane task.
    """
    rows, errs = [], []
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist >= IK_MARGIN:
            continue                        # outside the look-ahead band
        b1 = m.geom_bodyid[con.geom[0]]
        b2 = m.geom_bodyid[con.geom[1]]
        if _is_env(m, b1) and _is_env(m, b2):
            continue                                    # scenery vs scenery
        # which side is the robot?
        robot_b, sign = (b1, -1.0) if b1 != 0 and not _is_env(m, b1) else (b2, 1.0)
        if _is_env(m, robot_b):
            continue
        n = con.frame[:3].copy()                        # contact normal, world
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, con.pos.copy(), robot_b)
        rows.append(sign * (n @ jacp).reshape(1, -1))
        errs.append(np.array([-con.dist]))              # push out to depth 0
    return rows, errs


def forearm_heading_rows(m, d, n_dof, heading_deg):
    """One Gauss-Newton row that aims the bracing forearm along a chosen compass
    heading in the table plane.

    Why this exists (user, 2026-08-04): the decentred stance bought whole-table
    access by translating the robot, and translation is a blunt instrument -- it
    swung the body 7.4 deg of roll onto the right leg, which then carried 81 % of
    body weight with its ankle pitch at 0.49 of the clamped limit.  The thing
    that actually needed fixing was the forearm's DIRECTION: the table is 0.595 m
    wide and 1.18 m deep, so a forearm lying along +x fits and a forearm at 25 deg
    runs off the side.  Aiming the forearm directly is the targeted version of the
    same fix, and it should let the stance come back toward centre.

    HEADING ONLY, deliberately.  Constraining the forearm's full 3-D direction
    would also pin its pitch, which is the freedom the brace needs to get down
    onto the table at different reach distances.  So the task is on the
    projection: with v the forearm axis in world and theta = atan2(v_y, v_x),

        r = wrap(heading - theta)
        dtheta/domega = [ -v_x v_z,  -v_y v_z,  v_x^2 + v_y^2 ] / (v_x^2 + v_y^2)

    from  vdot = omega x v.  One row, one scalar residual: this REPLACES the
    stance offset rather than adding to the objective stack.
    """
    a = point_world(m, d, "%s_elbow_link" % BRACE_ARM, np.zeros(3))
    b = point_world(m, d, "%s_wrist_roll_link" % BRACE_ARM, np.zeros(3))
    v = b - a
    nv2 = v[0] ** 2 + v[1] ** 2
    if nv2 < 1e-9:                       # forearm is vertical: heading undefined
        return np.zeros((1, n_dof)), np.zeros(1)
    theta = np.arctan2(v[1], v[0])
    err = np.arctan2(np.sin(np.radians(heading_deg) - theta),
                     np.cos(np.radians(heading_deg) - theta))
    dtheta = np.array([-v[0] * v[2], -v[1] * v[2], nv2]) / nv2
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, None, jacr, b, bid(m, "%s_wrist_roll_link" % BRACE_ARM))
    return (dtheta @ jacr[:, :n_dof]).reshape(1, -1), np.array([err])


def palm_align_rows(m, d, n_dof):
    """Gauss-Newton rows that roll the wrist until the gripper's flat SIDE face
    lies against the table.

    The face's outward normal is PALM_AXIS, DERIVED from the gripper's
    collision box (see _resolve_palm) rather than assumed to be +-y.  It
    must end up pointing at the table, i.e. along world -z.  Writing a for that
    body-fixed axis in world and t = (0,0,-1), the residual is

        r = a x t        ( |r| = sin(angle), r = 0 exactly when a || t )

    and, since a body-fixed axis obeys  da/dt = w x a  for the body's angular
    velocity w,

        dr/dt = (w x a) x t = a (t.w) - w (t.a) = [ a t' - (a.t) I ] w

    so the row block is  M @ Jr  with  M = a t' - (a.t) I  and Jr the body's
    ROTATIONAL Jacobian.  (Using 1 - a.t as a scalar cost instead has zero
    gradient at BOTH the aligned and the anti-aligned pose, so the solver can
    park the face pointing at the ceiling; the cross-product residual does not
    have that stationary point on the wrong side.)

    Which of the two faces is asked to go down is decided ONCE per call by
    whichever is already nearer -- the wrist has 344 deg of roll travel, so both
    are reachable and picking the near one just avoids a needless half-turn.
    """
    body, _ = SITES["palm"]
    b = bid(m, body)
    R = d.xmat[b].reshape(3, 3)
    t = np.array([0.0, 0.0, -1.0])
    a = R @ (-PALM_AXIS)          # outward normal of the widest face (derived)
    if a @ t < 0:
        a = -a                       # the other face is the near one
    M = np.outer(a, t) - (a @ t) * np.eye(3)
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, None, jacr, d.xpos[b], b)
    return M @ jacr[:, :n_dof], -np.cross(a, t)


_ENV_CACHE = {}


def _is_env(m, b):
    """True for world/table bodies (everything that is not the robot)."""
    if b in _ENV_CACHE:
        return _ENV_CACHE[b]
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    v = (b == 0) or ("table" in n) or ("object" in n)
    _ENV_CACHE[b] = v
    return v


def solve_ik(m, d, target, subset, iters=250, tol=2e-3, lock_torso=False,
             lock_joints=None, soft_joints=None):
    """Gauss-Newton IK over the 33 robot DOFs, one QP per step.

    Each step solves, for the joint-velocity-like step dq:

        min  1/2 dq' G dq - a' dq
        s.t. J_foot dq = e_foot                     (feet are pinned: HARD)
             n' J_c dq >= -(dist + PEN_TOL)         (non-penetration: INEQUALITY)
             -TRUST <= dq_i <= TRUST                (trust region, in the QP)

    with G / a assembled from the weighted objective terms (reach, bracing sites,
    posture toward the seed) plus a Levenberg term.  This replaces the previous
    weighted normal equations + post-hoc step clipping, which did not descend
    (see plan file S3.3).  Weights are NOT the lever here and should not be tuned.

    `lock_torso=True` freezes torso_joint at its seed value -- the "assume we do
    not use the torso" case.  The shipped model does NOT lock it (only penalized,
    weight 30), so False is the faithful default.

    `soft_joints` is {joint_name: value_rad}, driven by a heavily weighted posture
    term instead of clamped.  Prefer it to `lock_joints` for the LEG chain: the
    feet are held by hip pitch + knee + ankle pitch together, so hard-clamping
    hip pitch removes a DOF the pinned-foot task needs and the feet drift ~40 mm
    (measured) -- which silently turns a reach result into a foot-sliding result.
    A strong soft target answers the same physical question without over-
    constraining the stance; the ACHIEVED angle is returned so the commanded and
    realised values can be compared rather than assumed equal.

    `lock_joints` is {joint_name: value_rad}, hard-clamped after every integration
    step.  It exists so HIP PITCH can be treated as the independent variable it
    physically is: the user's model of this task is that contacts do not lengthen
    the arm, they raise the stability ceiling, which permits more hip flexion,
    which is what actually carries the hand forward.  Testing that needs hip pitch
    COMMANDED and reach MEASURED, which is the opposite of the target-grid sweep.

    Returns dict of per-task residuals: reach, foot, site_max, penetration, ok.
    """
    import quadprog

    q_seed = d.qpos.copy()
    foot_targets = []
    for f in FEET:
        b = bid(m, f)
        foot_targets.append((d.xpos[b].copy(), d.xquat[b].copy()))
    z_tab = table_top_z(m, d)
    x_lo, x_hi = table_x_range(m, d)
    y_lo, y_hi = table_y_range(m, d)
    jnt_dofs = np.arange(6, N_ROBOT_DOF)
    n = N_ROBOT_DOF

    parts = {"reach": np.inf, "foot": np.inf, "site": {}}
    stats = {"qp_fallback": 0}
    for _ in range(iters):
        mujoco.mj_forward(m, d)

        # ---- hard equality: both feet hold their seed pose -------------------
        Jeq, eeq, foot_err = [], [], []
        for f, (p_t, q_t) in zip(FEET, foot_targets):
            b = bid(m, f)
            jp = np.zeros((3, m.nv)); jr = np.zeros((3, m.nv))
            mujoco.mj_jac(m, d, jp, jr, d.xpos[b], b)
            ep = p_t - d.xpos[b]
            foot_err.append(np.linalg.norm(ep))
            Jeq.append(jp[:, :n]); eeq.append(ep)
            qerr = np.zeros(3); dq4 = np.zeros(4)
            mujoco.mju_mulQuat(dq4, q_t, np.array([d.xquat[b][0], -d.xquat[b][1],
                                                   -d.xquat[b][2], -d.xquat[b][3]]))
            mujoco.mju_quat2Vel(qerr, dq4, 1.0)
            Jeq.append(jr[:, :n]); eeq.append(qerr)
        Jeq = np.vstack(Jeq); eeq = np.concatenate(eeq)
        # only ask the step to remove what it can this iteration
        eeq = np.clip(eeq, -TRUST, TRUST)

        # ---- objective terms -------------------------------------------------
        G = np.zeros((n, n)); a = np.zeros(n)

        p = point_world(m, d, REACH_BODY, REACH_OFF)
        parts["reach"] = float(np.linalg.norm(target - p))
        Jr = point_jac(m, d, REACH_BODY, REACH_OFF)[:, :n]
        er = np.clip(target - p, -TRUST, TRUST)
        G += (W_REACH ** 2) * Jr.T @ Jr;  a += (W_REACH ** 2) * Jr.T @ er

        parts["site"] = {}
        for st in subset:
            body, off = SITES[st]
            ps = point_world(m, d, body, off)
            Js = point_jac(m, d, body, off)[:, :n]
            parts["site"][st] = float(abs(z_tab - ps[2]))
            rows_s = [Js[2:3]]; errs_s = [np.array([z_tab - ps[2]])]
            # keep the site ON the slab, not merely at its height: pull x and y
            # back inside the footprint whenever they leave it (see table_y_range)
            for ax, (lo_a, hi_a) in ((0, (x_lo, x_hi)), (1, (y_lo, y_hi))):
                if ps[ax] > hi_a:
                    rows_s.append(Js[ax:ax + 1])
                    errs_s.append(np.array([hi_a - ps[ax]]))
                elif ps[ax] < lo_a:
                    rows_s.append(Js[ax:ax + 1])
                    errs_s.append(np.array([lo_a - ps[ax]]))
            Js2 = np.vstack(rows_s); es2 = np.clip(np.concatenate(errs_s), -TRUST, TRUST)
            G += (W_SITE ** 2) * Js2.T @ Js2;  a += (W_SITE ** 2) * Js2.T @ es2

        # aim the forearm along the table's long axis (heading only)
        if FOREARM_ALIGN and ("forearm" in subset or "elbow" in subset):
            Jf, ef = forearm_heading_rows(m, d, n, FOREARM_HEADING)
            parts["forearm_heading_err"] = float(abs(ef[0]))
            ef = np.clip(ef, -TRUST, TRUST)
            G += (W_FOREARM ** 2) * Jf.T @ Jf
            a += (W_FOREARM ** 2) * Jf.T @ ef

        # roll the wrist so the gripper's flat side, not its edge, meets the wood
        if PALM_ALIGN and SITE_SET == "v2" and "palm" in subset:
            Ja, ea = palm_align_rows(m, d, n)
            parts["palm_align"] = float(np.linalg.norm(ea))
            ea = np.clip(ea, -TRUST, TRUST)
            G += (W_PALM_ALIGN ** 2) * Ja.T @ Ja
            a += (W_PALM_ALIGN ** 2) * Ja.T @ ea

        # posture pull toward the seed, on joint DOFs only
        dq_post = np.zeros(n)
        wp = np.zeros(n)
        for dof in jnt_dofs:
            j = int(np.argmax(m.jnt_dofadr == dof))
            adr = m.jnt_qposadr[j]
            dq_post[dof] = np.clip(q_seed[adr] - d.qpos[adr], -TRUST, TRUST)
            wp[dof] = W_POST
        if lock_torso:
            wp[TORSO_DOF] *= 100.0
        for jn, val in (soft_joints or {}).items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                continue
            dof = m.jnt_dofadr[jid]
            dq_post[dof] = np.clip(val - d.qpos[m.jnt_qposadr[jid]], -TRUST, TRUST)
            wp[dof] = 60.0
        G += np.diag(wp ** 2);  a += (wp ** 2) * dq_post

        G += LEVENBERG * np.eye(n)          # keeps G positive definite

        # ---- inequalities: non-penetration + trust region ---------------------
        Cin, bin_ = [], []
        for r, er_c in zip(*collision_rows(m, d, set())):
            # want dist + n'J dq >= -PEN_TOL, i.e. n'J dq >= depth - PEN_TOL.
            # Demanding more than the trust region can deliver makes the QP
            # infeasible, so ask for at most half of it per step and let
            # successive iterations walk a deep penetration out.
            want = min(float(er_c[0]) - PEN_TOL, 0.5 * TRUST)
            Cin.append(r[0, :n]); bin_.append(want)
        Cin.append(np.eye(n));  bin_.append(np.full(n, -TRUST))
        Cin.append(-np.eye(n)); bin_.append(np.full(n, -TRUST))
        Cin = np.vstack([c.reshape(-1, n) if c.ndim == 1 else c for c in Cin])
        bin_ = np.concatenate([np.atleast_1d(x) for x in bin_])

        C = np.vstack([Jeq, Cin]).T          # quadprog wants C' x >= b, C is (n, m)
        bvec = np.concatenate([eeq, bin_])
        meq = Jeq.shape[0]

        parts["foot"] = float(max(foot_err))
        if (parts["reach"] < tol and parts["foot"] < tol
                and max(parts["site"].values(), default=0.0) < tol):
            break

        # Feet enter the objective at W_FOOT=300 rather than as a hard equality:
        # a hard equality plus the collision inequalities is frequently infeasible
        # inside the trust region, and every such failure used to silently DROP
        # non-penetration.  Soft feet at this weight hold to <1e-3 m in practice
        # (verified), and it lets non-penetration stay hard, which is what we
        # actually cannot compromise on.
        Gs = G + (W_FOOT ** 2) * Jeq.T @ Jeq
        asf = a + (W_FOOT ** 2) * Jeq.T @ eeq
        nbox = 2 * n
        try:
            dq = quadprog.solve_qp(Gs, asf, Cin.T, bin_, 0)[0]
        except ValueError:
            stats["qp_fallback"] += 1
            try:
                dq = quadprog.solve_qp(Gs, asf, Cin[-nbox:].T, bin_[-nbox:], 0)[0]
            except ValueError:
                dq = np.zeros(n)

        full = np.zeros(m.nv)
        full[:n] = dq
        mujoco.mj_integratePos(m, d.qpos, full, 1.0)

        for j in range(1, m.njnt):
            if m.jnt_limited[j] and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                adr_j = m.jnt_qposadr[j]
                d.qpos[adr_j] = np.clip(d.qpos[adr_j],
                                        m.jnt_range[j][0], m.jnt_range[j][1])
        if lock_torso:
            d.qpos[m.jnt_qposadr[13]] = q_seed[m.jnt_qposadr[13]]
        for jn, val in (lock_joints or {}).items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid >= 0:
                d.qpos[m.jnt_qposadr[jid]] = val

    mujoco.mj_forward(m, d)
    pen = 0.0
    for c in range(d.ncon):
        con = d.contact[c]
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        # object<->table is scenery penetrating scenery (42.5 mm at the seed
        # keyframe) and has nothing to do with the robot's pose.
        if _is_env(m, b1) and _is_env(m, b2):
            continue
        pen = min(pen, con.dist)
    parts["penetration"] = float(-pen)
    parts["qp_fallback"] = stats["qp_fallback"]
    parts["site_max"] = float(max(parts.get("site", {}).values(), default=0.0))

    # A bracing site counts as PLACED when its link is actually in contact with
    # the table, not when the nominal body-frame point reaches the table plane:
    # the point sits inside the link, so a resting contact still leaves ~35 mm of
    # link thickness between the point and the surface.
    tbl = bid(m, "table")
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist > IK_MARGIN * 0.5:
            continue
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl:
            touching.add(b2)
        elif b2 == tbl:
            touching.add(b1)
    parts["achieved"] = {}
    for jn in list((soft_joints or {}).keys()) + list((lock_joints or {}).keys()):
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid >= 0:
            parts["achieved"][jn] = float(d.qpos[m.jnt_qposadr[jid]])
    parts["sites_placed"] = {st: (bid(m, SITES[st][0]) in touching) for st in subset}
    parts["all_placed"] = all(parts["sites_placed"].values())
    # The step QP has a two-stage fallback (drop the collision rows, then give up
    # and take dq=0).  It used to be counted and thrown away, so a run where EVERY
    # iteration failed was indistinguishable from one that converged badly -- the
    # residual just sat at its seed value, which reads as "unreachable".
    parts["qp_fallback"] = stats["qp_fallback"]
    parts["ok"] = (parts["reach"] < 0.03 and parts["foot"] < 0.02
                   and parts["penetration"] < 0.01 and parts["all_placed"])
    return parts


# --------------------------------------------------------------------------- #
# static equilibrium QP
# --------------------------------------------------------------------------- #
def equilibrium_qp(m, d, subset, w_tau=1.0, w_lam=1e-4):
    """Solve for contact forces + torques at the current pose.

    Returns dict with feasibility, tau, lambda, and the normalized effort.
    """
    # Pure gravity bias.  NOT mj_inverse: qfrc_inverse folds in qfrc_constraint,
    # i.e. whatever contacts the pose happens to be penetrating (20 kN at the
    # seed keyframe), which swamps the 668 N we are actually balancing.
    d.qvel[:] = 0
    d.qacc[:] = 0
    mujoco.mj_forward(m, d)
    g = d.qfrc_bias[:N_ROBOT_DOF].copy()

    # Contacts come from MuJoCo's narrowphase where one exists, so the force is
    # applied where the contact IS and its cone is built around the REAL normal
    # (see resolved_contacts).  Force variables are in the contact frame:
    # lam = (lam_n, lam_t1, lam_t2), world force = R' lam.
    contacts = resolved_contacts(m, d, subset)
    nc = len(contacts)
    J = []
    for b, p, R in contacts:
        jacp = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacp, None, p, b)
        J.append(R @ jacp[:, :N_ROBOT_DOF])    # rows: n, t1, t2
    A = np.hstack([Ji.T for Ji in J])          # (33, 3*nc)

    tau_max = torque_limits(m)          # see the TORQUE BASIS block above

    # ---- variables: lambda (3*nc) ------------------------------------------
    # base rows are a hard equality; joint rows define tau.
    Ab, gb = A[:6], g[:6]
    Aj, gj = A[6:], g[6:]

    # least-squares with the base equality enforced by heavy weighting, plus
    # bounds; friction handled by an iterative clamp (pyramid) -- see note.
    W_BASE = 1e6
    rows = [W_BASE * Ab, (np.diag(1.0 / tau_max) @ Aj) * np.sqrt(w_tau)]
    rhs = [W_BASE * gb, (gj / tau_max) * np.sqrt(w_tau)]
    rows.append(np.eye(3 * nc) * np.sqrt(w_lam))
    rhs.append(np.zeros(3 * nc))
    M = np.vstack(rows)
    b = np.concatenate(rhs)

    # bounds: normal (z) force >= 0; tangential bounded by the pyramid at a
    # nominal normal load, then re-clamped against the solution below.
    lo = np.full(3 * nc, -np.inf)
    hi = np.full(3 * nc, np.inf)
    # component 0 of each triple is the NORMAL now (contact frame), not world z
    for i in range(nc):
        lo[3 * i] = 0.0                         # unilateral: push only

    sol = lsq_linear(M, b, bounds=(lo, hi), max_iter=200)
    lam = sol.x

    # friction-cone repair: project tangential onto the pyramid, re-solve once
    for _ in range(6):
        viol = False
        for i in range(nc):
            n = lam[3 * i]
            t = lam[3 * i + 1:3 * i + 3]
            lim = MU * max(n, 0.0) / np.sqrt(2)
            if np.abs(t).max() > lim + 1e-9:
                viol = True
                hi[3 * i + 1:3 * i + 3] = lim
                lo[3 * i + 1:3 * i + 3] = -lim
        if not viol:
            break
        sol = lsq_linear(M, b, bounds=(lo, hi), max_iter=200)
        lam = sol.x

    tau = gj - Aj @ lam
    base_res = float(np.linalg.norm(Ab @ lam - gb))
    ratio = np.abs(tau) / tau_max
    return dict(
        subset=list(subset),
        base_residual=base_res,
        feasible=bool(base_res < 1.0 and ratio.max() <= 1.0),
        tau=tau, tau_ratio=ratio,
        max_ratio=float(ratio.max()),
        effort=float(np.sum(ratio ** 2)),
        lam=lam,
        brace_force=float(np.linalg.norm(lam[24:])) if len(subset) else 0.0,
        knee_l=float(abs(tau[3])), knee_r=float(abs(tau[9])),
    )


# --------------------------------------------------------------------------- #
def evaluate(target, verbose=False, lock_torso=False):
    """All 8 contact subsets for one reach target."""
    out = []
    for k in range(4):
        for subset in itertools.combinations(("elbow", "forearm", "palm"), k):
            m, d = load()
            ik_res, pen = solve_ik(m, d, np.asarray(target), subset,
                                   lock_torso=lock_torso)
            r = equilibrium_qp(m, d, subset)
            r["ik_residual"] = ik_res
            r["penetration"] = pen
            r["ik_ok"] = (ik_res < 0.05) and (pen < 0.01)
            r["target"] = list(map(float, target))
            out.append(r)
            if verbose:
                print(f"  {str(subset):40s} ik={ik_res:.3f} pen={pen*1000:5.1f}mm "
                      f"base_res={r['base_residual']:7.2f} "
                      f"max|tau|/lim={r['max_ratio']:.2f} effort={r['effort']:.3f} "
                      f"knee_L={r['knee_l']:6.1f} brace={r['brace_force']:6.1f} "
                      f"{'OK' if r['feasible'] and r['ik_ok'] else '--'}")
    return out


if __name__ == "__main__":
    m, d = load()
    print("table top z =", round(table_top_z(m, d), 3),
          " x range =", np.round(table_x_range(m, d), 3))
    print("left hand at seed:", np.round(point_world(m, d, REACH_BODY, REACH_OFF), 3))
    for s, (b, o) in SITES.items():
        print(f"  site {s:8s} -> {np.round(point_world(m, d, b, o), 3)}")

    tgt = [float(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4 else [1.20, 0.15, 0.85]
    print(f"\nreach target {tgt}")
    evaluate(tgt, verbose=True)
