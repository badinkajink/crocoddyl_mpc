#!/usr/bin/env python3
"""Pinocchio <-> MuJoCo model bridge for the H1-2 braced lean.

crocoddyl plans on a Pinocchio model; every number this study has produced so far
came out of the MJCF.  Those are two different models of the same robot, built by
two different importers from the same CL_Assets source, and the whole crocoddyl
plan is worthless if they disagree.  So the bridge is a module with a parity test
attached, not three lines of index arithmetic inline in the planner.

Three real differences have to be reconciled:

  1. GRIPPER JOINTS.  `h1_2_magpie.urdf` models each magpie gripper as a 6-joint
     linkage (lg_/rg_ hinge_1..3 per side, two of them RUBX i.e. unbounded).  The
     MJCF fuses each gripper into ONE rigid body with box collision proxies -- the
     brace study's `palm` site lives on that body.  So the Pinocchio model is
     REDUCED, freezing all 12 hinges at 0, which makes the gripper rigid and
     leaves nq = 7 + 27, nv = 6 + 27, matching the MJCF robot exactly.

  2. THE TABLE.  MJCF joint 28 `object_anchor_joint` is a FREE joint carrying the
     table, so MuJoCo's nq=41 / nv=39 are robot + table.  The robot occupies
     qpos[0:34] / dof[0:33].  crocoddyl gets a table as a fixed contact plane, not
     as a floating body, so the bridge slices the robot out and records the table
     top plane separately.

  3. QUATERNION ORDER.  MuJoCo free-joint qpos is [xyz, w,x,y,z]; Pinocchio
     free-flyer q is [xyz, x,y,z,w].  Silent and catastrophic if missed -- it
     reads as a plausible-looking rotated base rather than an error.

Joint ORDER for the 27 actuated joints is already identical (left leg 6, right
leg 6, torso, left arm 7, right arm 7), which is verified rather than assumed.

Run `python3 croco_bridge.py` for the parity report.
"""

import ctypes
import os
import sys

# RTLD_GLOBAL, set before ANY of the native extensions load -- see
# `import_crocoddyl` below for why.  It is left set for the rest of the process
# rather than restored: the flag has to cover pinocchio as well as crocoddyl
# (flagging crocoddyl alone, with pinocchio already resident RTLD_LOCAL, still
# crashes -- the typeinfo the casts compare is shared between them), and the
# configuration verified to run clean is the one where everything is global.
# This module is therefore the FIRST import in every croco_* script.
sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np
import pinocchio as pin
import mujoco
import crocoddyl as _crocoddyl

import contact_select as cs


def import_crocoddyl():
    """Import crocoddyl with RTLD_GLOBAL.  A plain `import crocoddyl` SEGFAULTS.

    The crocoddyl 3.2.1 cmeel wheel ships the same symbols twice: once in
    `libcrocoddyl.so` and again inlined into `libcrocoddyl_pywrap_float64...so`
    (34 and 93 definitions of ResidualDataContactWrenchCone's constructor
    respectively).  Python dlopen's extension modules RTLD_LOCAL by default, so
    the two copies never get reconciled -- and `typeinfo for ContactData6D` exists
    in only one of them.

    That matters because the contact-cone residuals identify their contact by
    `dynamic_cast` on the contact data.  Across two unreconciled copies the cast
    silently returns the wrong thing, and the residual's data constructor then
    memmoves against a bogus size: SIGSEGV inside
    ResidualDataContactWrenchConeTpl, raised from ShootingProblem's allocateData,
    with nothing in the Python traceback pointing at contacts or cones.  When the
    allocator happens to notice first it surfaces instead as a bare MemoryError or
    `RuntimeError: basic_string::_M_create`, and whether you get a crash, a
    MemoryError, or a clean run varies between identical invocations -- which is
    what makes it read like memory pressure and not like a linkage bug.

    Opening with RTLD_GLOBAL unifies the symbols and the casts resolve.  The
    import itself happens once at the top of this module, where the flags can be
    set before pinocchio loads; this just hands back the correctly-loaded module.
    """
    return _crocoddyl

# The URDF crocoddyl plans on.  Same provenance rule as contact_select._LEAN_DIR:
# it must be the SAME CL_Assets tree the MJCF was staged from, or the bridge is
# reconciling two different robots and its parity test is meaningless.
#
# Resolution order, first hit wins:
#   H12_URDF        explicit path
#   CL_ASSETS_DIR   the same variable stage_assets.sh takes -- so ONE export
#                   points both halves of the study at one CL_Assets checkout
#   build/_deps     the cmake FetchContent location, for a real MJPC build
#
# Was H12_URDF-or-FetchContent only, which meant a machine that staged with
# CL_ASSETS_DIR (because build/ is root-owned, as it is under docker) got a
# staged MJCF and no URDF, and failed here with urdfdom's "does not contain a
# valid URDF model" -- a parse error for a file that is simply absent.
def _find_urdf():
    rel = "ros_assets/h1_2_magpie.urdf"
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tried = []
    for cand in (os.environ.get("H12_URDF"),
                 os.path.join(os.environ["CL_ASSETS_DIR"], rel)
                 if os.environ.get("CL_ASSETS_DIR") else None,
                 os.path.join(root, "build/_deps/cl_assets-src", rel)):
        if not cand:
            continue
        tried.append(cand)
        if os.path.exists(cand):
            return cand
    raise SystemExit(
        "croco_bridge: no H1-2 magpie URDF found. Tried:\n  "
        + "\n  ".join(tried)
        + "\nSet CL_ASSETS_DIR to the CL_Assets checkout you staged the MJCF "
          "from (the same one stage_assets.sh took), or H12_URDF to the file.")


URDF = _find_urdf()
URDF_DIR = os.path.dirname(URDF)

# The 27 actuated joints, in MJCF order.  Pinocchio is asked to match this order
# rather than trusted to have it.
ACTUATED = [
    "left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_yaw_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# Gripper linkage joints the MJCF does not have -- frozen to make the gripper the
# rigid body the brace study braces on.
GRIPPER_JOINTS = [f"{s}_{h}_hinge_{i}" for s in ("lg", "rg")
                  for h in ("left", "right") for i in (1, 2, 3)]

# MJCF body -> URDF link, where the two importers named the same frame differently.
# The MJCF collapses each gripper into one body `*_magpie_gripper` mounted on the
# wrist with body_pos = 0 and body_quat = identity, i.e. COINCIDENT with
# `*_wrist_yaw_link`; the URDF keeps the linkage and has no such link.  So the
# palm site's body-frame offset is the same offset on the wrist frame.  That
# coincidence is asserted in `parity` rather than trusted.
BODY_ALIAS = {
    "left_magpie_gripper": "left_wrist_yaw_link",
    "right_magpie_gripper": "right_wrist_yaw_link",
}

NQ_ROBOT = 34   # MJCF qpos slice that is the robot: 7 free + 27 joints
NV_ROBOT = 33   # MJCF dof  slice that is the robot: 6 free + 27 joints

# FOOT SOLE frames.  The study pins `*_ankle_roll_link`, whose origin sits 55 mm
# ABOVE the sole and 33 mm behind the centre of the contact polygon.  That offset
# is irrelevant to a pinned-foot IK but not to a wrench cone, which is written
# about the contact frame and would otherwise place the centre of pressure in the
# wrong spot by 33 mm on a foot only 200 mm long.  Offsets and half-extents are
# contact_select.foot_corners', kept in one place here.
FOOT_HALF = (0.10, 0.04)                 # half-length, half-width [m]
SOLES = {
    "sole_left":  ("left_ankle_roll_link",  np.array([0.033, 0.0, -0.055])),
    "sole_right": ("right_ankle_roll_link", np.array([0.033, 0.0, -0.055])),
}


_MJ_CACHE = None


def mj_model():
    """Cached (model, data) for READ-ONLY queries.

    cs.load() compiles the MJCF from scratch every call -- meshes and all, ~400 MB
    resident -- because the study uses it to get a clean state per IK solve.  The
    bridge only ever reads geometry and inertia off it, and was loading it three
    times per OCP.  Anything that mutates qpos must still call cs.load() itself.
    """
    global _MJ_CACHE
    if _MJ_CACHE is None:
        _MJ_CACHE = cs.load()
    return _MJ_CACHE


def mj_joint_passive(m=None):
    """(damping, armature, frictionloss) over the 27 actuated joints, from the MJCF.

    THE SECOND HALF OF THE BRIDGE, and it was missing until 2026-08-06.  The
    inertia reconciliation below made the two models agree about MASS; they still
    disagreed about everything the MJCF puts on a joint that a URDF has no place
    for.  This model carries, on every actuated joint:

        damping       10 N m s/rad
        armature      0.1 kg m^2
        frictionloss  0.2 N m

    The damping is the one that matters and it is not a small correction: at the
    2.6 rad/s the plan reaches it is 26 N m of joint torque that crocoddyl does
    not know exists -- more than the entire clamp budget of a wrist and 54% of an
    ankle's.  A plan solved without it is solved for a different robot, and the
    replay symptom is exactly what was measured: a standing tracking error that
    grows fastest at the joints moving fastest, a base that pitches, and a fall
    at ~1.4 s that reads like "open loop cannot work" rather than like a missing
    term.  The armature is 0.1 against arm-link inertias of order 0.01-0.05, so
    it is not a rounding correction to the mass matrix either.

    Returned per ACTUATED joint in ACTUATED order, and asserted uniform per joint
    (one dof each) rather than assumed.
    """
    m = m or mj_model()[0]
    b, a, f = [], [], []
    for name in ACTUATED:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"{name} is not a hinge; passive map needs one dof")
        dof = m.jnt_dofadr[j]
        b.append(float(m.dof_damping[dof]))
        a.append(float(m.dof_armature[dof]))
        f.append(float(m.dof_frictionloss[dof]))
    return np.array(b), np.array(a), np.array(f)


def build_pin(reconcile_inertia=True, reconcile_passive=True):
    """Reduced Pinocchio model: free-flyer + the same 27 joints as the MJCF.

    With `reconcile_inertia` (the default) every link inertia is REPLACED by the
    composite inertia MuJoCo actually simulates.  The two importers do not agree
    about mass -- 0.782 kg of it, measured:

      * `back_equipment`, 0.670 kg bolted to torso_link in the MJCF, has no URDF
        link at all.  It is the single worst place to lose mass in this study: it
        rides the torso, which pitches ~32 deg in the brace, so it carries the
        longest moment arm of any payload on the robot.
      * each magpie gripper is 0.506 kg in the MJCF and 0.450 kg of linkage in the
        URDF.

    Left alone that is a robot 1.1% too light, leaning on a moment arm that is
    wrong by more than that, and every contact force crocoddyl predicts would be
    quietly low against the numbers the static QP certified.  Kinematics need no
    such treatment: link frames already agree to < 1e-6 m (see `parity`).
    """
    full = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
    lock = [full.getJointId(j) for j in GRIPPER_JOINTS if full.existJointName(j)]
    q0 = pin.neutral(full)
    model = pin.buildReducedModel(full, lock, q0)
    if model.nq != NQ_ROBOT or model.nv != NV_ROBOT:
        raise RuntimeError(f"reduced model is nq={model.nq} nv={model.nv}, "
                           f"expected {NQ_ROBOT}/{NV_ROBOT}")
    order = [n for n in model.names[2:]]     # skip universe + root_joint
    if order != ACTUATED:
        raise RuntimeError("joint order mismatch vs MJCF:\n"
                           f"  pin: {order}\n  mjc: {ACTUATED}")
    if reconcile_inertia:
        _apply_mj_inertia(model)
    if reconcile_passive:
        # ARMATURE goes into the model, where pinocchio's crba/aba add it to the
        # mass-matrix diagonal, so crocoddyl's contact dynamics pick it up with no
        # further plumbing.  DAMPING and FRICTIONLOSS cannot: pinocchio stores
        # model.damping/model.friction but no forward-dynamics algorithm applies
        # them, so they are carried by croco_plan.ActuationModelJointPassive
        # instead -- see there.
        b, a, f = mj_joint_passive()
        model.armature[6:] = a
        model.damping[6:] = b
        model.friction[6:] = f
    return model


def _mj_driving_joint(m, b):
    """MuJoCo joint id that moves body `b` (walking up through welded bodies)."""
    while b > 0:
        if m.body_jntnum[b] > 0:
            return int(m.body_jntadr[b])
        b = m.body_parentid[b]
    return -1


def _mj_body_inertia(m, b):
    """Spatial inertia of MuJoCo body `b`, expressed in that body's OWN frame."""
    R = _quat_mat(m.body_iquat[b])                    # body frame -> inertia frame
    I_c = R @ np.diag(m.body_inertia[b]) @ R.T        # about the CoM, body axes
    return pin.Inertia(float(m.body_mass[b]), np.asarray(m.body_ipos[b], float), I_c)


def _quat_mat(q):
    return pin.Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])).matrix()


def _apply_mj_inertia(model):
    """Overwrite model.inertias with MuJoCo's composite inertia per joint."""
    m, d = mj_model()
    mujoco.mj_forward(m, d)
    root = 1                                   # the free-joint body (pelvis)

    # One forward-kinematics pass for all 28 bodies, not one per body.
    data = model.createData()
    pin.forwardKinematics(model, data, mj_to_pin(d.qpos))

    comp = {}                                  # pin joint id -> pin.Inertia
    for b in range(1, m.nbody):
        if not _under(m, b, root):
            continue                           # table / object / target: not robot
        j = _mj_driving_joint(m, b)
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        jid = 1 if jname == "floating_base_joint" else model.getJointId(jname)

        # MuJoCo body frame -> the frame the driving joint's Pinocchio inertia is
        # expressed in.  Both are read off the CURRENT configuration, so this is a
        # pure frame change and does not depend on the pose being special.
        oMj = data.oMi[jid]
        oMb = pin.SE3(d.xmat[b].reshape(3, 3).copy(), d.xpos[b].copy())
        jMb = oMj.actInv(oMb)
        I = jMb.act(_mj_body_inertia(m, b))
        comp[jid] = I if jid not in comp else comp[jid] + I

    for jid, I in comp.items():
        model.inertias[jid] = I


def mj_to_pin(qpos):
    """MuJoCo qpos (41,) or (34,) -> Pinocchio q (34,).  Reorders the quaternion."""
    q = np.asarray(qpos, dtype=float)[:NQ_ROBOT]
    out = np.empty(NQ_ROBOT)
    out[0:3] = q[0:3]
    out[3:7] = q[[4, 5, 6, 3]]               # w,x,y,z -> x,y,z,w
    out[7:] = q[7:NQ_ROBOT]
    return out


def mj_to_pin_v(qvel, R_base):
    """MuJoCo qvel (39,) or (33,) -> Pinocchio v (33,).  Rotates the base linear.

    Same convention trap as the gravity rows in `parity`, and it bites harder
    here because this one feeds a FEEDBACK law.  A MuJoCo free joint reports its
    linear velocity in WORLD and its angular velocity in the BODY frame;
    Pinocchio's free-flyer reports both in the body frame.  So only the linear
    half is rotated, with R_base = d.xmat[1] (world <- pelvis).

    Getting it wrong does not look like an error: it looks like a controller that
    is stable while the base is level and drifts as the robot pitches, which is
    exactly the regime this maneuver lives in (torso ~32 deg).
    """
    v = np.asarray(qvel, dtype=float)[:NV_ROBOT].copy()
    v[0:3] = np.asarray(R_base, float).T @ v[0:3]
    return v


def pin_to_mj(q, qpos_full=None):
    """Pinocchio q (34,) -> MuJoCo qpos.  Keeps the table pose from `qpos_full`."""
    q = np.asarray(q, dtype=float)
    out = np.zeros(41) if qpos_full is None else np.asarray(qpos_full, float).copy()
    out[0:3] = q[0:3]
    out[3:7] = q[[6, 3, 4, 5]]               # x,y,z,w -> w,x,y,z
    out[7:NQ_ROBOT] = q[7:]
    return out


def mj_site_frames(model):
    """Add the brace/reach/foot sites of contact_select.SITES as Pinocchio frames.

    The study's sites are (body, body-frame offset) pairs.  MuJoCo body frames and
    URDF link frames coincide here (verified by `parity`), so the offset carries
    over unchanged as a fixed frame placement.
    """
    spec = {n: (b, o) for n, (b, o) in cs.SITES.items()}
    spec["reach"] = (cs.REACH_BODY, cs.REACH_OFF)
    spec.update(SOLES)
    added = {}
    for name, (body, off) in spec.items():
        fid = model.getFrameId(BODY_ALIAS.get(body, body))
        parent = model.frames[fid].parentJoint
        pl = model.frames[fid].placement * pin.SE3(np.eye(3), np.asarray(off, float))
        added[name] = model.addFrame(
            pin.Frame(f"site_{name}", parent, fid, pl, pin.FrameType.OP_FRAME))
    return added


# --------------------------------------------------------------------------- #
def parity(qpos=None, verbose=True):
    """Compare Pinocchio and MuJoCo at one pose.  Returns a dict of max errors."""
    m, d = cs.load()
    if qpos is not None:
        d.qpos[:len(qpos)] = qpos
    mujoco.mj_forward(m, d)

    # BODY_ALIAS is only valid while the aliased MJCF bodies really are coincident
    # with their URDF stand-ins.  Cheap to check, catastrophic to assume.
    for mj_body, urdf_link in BODY_ALIAS.items():
        b = cs.bid(m, mj_body)
        par = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.body_parentid[b])
        off = np.linalg.norm(m.body_pos[b])
        rot = np.linalg.norm(m.body_quat[b] - np.array([1.0, 0, 0, 0]))
        if par != urdf_link or off > 1e-9 or rot > 1e-9:
            raise RuntimeError(
                f"BODY_ALIAS[{mj_body}] -> {urdf_link} is stale: parent={par}, "
                f"|pos|={off:.2e}, |dquat|={rot:.2e}")

    model = build_pin()
    mj_site_frames(model)
    data = model.createData()
    q = mj_to_pin(d.qpos)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)

    # --- mass.  MuJoCo's nbody spans the whole SCENE: robot, table, `object` and
    #     `target`.  "everything that is not the table" silently keeps the last two
    #     and inflates the robot by 0.41 kg, so the robot is defined as the subtree
    #     of body 1 -- the body carrying the free joint.
    robot_mass = float(sum(m.body_mass[b] for b in range(1, m.nbody)
                           if _under(m, b, 1)))
    pin_mass = pin.computeTotalMass(model)

    # --- body/link frame placements
    pos_err = {}
    for b in range(1, m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if name is None or not _under(m, b, 1) or not model.existFrame(name):
            continue
        fid = model.getFrameId(name)
        pos_err[name] = float(np.linalg.norm(data.oMf[fid].translation - d.xpos[b]))

    # --- the study's contact sites, which is what crocoddyl will constrain
    site_err = {}
    for name in list(cs.SITES) + ["reach"]:
        if name == "reach":
            body, off = cs.REACH_BODY, cs.REACH_OFF
        else:
            body, off = cs.SITES[name]
        mj_p = cs.point_world(m, d, body, off)
        pin_p = data.oMf[model.getFrameId(f"site_{name}")].translation
        site_err[name] = float(np.linalg.norm(pin_p - mj_p))

    # --- centre of mass (robot only in Pinocchio; MuJoCo subtree_com[1] is the
    #     robot root's subtree, i.e. robot without the table)
    pin_com = pin.centerOfMass(model, data, q)
    mj_com = d.subtree_com[1].copy()
    com_err = float(np.linalg.norm(pin_com - mj_com))

    # --- gravity generalized force.  This is the term the static QP balances and
    #     the term crocoddyl's dynamics differentiates, so it is the one that has
    #     to agree.  MuJoCo qfrc_bias at rest is +g(q); Pinocchio computeGeneralizedGravity
    #     returns the same sign convention.
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    g_mj = d.qfrc_bias[:NV_ROBOT].copy()   # qvel = 0 => Coriolis drops out
    g_pin = pin.computeGeneralizedGravity(model, data, q)
    # The 6 base rows are expressed in a MIXED frame by MuJoCo and a LOCAL one by
    # Pinocchio, and the mix is not symmetric: a MuJoCo free joint reports its
    # linear dofs in WORLD and its angular dofs in the BODY frame, while
    # Pinocchio's free-flyer reports both in the body frame.  So only the linear
    # half gets rotated.  Verified numerically, not assumed: at the brace pose the
    # world moment r_com x (0,0,-Mg) reproduces MuJoCo's angular rows only under
    # this convention (rotating both halves leaves ~2 Nm of residual).
    R = d.xmat[1].reshape(3, 3)
    g_pin_world = g_pin.copy()
    g_pin_world[0:3] = R @ g_pin[0:3]
    g_err_joint = float(np.max(np.abs(g_pin_world[6:] - g_mj[6:])))
    g_err_base = float(np.max(np.abs(g_pin_world[:6] - g_mj[:6])))

    # --- mass matrix.  The inertia reconciliation above is about MASS; this is
    #     the row that catches ARMATURE, which is 0.1 kg m^2 on every actuated
    #     joint and shows up nowhere else.  Compared over the 27 actuated rows
    #     only: the 6 base rows are in different frames (see the gravity note).
    mujoco.mj_forward(m, d)
    M_mj = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M_mj, d.qM)
    M_pin = pin.crba(model, data, q)
    M_pin = np.triu(M_pin) + np.triu(M_pin, 1).T
    m_err = float(np.max(np.abs(M_pin[6:, 6:] - M_mj[6:NV_ROBOT, 6:NV_ROBOT])))
    b_mj, a_mj, f_mj = mj_joint_passive(m)
    passive_err = float(max(np.max(np.abs(model.armature[6:] - a_mj)),
                            np.max(np.abs(model.damping[6:] - b_mj)),
                            np.max(np.abs(model.friction[6:] - f_mj))))

    res = dict(mass_matrix_err=m_err, passive_err=passive_err,
               mass_mj=robot_mass, mass_pin=float(pin_mass),
               mass_err=float(abs(robot_mass - pin_mass)),
               body_pos_err_max=max(pos_err.values()) if pos_err else 0.0,
               body_pos_err_argmax=max(pos_err, key=pos_err.get) if pos_err else "",
               n_bodies_compared=len(pos_err),
               site_err=site_err, com_err=com_err,
               g_err_joint=g_err_joint, g_err_base=g_err_base)

    if verbose:
        print(f"mass      MuJoCo {robot_mass:9.5f} kg   Pinocchio {pin_mass:9.5f} kg"
              f"   |d| = {res['mass_err']:.2e}")
        print(f"CoM       |d| = {com_err:.3e} m")
        print(f"bodies    {len(pos_err)} compared, worst {res['body_pos_err_max']:.3e} m"
              f"  ({res['body_pos_err_argmax']})")
        for k, v in site_err.items():
            print(f"  site {k:8s} |d| = {v:.3e} m")
        print(f"gravity   joints max|d| = {g_err_joint:.3e} Nm"
              f"   base max|d| = {g_err_base:.3e}")
        print(f"mass mat  27x27 actuated block max|d| = {m_err:.3e} kg m^2")
        print(f"passive   damping/armature/friction max|d| = {passive_err:.3e}")
    return res


def _under(m, b, root):
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


if __name__ == "__main__":
    import sys
    qpos = None
    if len(sys.argv) > 1:
        qpos = np.loadtxt(sys.argv[1])
        print(f"pose: {sys.argv[1]}  ({len(qpos)} values)\n")
    else:
        print("pose: model default (seed keyframe)\n")
    parity(qpos)
