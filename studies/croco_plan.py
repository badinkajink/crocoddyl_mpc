#!/usr/bin/env python3
"""crocoddyl MVP: plan stand -> braced contact mode for the H1-2 lean.

WHAT THIS IS FOR.  Today the braced lean is authored by hand: eight keyframes, a
14 s weight ramp, and a 10-sample Cross-Entropy MPC with a 1 s horizon tracking
the setpoint they drag past it.  The keyframes ARE the plan; CEM is a
short-horizon stabilizer.  The offline study separately certifies a pose q* and a
contact subset by enumeration + a static-equilibrium LP/QP, and the hand-authored
ramp lands ~1.5 rad away from it.  This closes that gap the other way: take the
enumeration's (q*, contact schedule) and ask a DDP to produce the trajectory
that gets there, with the contact schedule PRESCRIBED -- which is the input
crocoddyl wants and the thing the enumeration already produces.

STRUCTURE.  Phases over a prescribed schedule:

    A  approach   feet only.        Lower the CoM and swing the bracing arm out
                                    over the table, everything kept OUT of the
                                    table box, CoM kept over the feet.
    I  impact     impulse at the    Zero-velocity touchdown of the bracing sites.
       brace sites                  (off by default -- see `build`)
    B  braced     feet + brace.     Settle onto q*, reach the target, come to
                                    rest inside the friction cones, with the
                                    brace sites HELD on their landing spots.
    R  return     feet only.        Optional: lift the brace off the table and
                                    come back to the start pose, CoM over feet
                                    again.  `n_return > 0`.

WHAT S12 GOT WRONG, AND WHAT CHANGED (2026-08-06).  The S12 plan solved and did
not survive MuJoCo.  A node-by-node geometry readout (`croco_diag.py`) says why,
and it is not what the S12 write-up guessed:

  1. THE BRACED PHASE WAS LEANING ON THIN AIR.  Over all 41 braced nodes the
     elbow site sat 52.6 mm above the tabletop and the forearm site 19.0 mm above
     it, against 42.9 / 33.6 mm at the certified pose -- so the elbow was ~10 mm
     too high and the forearm ~15 mm too low, i.e. one link hovering and the
     other buried.  The contact models were happily supplying force at both.
     Nothing in the OCP made the landing accurate: the `land_` cost ramped to a
     weight of 50 against a reach cost of 100 and a terminal state cost, and
     ContactModel3D with Baumgarte Kp = 0 constrains velocity, not position, so
     once the phase starts the site stays wherever it arrived.  FIX: the landing
     cost is z-weighted and much heavier, and the braced phase carries an
     explicit `hold_` cost at the landing spot.
  2. THE HALF-SPACE BARRIER COULD NOT EXPRESS THE TABLE.  See croco_geom: the
     tabletop is a box, the certified pose hangs geometry over its edge, and
     `z >= table_z` cannot say "below the top but outside the footprint".
     FIX: a box-SDF keep-out, thresholded so the certified pose is feasible.
  3. THE GRIPPER WAS NOT A CONTACT BUT WAS TOUCHING.  At q*, MuJoCo's narrowphase
     reports the bracing gripper on the table (3 contacts, 0.9-3.7 mm deep)
     alongside the elbow and forearm links.  Planning `elbow+forearm` therefore
     left a body that is IN CONTACT in the certified pose completely
     unconstrained, which is how it ended up 68 mm through the slab.  FIX: run
     the mode that matches what the pose actually does -- `elbow+forearm+palm`.
  4. NOTHING KEPT THE CoM OVER THE FEET.  The approach leans out over the table
     on feet-only support for over a second, and rigid prescribed contacts let
     the optimiser borrow support it does not have yet.  FIX: a quadratic barrier
     holding the CoM inside the foot polygon (shrunk by a margin) whenever the
     brace is not down.

WHAT IS AND IS NOT MODELLED.  Honest list, because these are the costs that decide
whether the result transfers:

  * Contacts are RIGID and prescribed.  This is not MuJoCo's soft contact model,
    so predicted forces will not match MuJoCo's exactly -- `croco_replay.py`
    measures the difference rather than assuming it away.
  * The table is a contact reference and a keep-out box built from
    `table_top_collision`.  Its LEGS are not modelled; nothing in this maneuver
    goes near them (verified: min clearance over the plan is reported by
    croco_diag).
  * The feet are 6D-pinned at their placement in the START pose, not at q*'s.
    The study's IK pinned feet at the brace SEED keyframe, whose stance is 19 mm
    narrower than `stand`; a rigid contact cannot be in two places, so the start
    pose wins and q* enters as a posture target rather than a hard terminal
    constraint.  How far the result actually lands from q* is measured, not
    assumed -- see the report at the end of a run.
  * Torque limits are the study's `clamp` basis, enforced as hard box bounds by
    SolverBoxFDDP.

`legacy=True` restores the S12 formulation exactly (half-space barrier, no hold,
no CoM barrier, no return phase) and reproduces its numbers to 7 figures.
"""

import argparse
import json
import os
import sys
import subprocess
import time

import numpy as np
import pinocchio as pin
import mujoco

import contact_select as cs
import croco_bridge as cb
import croco_geom as cg

# NOT `import crocoddyl` -- see croco_bridge.import_crocoddyl for why a plain
# import segfaults inside ShootingProblem.
crocoddyl = cb.import_crocoddyl()

# The feet enter as 6D contacts at the SOLE frames (see croco_bridge.SOLES), which
# is where a wrench cone has to be written for its centre-of-pressure rows to mean
# anything.
FEET = ["sole_left", "sole_right"]

# Baumgarte contact stabilisation gains (Kp, Kd) and the KKT inverse-damping used
# by the contact forward dynamics.  Module-level so they can be swept.
CONTACT_GAINS = np.array([0.0, 50.0])

# INV_DAMPING is not a nuisance parameter -- it is what makes the braced phase
# solvable at all.  Two 6D feet plus the bracing sites is 18 constraint rows on a
# 33-DOF robot, and the contacts are REDUNDANT: many force splits between feet and
# brace produce the same motion, so the KKT system is singular and the force the
# friction-cone cost differentiates is arbitrary.  Measured: with damping 0 or
# 1e-6 the braced problem fails from every cone weight and min-normal-force tried
# (stop stalls at 0.1-2.5 after 80 iterations); at 1e-4 all six of those
# configurations converge to stop < 4e-5.  This is the same regularizer the
# study's static QP already needs for the same reason -- its w_lam min-norm term
# on lambda, which its docstring says "is what should make triple contact emerge".
INV_DAMPING = 1e-4


# Smoothing width for Coulomb friction [rad/s].  MuJoCo's frictionloss is a hard
# 0.2 N.m constraint; a DDP needs a differentiable stand-in, and tanh(v/eps) with
# eps = 0.05 is within 1% of the constraint above 0.15 rad/s while staying
# Lipschitz through zero.  It is 0.2 N.m -- the reason it is here at all is that
# leaving it out means the plan's torques are systematically low on every joint
# that is moving, and the systematic part is what the servo has to make up.
FRICTION_EPS = 0.05


class ActuationModelJointPassive:
    """Placeholder; the real class is built in `_make_actuation` (needs crocoddyl)."""


def _cpp_actuation(state, damping, friction, eps):
    """The C++ transcription of `_make_actuation`, or None if it is not built.

    Same model, same numbers (croco_ext/test_passive.py); it is here because the
    Python one is the last interpreted object in the hot path -- 4.7 us of a
    323 us node in calcDiff, and 2.0 us in the `calc` that FDDP's line search
    runs once per trial step.  CROCO_PASSIVE=python forces the interpreter path
    so the difference stays measurable.
    """
    if os.environ.get("CROCO_PASSIVE", "cpp") == "python":
        return None
    try:
        ext = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "croco_ext")
        if ext not in sys.path:
            sys.path.insert(0, ext)
        import croco_passive
    except ImportError:
        return None
    return croco_passive.ActuationModelJointPassive(
        state, np.asarray(damping, float), np.asarray(friction, float),
        float(eps))


def _make_actuation(state, damping, friction, eps=FRICTION_EPS):
    """Floating-base actuation that also carries the plant's PASSIVE joint torques.

    crocoddyl's stock ActuationModelFloatingBase maps u straight onto the
    actuated rows: tau = [0; u].  That is the right model for a robot whose
    joints are frictionless and undamped, and this one is not -- the MJCF puts
    10 N m s/rad of damping and 0.2 N m of friction loss on all 27 joints, which
    at the plan's velocities is up to 26 N m per joint that MuJoCo subtracts and
    crocoddyl never budgeted for (see croco_bridge.mj_joint_passive).

    So the actuation becomes

        tau = [0; u - b v_j - f tanh(v_j / eps)]

    which is exactly what the plant does, and the plan's torques become the
    torques the actuator has to produce rather than the net joint torques.  The
    armature is handled the other way, inside the pinocchio model, because
    pinocchio's mass matrix already knows how to carry it.
    """
    cpp = _cpp_actuation(state, damping, friction, eps)
    if cpp is not None:
        return cpp

    class _Passive(crocoddyl.ActuationModelAbstract):
        def __init__(self):
            crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv - 6)
            self.b = np.asarray(damping, float)
            self.f = np.asarray(friction, float)
            self.eps = float(eps)
            nv, nu = state.nv, state.nv - 6
            self._S = np.zeros((nv, nu))
            self._S[6:, :] = np.eye(nu)

        def calc(self, data, x, u):
            vj = x[state.nq + 6:]
            tau = np.zeros(state.nv)
            tau[6:] = u - self.b * vj - self.f * np.tanh(vj / self.eps)
            data.tau = tau

        def calcDiff(self, data, x, u):
            vj = x[state.nq + 6:]
            nv = state.nv
            dtau_dx = np.zeros((nv, 2 * nv))
            sech2 = 1.0 - np.tanh(vj / self.eps) ** 2
            dtau_dx[6:, nv + 6:] = -np.diag(self.b + self.f * sech2 / self.eps)
            data.dtau_dx = dtau_dx
            data.dtau_du = self._S

        def commands(self, data, x, tau):
            data.u = tau[6:]

    return _Passive()


# --------------------------------------------------------------------------- #
def _joint_bounds(model_mj):
    """Joint position bounds (27 actuated joints) straight from the MJCF."""
    lb, ub = [], []
    for name in cb.ACTUATED:
        j = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model_mj.jnt_limited[j]:
            lb.append(model_mj.jnt_range[j][0])
            ub.append(model_mj.jnt_range[j][1])
        else:
            lb.append(-np.inf)
            ub.append(np.inf)
    return np.array(lb), np.array(ub)


class LeanOCP:
    def __init__(self, subset, q_star_mj, q0_mj, mu=0.6, table_z=None,
                 reach_target=None, clearance=0.02, cones=True, legacy=False,
                 keepout=True, com_margin=0.03, w_land=5e3, w_hold=1e3,
                 w_com=1e3, land_z_weight=3.0, w_cone=1e1, w_reach=1e2,
                 cop_shrink=1.0, min_nforce=1.0, w_com_track=0.0,
                 w_ctrl=1e-3, n_hold=0, w_hold_state=1e2, w_com_damp=0.0,
                 w_jlim=1e3, w_vel=0.0, reach_rot=None, w_reach_rot=0.0,
                 return_q_mj=None, drop=()):
        self.rmodel = cb.build_pin(reconcile_passive=not legacy)
        self.sites = cb.mj_site_frames(self.rmodel)
        self.subset = list(subset)
        self.legacy = legacy
        m_mj, d_mj = cb.mj_model()

        # Keep-out points get their Pinocchio frames HERE, before StateMultibody
        # is constructed: crocoddyl takes the model by reference but every
        # pinocchio Data downstream is sized from it, and adding frames after the
        # state exists is the kind of thing that works until it silently does not.
        self.keepout, self.table_c, self.table_half = [], None, None
        if keepout and not legacy:
            self.keepout, self.table_c, self.table_half = cg.keepout_spec(
                m_mj, d_mj, q_star_mj, self.subset)
            seen = {}
            for p in self.keepout:
                key = (p["body"], tuple(np.round(p["offset"], 9)))
                if key not in seen:
                    fid = self.rmodel.getFrameId(
                        cb.BODY_ALIAS.get(p["body"], p["body"]))
                    pl = self.rmodel.frames[fid].placement * pin.SE3(
                        np.eye(3), p["offset"])
                    seen[key] = self.rmodel.addFrame(pin.Frame(
                        f"ko_{len(seen)}", self.rmodel.frames[fid].parentJoint,
                        fid, pl, pin.FrameType.OP_FRAME))
                p["fid"] = seen[key]

        self.rdata = self.rmodel.createData()
        self.state = crocoddyl.StateMultibody(self.rmodel)
        b, _, f = cb.mj_joint_passive(m_mj)
        self.damping, self.friction = b, f
        self.actuation = (crocoddyl.ActuationModelFloatingBase(self.state)
                          if legacy else _make_actuation(self.state, b, f))
        self.nu = self.actuation.nu

        self.q0 = cb.mj_to_pin(q0_mj)
        self.q_star = cb.mj_to_pin(q_star_mj)
        self.x0 = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])
        self.x_star = np.concatenate([self.q_star, np.zeros(self.rmodel.nv)])
        # WHERE THE RETURN PHASE GOES, which is not always where it started.
        # `x0` was doing double duty: the initial state AND the pose the R
        # phase regulates back to. That is right for the round trip (stand ->
        # brace -> stand) and wrong for the only maneuver that makes chaining
        # possible -- a recovery that BEGINS braced, whose x0 is the braced
        # pose and whose destination is the stand. Left as x0 such a plan
        # "returns" to the brace it is supposed to be leaving.
        self.x_return = (self.x0 if return_q_mj is None else
                         np.concatenate([cb.mj_to_pin(np.asarray(return_q_mj)),
                                         np.zeros(self.rmodel.nv)]))
        self.mu = mu
        self.cones = cones
        self.table_z = table_z
        self.clearance = clearance
        self.reach_target = reach_target

        # Contact references.  Feet from the START pose (see the module docstring);
        # bracing sites from q*, i.e. the places on the table the static QP
        # certified and solved forces for.
        pin.forwardKinematics(self.rmodel, self.rdata, self.q0)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self.foot_ref = {f: self.rdata.oMf[self.sites[f]].copy() for f in FEET}
        pin.forwardKinematics(self.rmodel, self.rdata, self.q_star)
        pin.updateFramePlacements(self.rmodel, self.rdata)
        self.site_ref = {s: self.rdata.oMf[self.sites[s]].translation.copy()
                         for s in self.subset}
        self.reach_ref = self.rdata.oMf[self.sites["reach"]].translation.copy()
        # The gripper orientation the IK pose ALREADY reaches. `reach_rot=
        # "auto"` references the cost to this, which makes the term cost
        # nothing at q* -- so it can be carried at a token weight purely so it
        # EXISTS in the built models. That existence is the whole point: a
        # residual's `.reference` is live-settable, so a term that is present
        # can be rotated from the panel, and a term that is absent cannot be
        # added to a built model without reallocating its data.
        self.reach_rot_auto = self.rdata.oMf[self.sites["reach"]].rotation.copy()

        self.tau_lim = cs.torque_limits(m_mj)
        self.jlb, self.jub = _joint_bounds(m_mj)
        # S12 used w_land = 5e1 with an unweighted 3-D landing residual; `legacy`
        # restores exactly that so the old numbers are reproducible from this file.
        self.w_land = 5e1 if legacy else w_land
        self.w_hold, self.w_com = w_hold, w_com
        # THE COST BALANCE IS THE ROBUSTNESS STORY (S17).  These two were
        # hard-coded constants until the noise sweep was traced to them: a
        # quadratic barrier at weight 1e3 charges 0.14 for standing 17 mm outside
        # the foot polygon, against ~25 for the landing cost that is pulling the
        # arm down, so the optimiser sells the support polygon to buy the brace
        # and does it three orders of magnitude below cost.  Same for the cones,
        # which are the only thing stopping the plan from asking the rigid feet
        # for a wrench they cannot supply.
        self.w_cone, self.w_reach = w_cone, w_reach
        # COP MARGIN.  The wrench cone's centre-of-pressure rows are written
        # against the sole RECTANGLE, so a plan whose CoP rides the heel edge
        # satisfies them exactly -- and that is what the S13-S16 plan does.  A
        # CoP on the edge IS the tipping boundary, and the rigid 6D foot contact
        # in this OCP cannot represent the robot rotating over it: the instant
        # the plant does, the measured state is off the model's own constraint
        # manifold and the DDP's line search collapses (measured: alpha = 0.002,
        # no descent, at 1, 2 and 5 iterations and at every horizon up to the
        # whole remaining maneuver).  Shrinking the box the cone is written
        # against is how the plan is asked to keep a margin it can lose and
        # still be inside a model that describes what happens next.
        self.cop_shrink, self.min_nforce = cop_shrink, min_nforce
        self.w_com_track = w_com_track
        # LAUNCHING FROM REST.  The plan's first control was 112 N.m away from
        # the torque that statically holds the start pose (knees at -67 N.m
        # where +10 is wanted), so the maneuver opens with a lurch the plant
        # cannot absorb: held properly at `stand` and then handed this plan, the
        # robot topples.  Two knobs against it.  `w_ctrl` is crocoddyl's control
        # regulariser, which was 1e-3 -- effectively off -- and matters more
        # here than a control-effort term usually does, because the feet are
        # BILATERAL 6D welds: the two leg chains form a closed loop, so a whole
        # subspace of internal preload produces no motion in the model and costs
        # the objective nothing, while MuJoCo's unilateral feet answer it by
        # sliding.  `n_hold` gives the OCP the first `n_hold` nodes of the
        # approach to sit still at x0, so the departure from rest is planned
        # rather than assumed.
        self.w_ctrl, self.n_hold, self.w_hold_state = \
            w_ctrl, n_hold, w_hold_state
        # Horizontal centroidal LINEAR momentum, i.e. CoM velocity times mass.
        # The position terms above -- barrier and tracking alike -- can only
        # answer an excursion that has already happened; this is the derivative
        # half, and it is the one a topple announces itself in first.  The
        # angular rows are left alone: the lean is a deliberate rotation of the
        # trunk and penalising its momentum penalises the maneuver.
        self.w_com_damp = w_com_damp
        # ABLATION.  Names of cost GROUPS to leave out of every node, so
        # "is this term necessary" can be answered by deleting it and running
        # the same verdict rather than by reasoning about the objective.  A
        # group that can be dropped with no measurable consequence is a group
        # that should not be paid for at 60 Hz.
        self.drop = set(drop)
        # JOINT LIMITS.  The weight was hard-coded at 1e1 and the S17 ablation
        # called the term inert -- which was true of the single target it was
        # ablated on and false of the grid.  Over the 25 planned cells EVERY
        # plan comes within 5% of a joint bound and seven of them cross one, the
        # hip yaws by up to 2.6 deg (their range is only +-24.6 deg).  So the
        # barrier is active where it matters and, at 1e1, too weak to hold.
        self.w_jlim = w_jlim
        # HOW FAST THE APPROACH MOVES.  The planned move-to-brace peaks at
        # 2.5-3.2 rad/s and 68-104 rad/s^2, on the bracing shoulder roll: the
        # arm is thrown out rather than placed.  Nothing in the objective priced
        # that -- the state regulariser's velocity block rides at w_state = 1e-1
        # on the approach -- so this is a velocity-only term with its own weight.
        self.w_vel = w_vel
        # REACHING-HAND ORIENTATION.  Unconstrained, the solve keeps the tool
        # axis within 0.1 deg of horizontal all by itself, so asking for
        # `flat` should be free and asking for `down` should not.  What each
        # costs is the point of the experiment.
        self.reach_rot, self.w_reach_rot = reach_rot, w_reach_rot
        self.land_w = np.ones(3) if legacy else \
            np.array([1.0, 1.0, land_z_weight]) ** 2

        # SUPPORT POLYGON for the CoM barrier, as the axis-aligned box spanned by
        # the eight sole corners at the START pose.  Axis-aligned and shrunk by
        # `com_margin` on purpose: this is a barrier meant to keep the optimiser
        # from borrowing support it does not have during the feet-only phases,
        # not a tight stability certificate -- stability.equilibrium_region is
        # the certificate, and it is measured afterwards on what came out.
        d_mj.qpos[:len(q0_mj)] = q0_mj
        mujoco.mj_forward(m_mj, d_mj)
        corners = np.array([cs.point_world(m_mj, d_mj, b, o)
                            for f in cs.FEET
                            for b, o in cs.foot_corners(m_mj, d_mj, f)])
        self.support_lo = corners[:, :2].min(axis=0) + com_margin
        self.support_hi = corners[:, :2].max(axis=0) - com_margin

        # State-regularisation weights.  The floating base is NOT regularised to
        # q* (the base pose is an outcome of the lean, not a target); joints are,
        # so q* biases the solution without fighting the contacts.
        #
        # These are FLAT across the joints on purpose.  Weighting the bracing arm
        # 10x, to pull it onto the collision-free certified pose and keep the
        # gripper out of the table, destroys the solve outright: the staged
        # continuation goes back to the stalled signature (17 iterations, no step
        # accepted, cost stuck at its warm-start value).  The gripper penetration
        # is real and still open -- see the write-up -- but per-joint reweighting
        # is not the lever that fixes it.
        wq = np.concatenate([np.zeros(6), np.ones(self.rmodel.nv - 6)])
        wv = np.ones(self.rmodel.nv)
        self.w_state = np.concatenate([wq, wv]) ** 2

    # ----------------------------------------------------------------- costs --
    def _contacts(self, braced):
        contacts = crocoddyl.ContactModelMultiple(self.state, self.nu)
        for f in FEET:
            contacts.addContact(f, crocoddyl.ContactModel6D(
                self.state, self.sites[f], self.foot_ref[f],
                pin.LOCAL_WORLD_ALIGNED, self.nu, CONTACT_GAINS))
        if braced:
            for s in self.subset:
                contacts.addContact(f"brace_{s}", crocoddyl.ContactModel3D(
                    self.state, self.sites[s], self.site_ref[s],
                    pin.LOCAL_WORLD_ALIGNED, self.nu, CONTACT_GAINS))
        return contacts

    def _base_costs(self, braced, w_state=1e-1, w_ctrl=None, cones=None,
                    x_ref=None):
        if w_ctrl is None:
            w_ctrl = self.w_ctrl
        costs = crocoddyl.CostModelSum(self.state, self.nu)
        if "stateReg" not in self.drop:
            costs.addCost("stateReg", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(self.w_state),
                crocoddyl.ResidualModelState(
                    self.state, self.x_star if x_ref is None else x_ref,
                    self.nu)),
                w_state)
        if self.w_vel and "vel" not in self.drop:
            nv = self.state.nv
            costs.addCost("velReg", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.concatenate([np.zeros(nv), np.ones(nv)])),
                crocoddyl.ResidualModelState(self.state, self.state.zero(),
                                             self.nu)), self.w_vel)
        if "ctrlReg" not in self.drop:
            costs.addCost("ctrlReg", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelControl(self.state, self.nu)),
                w_ctrl)

        # Joint limits, as a two-sided barrier on the state.  The reference MUST be
        # state.zero() and not a zeroed copy of x_star: zeroing x_star zeroes the
        # floating base QUATERNION too, and every state.diff against a (0,0,0,0)
        # quaternion normalises by zero.  crocoddyl does not check for it -- the
        # process segfaults while the ShootingProblem is being assembled.
        nv = self.state.nv
        lb = np.concatenate([self.state.lb[1:nv + 1], self.state.lb[-nv:]])
        ub = np.concatenate([self.state.ub[1:nv + 1], self.state.ub[-nv:]])
        if "jointLim" not in self.drop:
            costs.addCost("jointLim", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelQuadraticBarrier(
                    crocoddyl.ActivationBounds(lb, ub)),
                crocoddyl.ResidualModelState(self.state, self.state.zero(),
                                             self.nu)), self.w_jlim)

        # Contact cones.  mu is the study's 0.6, and min_nforce > 0 is what makes
        # these UNILATERAL -- without it a "contact" is free to pull the robot down.
        #
        # The feet get a WRENCH cone and the bracing sites a FRICTION cone, and the
        # pairing is forced: a wrench cone is the 6D object (3 force rows + the
        # centre-of-pressure rows that keep the CoP inside the 0.20 x 0.08 m sole),
        # a friction cone is the 3D one.  Handing a 5-row friction cone to a 6D
        # contact does not raise a type error -- crocoddyl mis-sizes an allocation
        # and dies with a bare MemoryError, which is how this was found.
        if cones is None:
            cones = self.cones
        if "cones" in self.drop:
            cones = False
        if not cones:
            return costs
        wrench = crocoddyl.WrenchCone(np.eye(3), self.mu,
                                      np.array(cb.FOOT_HALF) * 2.0
                                      * self.cop_shrink,
                                      4, False, self.min_nforce, 1e4)
        wact = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(wrench.lb, wrench.ub))
        for f in FEET:
            costs.addCost(f"cone_{f}", crocoddyl.CostModelResidual(
                self.state, wact, crocoddyl.ResidualModelContactWrenchCone(
                    self.state, self.sites[f], wrench, self.nu)), self.w_cone)
        if braced:
            cone = crocoddyl.FrictionCone(np.eye(3), self.mu, 4, False,
                                          self.min_nforce, 1e4)
            act = crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(cone.lb, cone.ub))
            for s in self.subset:
                costs.addCost(f"cone_{s}", crocoddyl.CostModelResidual(
                    self.state, act, crocoddyl.ResidualModelContactFrictionCone(
                        self.state, self.sites[s], cone, self.nu)), self.w_cone)
        return costs

    # Named orientations for the REACHING gripper, as world rotations of its
    # frame.  The frame's x axis is the tool axis (REACH_OFF points along it),
    # so:
    #   flat   identity: tool axis forward, horizontal, palm down -- which is
    #          what the unconstrained solve already does to within 0.1 deg
    #   down   tool axis straight down, i.e. approach the tabletop from above,
    #          the orientation a pick off the table would need
    #   side   tool axis forward, rolled 90 deg: palm facing the far rail
    REACH_ROT = {
        "flat": np.eye(3),
        "down": np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]]),
        "side": np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]]),
    }

    def _reach_orientation(self, costs, nu, weight):
        """Hold the reaching gripper at a commanded world orientation."""
        if not self.reach_rot or not weight:
            return
        R = (self.reach_rot_auto if self.reach_rot == "auto"
             else self.REACH_ROT[self.reach_rot])
        costs.addCost("reachRot", crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelFrameRotation(
                self.state, self.sites["reach"], R, nu)), weight)

    def _keep_out(self, costs, nu, weight=1e3):
        """Keep every non-contact geom out of the TABLE BOX (croco_geom).

        One `ResidualModelFrameTranslation` per sampled point, referenced to the
        box centre, under the box keep-out activation.  The activation is exactly
        zero outside the inflated box, which is what lets the certified pose
        (gripper hanging over the near edge, 7 mm below the top plane) cost
        nothing while a body actually inside the slab is pushed out.  This is the
        term the S12 half-space could not be.

        `cg.activation` returns the C++ implementation when croco_ext is built
        and the identical Python one otherwise (cg.IMPL says which).

        With `cg.IMPL == "fused"` the whole point set is ONE cost term instead of
        86.  That is not a micro-optimisation of the same structure: 86 terms is
        ~130 us per node of CostModelSum accumulation that computes nothing (see
        croco_geom's docstring), and it is the single largest item in the MPC
        step.  The objective is unchanged -- same points, same thresholds, same
        weight, same diag(g^2) Hessian -- and croco_ext/test_keepout.py checks
        that on the real states.
        """
        if not self.keepout or "keepout" in self.drop:
            return
        fused = cg.fused_cost(self.state, nu, self.table_half, self.table_c,
                              self.keepout)
        if fused is not None:
            costs.addCost("ko_all", fused, weight)
            return
        for p in self.keepout:
            act = cg.activation(self.table_half, p["thresh"])
            costs.addCost(
                f"ko_{p['geom']}_{p['fid']}",
                crocoddyl.CostModelResidual(
                    self.state, act, crocoddyl.ResidualModelFrameTranslation(
                        self.state, p["fid"], self.table_c, nu)), weight)

    def _com_over_feet(self, costs, nu):
        """Barrier keeping the horizontal CoM inside the (shrunk) foot polygon.

        Only used on the FEET-ONLY phases.  Without it the approach is free to
        lean the CoM out over the table before there is anything under it, which
        is admissible to a prescribed-contact DDP (the feet are rigid pins that
        can pull) and is a topple in MuJoCo.  The braced phase deliberately does
        NOT carry it: extending the support region forward is the entire point of
        bracing, and the brace contacts are in the problem by then.
        """
        lo = np.array([self.support_lo[0], self.support_lo[1], -np.inf])
        hi = np.array([self.support_hi[0], self.support_hi[1], np.inf])
        act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lo, hi))
        if "comSupport" in self.drop:
            return
        costs.addCost("comSupport", crocoddyl.CostModelResidual(
            self.state, act, crocoddyl.ResidualModelCoMPosition(
                self.state, np.zeros(3), nu)), self.w_com)
        # A barrier is silent until it is already too late, and its weight is
        # therefore not the robustness knob it looks like: the plan cost is
        # IDENTICAL to 15 digits at w_com = 1e3 and at 1e6, because the nominal
        # trajectory never leaves the box -- and the perturbed runs that do
        # leave it are already falling when the term first speaks.  Measured in
        # the S17 weight table: 1e3 -> 1e6 changes the survival rate by nothing.
        # `w_com_track` is the always-on version: pull the CoM toward the middle
        # of the feet through the feet-only phases, so there is a gradient
        # BEFORE the excursion rather than a penalty after it.  Off by default.
        if self.w_com_track:
            mid = 0.5 * (self.support_lo + self.support_hi)
            costs.addCost("comTrack", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(np.array([1.0, 1.0, 0.0])),
                crocoddyl.ResidualModelCoMPosition(
                    self.state, np.array([mid[0], mid[1], 0.0]), nu)),
                self.w_com_track)
        if self.w_com_damp:
            costs.addCost("comDamp", crocoddyl.CostModelResidual(
                self.state,
                crocoddyl.ActivationModelWeightedQuad(
                    np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
                crocoddyl.ResidualModelCentroidalMomentum(
                    self.state, np.zeros(6), nu)), self.w_com_damp)

    def _table_barrier(self, costs, nu, weight=5e2):
        """Keep the whole bracing arm and the reaching hand ABOVE the table plane.

        The table is not a collision body in this OCP (see module docstring), so
        without this the arm is free to sweep straight through the slab.

        The barrier covers EVERY arm site -- elbow, forearm and palm -- not just
        the ones in the contact subset, and that distinction is the whole point.
        A site that is not selected as a contact is not thereby harmless: with the
        subset elbow+forearm and the barrier applied only to those two, the
        unconstrained magpie GRIPPER drove 80 mm through the tabletop (measured in
        MuJoCo off the planned trajectory; the resulting contact force was 154 kN).
        The links that are not bracing still have to not be inside the table.
        Checked against q*: every arm site clears table + 20 mm there -- palm by
        48 mm, elbow 23, forearm 14 -- so this constrains nothing the certified
        pose was relying on.  z-only: x and y are unbounded.
        """
        if self.table_z is None:
            return
        lo = np.array([-np.inf, -np.inf, self.table_z + self.clearance])
        hi = np.array([np.inf, np.inf, np.inf])
        act = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds(lo, hi))
        keep_out = sorted(set(self.subset) | set(cs.ARM_SITES) | {"reach"})
        for s in keep_out:
            fid = self.sites[s]
            costs.addCost(f"above_{s}", crocoddyl.CostModelResidual(
                self.state, act, crocoddyl.ResidualModelFrameTranslation(
                    self.state, fid, np.zeros(3), nu)), weight)

    def _geometry(self, costs, nu, feet_only):
        """The table and support-polygon terms for one node."""
        if self.legacy:
            self._table_barrier(costs, nu)
            return
        self._keep_out(costs, nu)
        # `not self.subset` is the legs-only mode, which the (mode x target)
        # sweep uses as its no-brace baseline.  There the "braced" phase has no
        # brace in it, so dropping the CoM barrier -- correct when a brace is
        # about to extend the support region -- would let the optimiser lean the
        # CoM out over nothing for the whole trajectory and call it a plan.
        if feet_only or not self.subset:
            self._com_over_feet(costs, nu)

    # --------------------------------------------------------------- phases --
    def build(self, dt=0.01, n_approach=120, n_braced=80, impulse=False,
              cones=None, w_terminal=1e0, n_return=0, dwell=0,
              enable_force=None):
        # `enable_force` is the last argument of
        # DifferentialActionModelContactFwdDynamics and it gates a real block of
        # work in calcDiff: df_dx, df_du and the per-contact updateForceDiff,
        # measured at 13 us of a 149 us braced node.  The ONLY costs that read
        # those derivatives are the cone residuals, so it follows `cones` rather
        # than being hard-coded True.  It does not gate the forces themselves --
        # `calc` still solves for lambda_c -- so croco_forces.py still reads the
        # planned wrenches off a cone-free solve.
        if enable_force is None:
            enable_force = self.cones if cones is None else bool(cones)
        approach = []
        n_hold = min(self.n_hold, max(n_approach - 2, 0))
        for k in range(n_approach):
            in_hold = k < n_hold
            costs = self._base_costs(
                braced=False, cones=cones,
                w_state=self.w_hold_state if in_hold else 1e-1,
                x_ref=self.x0 if in_hold else None)
            self._geometry(costs, self.nu, feet_only=True)
            # Guide each bracing site toward its certified landing spot, ramped in
            # over the phase so the pull is weak while the robot is still upright
            # and strong just before touchdown.
            #
            # S12 ramped this to a weight of 50 and left the reference at the full
            # 3-D landing point.  Measured outcome: the sites arrived 35 mm away
            # and, more damagingly, at the WRONG HEIGHT -- elbow 10 mm high,
            # forearm 15 mm low -- so the braced phase that follows was leaning on
            # a contact that in MuJoCo does not exist.  Two changes: the weight is
            # `w_land` (5e3 by default, i.e. above the reach cost rather than
            # below it) and the residual is z-weighted, because a lateral miss of
            # 20 mm still lands on a 0.6 x 1.2 m table while a vertical miss of
            # 20 mm is the difference between a contact and thin air.
            frac = 0.0 if in_hold else \
                (k - n_hold) / max(n_approach - n_hold - 1, 1)
            w = self.w_land * frac ** 2
            if w > 0 and "land" not in self.drop:
                for s in self.subset:
                    tgt = self.site_ref[s] + np.array(
                        [0, 0, self.clearance * (1 - frac)])
                    costs.addCost(f"land_{s}", crocoddyl.CostModelResidual(
                        self.state,
                        crocoddyl.ActivationModelWeightedQuad(self.land_w),
                        crocoddyl.ResidualModelFrameTranslation(
                            self.state, self.sites[s], tgt, self.nu)), w)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(False), costs, INV_DAMPING, enable_force)
            approach.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt))

        # Optional impulse node: the bracing sites touch down at zero velocity.
        #
        # OFF by default, and the default is the honest one for this maneuver.  An
        # impulse node has nu = 0, which makes the control trajectory ragged and,
        # more to the point, makes SolverBoxFDDP misbehave here -- it returns after
        # a handful of iterations with an uninitialised `stop` value and a
        # trajectory identical to the warm start.  Physically the cost of dropping
        # it is small: the brace is a hand placed deliberately on a table, so the
        # touchdown velocity the impulse would absorb is near zero by construction,
        # and the contact's Baumgarte gains acquire it over the first few braced
        # nodes.  Turning it on is `--impulse`, and the ragged-control handling
        # downstream is kept so that path still runs.
        impact = []
        if impulse:
            impulses = crocoddyl.ImpulseModelMultiple(self.state)
            for s in self.subset:
                impulses.addImpulse(f"brace_{s}", crocoddyl.ImpulseModel3D(
                    self.state, self.sites[s], pin.LOCAL_WORLD_ALIGNED))
            icosts = crocoddyl.CostModelSum(self.state, 0)
            icosts.addCost("stateReg", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ActivationModelWeightedQuad(self.w_state),
                crocoddyl.ResidualModelState(self.state, self.x_star, 0)), 1e-1)
            impact = [crocoddyl.ActionModelImpulseFwdDynamics(
                self.state, impulses, icosts)]

        braced = []
        for k in range(n_braced):
            costs = self._base_costs(braced=True, w_state=1e0, cones=cones)
            self._geometry(costs, self.nu, feet_only=False)
            # DWELL.  Without a return phase the terminal node holds the hand on
            # the target and that is enough.  With one, the DDP smooths across the
            # phase boundary and starts leaving before it has arrived -- measured
            # 81 mm of reach error at the end of the braced phase against 1.7 mm
            # for the same plan without a return, and |v| = 1.5 where it should be
            # at rest.  So the last `dwell` braced nodes get the terminal reach
            # weight and a velocity regulariser: arrive, stop, hold, then leave.
            in_dwell = dwell and k >= n_braced - dwell
            if self.reach_target is not None:
                costs.addCost("reach", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ResidualModelFrameTranslation(
                        self.state, self.sites["reach"], self.reach_target,
                        self.nu)),
                    10 * self.w_reach if in_dwell else self.w_reach)
                self._reach_orientation(
                    costs, self.nu,
                    (10 if in_dwell else 1) * self.w_reach_rot)
            if in_dwell:
                nv = self.rmodel.nv
                w_rest = np.concatenate([np.zeros(nv), np.ones(nv)])
                costs.addCost("restReg", crocoddyl.CostModelResidual(
                    self.state, crocoddyl.ActivationModelWeightedQuad(w_rest),
                    crocoddyl.ResidualModelState(self.state, self.x_star,
                                                 self.nu)), 1e2)
            # HOLD the brace where it landed.  ContactModel3D is written with
            # Baumgarte Kp = 0, so it constrains the site's VELOCITY and not its
            # position: once the braced phase starts, whatever height the site
            # arrived at is the height it keeps, and the S12 plan spent all 41
            # braced nodes with the elbow 10 mm above the table and the forearm
            # 15 mm inside it.  This is the position half the contact does not do.
            if not self.legacy and "hold" not in self.drop:
                for s in self.subset:
                    costs.addCost(f"hold_{s}", crocoddyl.CostModelResidual(
                        self.state,
                        crocoddyl.ActivationModelWeightedQuad(self.land_w),
                        crocoddyl.ResidualModelFrameTranslation(
                            self.state, self.sites[s], self.site_ref[s],
                            self.nu)), self.w_hold)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(True), costs, INV_DAMPING, enable_force)
            braced.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt))

        # RETURN: brace released, feet only again, back to the start pose.  The
        # brace contacts simply leave the problem -- a release is a contact
        # REMOVAL, which needs no impulse and no complementarity, and is the one
        # mode transition a prescribed-schedule DDP handles for free.  The state
        # regulariser flips from q* to the start pose and the CoM barrier comes
        # back, since the robot is on its feet alone again.
        ret = []
        for k in range(n_return):
            costs = self._base_costs(braced=False, w_state=1e0, cones=cones,
                                     x_ref=self.x_return)
            self._geometry(costs, self.nu, feet_only=True)
            dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
                self.state, self.actuation, self._contacts(False), costs, INV_DAMPING, enable_force)
            ret.append(crocoddyl.IntegratedActionModelEuler(dmodel, dt))

        # Terminal.  Without a return phase this is the braced pose: at q*, at
        # rest, hand on target.  WITH one it is the start pose, on the feet, brace
        # released -- so the terminal contact set has to follow the last running
        # phase or the problem ends with a contact the node before it does not have.
        terminal_braced = n_return == 0
        tcosts = self._base_costs(braced=terminal_braced, w_state=w_terminal,
                                  w_ctrl=0.0, cones=cones,
                                  x_ref=None if terminal_braced else self.x_return)
        tcosts.removeCost("ctrlReg")
        self._geometry(tcosts, self.nu, feet_only=not terminal_braced)
        if self.reach_target is not None and terminal_braced:
            tcosts.addCost("reach", crocoddyl.CostModelResidual(
                self.state, crocoddyl.ResidualModelFrameTranslation(
                    self.state, self.sites["reach"], self.reach_target, self.nu)),
                10 * self.w_reach)
            self._reach_orientation(tcosts, self.nu, 10 * self.w_reach_rot)
        tdmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
            self.state, self.actuation, self._contacts(terminal_braced), tcosts,
            INV_DAMPING, enable_force)
        terminal = crocoddyl.IntegratedActionModelEuler(tdmodel, 0.0)

        # Hand the torque limits to the solver as BOX BOUNDS.  SolverBoxFDDP only
        # enforces limits that the action models actually carry; without this it
        # silently degrades to plain FDDP and the `clamp` basis is documented but
        # not applied.
        for mdl in approach + braced + ret:
            mdl.u_lb = -self.tau_lim
            mdl.u_ub = self.tau_lim

        self.dt = dt
        self.n_approach, self.n_braced, self.n_return = \
            n_approach, n_braced, n_return
        return crocoddyl.ShootingProblem(
            self.x0, approach + impact + braced + ret, terminal)


# --------------------------------------------------------------------------- #
def solve_staged(ocp, dt, n_approach, n_braced, iters=200, impulse=False,
                 w_terminal=1e0, verbose=True, n_return=0, dwell=0):
    """Two-stage continuation: solve without cones, then re-solve with them.

    Handing the cone costs an infeasible warm start does not work.  The
    interpolated guess has zero velocity everywhere, so the forces it implies are
    nothing like the ones that carry the motion, and the cone barriers evaluate at
    ~3150 against a total cost of ~2.8 for everything else -- three orders of
    magnitude of the objective is a constraint the guess was never going to
    satisfy.  From there FDDP's backward pass fails at every trial step, the
    regularization walks 1e-8 -> 1e9, and it gives up after 17 iterations having
    moved nothing.  Measured at four cone weights and two min-normal-forces: all
    fail identically.

    Solving the same problem with the cones OFF converges in a handful of
    iterations, and its trajectory is a warm start that already has plausible
    contact forces -- so the cones then enter as a correction rather than as the
    dominant term.  Stage 1 is not a throwaway: it is the only thing that makes
    stage 2 well-posed.
    """
    stage1 = ocp.build(dt=dt, n_approach=n_approach, n_braced=n_braced,
                       impulse=impulse, cones=False, w_terminal=w_terminal,
                       n_return=n_return, dwell=dwell)
    s1, ok1, t1 = solve(stage1, ocp, iters=iters, verbose=verbose,
                        n_approach=n_approach, n_return=n_return)
    if not ocp.cones:
        return s1, ok1, t1, None

    stage2 = ocp.build(dt=dt, n_approach=n_approach, n_braced=n_braced,
                       impulse=impulse, cones=True, w_terminal=w_terminal,
                       n_return=n_return, dwell=dwell)
    s2 = crocoddyl.SolverBoxFDDP(stage2)
    if verbose:
        s2.setCallbacks([crocoddyl.CallbackVerbose()])
    t0 = time.time()
    ok2 = s2.solve(list(s1.xs), list(s1.us), iters, False, 1e-9)
    return s2, ok2, t1 + (time.time() - t0), (ok1, float(s1.cost))


def solve(problem, ocp, iters=200, verbose=True, n_approach=None, n_return=0):
    """FDDP with a warm start that interpolates stand -> q* (-> stand).

    Warm-starting every node at x0 (the usual crocoddyl boilerplate) is a bad
    initial guess here: the whole point of the problem is a large, slow
    configuration change, so a constant guess puts every node at the far end of
    the very nonlinearity the rollout has to cross, and the line search spends its
    iterations at step 2e-3 without ever leaving the initial point.

    The interpolation has to run over the WHOLE configuration via pin.interpolate,
    not just the 27 joint angles.  Interpolating joints while holding the floating
    base at its stand value swings the legs underneath a pelvis that never
    descends, which lifts both feet ~0.39 m off the floor by the final node -- a
    guess that violates the very contacts the problem is built around, and a worse
    starting point than the constant one it replaced.  pin.interpolate also does
    the right thing on the base quaternion, which a componentwise lerp does not.
    """
    solver = crocoddyl.SolverBoxFDDP(problem)
    if verbose:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])

    # With a return phase the guess has to come BACK: interpolating monotonically
    # to q* over a problem that ends at the start pose gives every return node a
    # guess at the far end of the maneuver, which is the same bad-guess failure
    # the constant warm start had, just at the other end of the trajectory.
    T = problem.T
    turn = (T - n_return) if n_return else T
    xs = []
    for k in range(T + 1):
        s = k / turn if k <= turn else 1.0 - (k - turn) / max(T - turn, 1)
        q = pin.interpolate(ocp.rmodel, ocp.q0, ocp.q_star, min(max(s, 0.0), 1.0))
        xs.append(np.concatenate([q, np.zeros(ocp.rmodel.nv)]))
    # CLAMP the warm-start controls into the torque box.  quasiStatic() answers
    # "what torque holds this configuration", and on the interpolated guess -- a
    # deep lean with the brace not yet down -- that answer is 2.2x the clamp basis
    # for elbow+forearm and 2.9x for elbow+forearm+palm.  SolverBoxFDDP projects
    # the control onto its box before it uses it, so those out-of-box entries
    # become a rollout the guess was never evaluated at: cost 4686 against 132 for
    # the guess itself, no trial step ever accepted, and the run ends after 17
    # iterations with an UNINITIALISED `stop` (9e-310).  That signature has been
    # blamed on the cones, on the impulse node and on the terminal weight in this
    # study; here it is simply an initial control outside the bounds the solver
    # enforces, and clamping it is what lets the three-contact mode solve at all.
    us = [np.clip(u, -ocp.tau_lim, ocp.tau_lim) for u in problem.quasiStatic(xs[:-1])]
    t0 = time.time()
    ok = solver.solve(xs, us, iters, False, 1e-9)
    return solver, ok, time.time() - t0


def report(solver, ocp):
    """Measure the plan against the things it was supposed to achieve.

    Scored at the END OF THE BRACED PHASE, not at the terminal node: with a
    return phase the terminal node is back at the start pose, where "how far is
    the hand from the target" is not a question about the plan's quality.
    """
    xs = np.array(solver.xs)
    n_ret = getattr(ocp, "n_return", 0)
    k_score = len(xs) - 1 - n_ret
    q_end = xs[k_score][:ocp.rmodel.nq]
    v_end = xs[k_score][ocp.rmodel.nq:]
    pin.forwardKinematics(ocp.rmodel, ocp.rdata, q_end)
    pin.updateFramePlacements(ocp.rmodel, ocp.rdata)

    out = {"cost": float(solver.cost), "iters": int(solver.iter),
           "stop": float(solver.stop), "is_feasible": bool(solver.isFeasible),
           "score_node": int(k_score)}
    out["site_err"] = {s: float(np.linalg.norm(
        ocp.rdata.oMf[ocp.sites[s]].translation - ocp.site_ref[s]))
        for s in ocp.subset}
    if ocp.reach_target is not None:
        out["reach_err"] = float(np.linalg.norm(
            ocp.rdata.oMf[ocp.sites["reach"]].translation - ocp.reach_target))
    # NB the sole frames are registered as "site_sole_left"/"site_sole_right", so
    # they have to be looked up through ocp.sites, not getFrameId(f).
    out["foot_err"] = {f: float(np.linalg.norm(
        ocp.rdata.oMf[ocp.sites[f]].translation - ocp.foot_ref[f].translation))
        for f in FEET}
    out["terminal_vel_norm"] = float(np.linalg.norm(v_end))
    # distance from the certified pose, over the 27 actuated joints only
    out["q_err_vs_qstar_rad"] = float(np.linalg.norm(q_end[7:] - ocp.q_star[7:]))
    out["q_err_max_joint_rad"] = float(np.max(np.abs(q_end[7:] - ocp.q_star[7:])))
    # The impulse node carries nu = 0, so solver.us is a ragged list: one entry is
    # a length-0 array and np.array() on the whole thing gives an object array.
    us = np.array([u for u in solver.us if len(u) == ocp.nu])
    out["max_torque_ratio"] = float(np.max(np.abs(us) / ocp.tau_lim))
    out["torque_ratio_argmax_joint"] = cb.ACTUATED[
        int(np.argmax(np.max(np.abs(us) / ocp.tau_lim, axis=0)))]

    # BRACE HEIGHT over the braced phase.  The single number the S12 plan needed
    # and did not have: a brace contact that the OCP is drawing force from while
    # the site hovers above the table is a contact that does not exist in MuJoCo.
    # Reported as an offset from the CERTIFIED site height, so 0 means "as far
    # into the table as the pose the static QP signed off".
    if getattr(ocp, "n_approach", None) is not None and ocp.subset:
        b0, b1 = ocp.n_approach, len(xs) - n_ret
        dz = {s: [] for s in ocp.subset}
        for k in range(b0, b1):
            pin.forwardKinematics(ocp.rmodel, ocp.rdata, xs[k][:ocp.rmodel.nq])
            pin.updateFramePlacements(ocp.rmodel, ocp.rdata)
            for s in ocp.subset:
                dz[s].append(float(
                    ocp.rdata.oMf[ocp.sites[s]].translation[2]
                    - ocp.site_ref[s][2]))
        out["brace_dz_vs_qstar"] = {
            s: {"min": min(v), "max": max(v), "mean": float(np.mean(v))}
            for s, v in dz.items()}
        out["brace_dz_worst_mm"] = 1000.0 * max(
            max(abs(min(v)), abs(max(v))) for v in dz.values())
    return out
