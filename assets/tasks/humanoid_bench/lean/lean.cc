#include "mjpc/tasks/humanoid_bench/lean/lean.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>

#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc {

// T1 REFERENCE TRIM v2 -- ported from stabilize.cc (commit 1708253) 2026-07-20.
// Written by TransitionLocked (real state, once per plan) and read by the
// rollout workers in Residual (benign double read; changes on a seconds
// timescale, same pattern as stabilize's s_cap_ex_dc).
//
// WHY THE STAND NEEDS IT (2026-07-20 real): with the IMU offset finally
// measured honest (pitch 0.0 / roll 1.3) the stand held 83 s with base_z
// sd 0.005 -- but still walked a slow FORWARD ramp (lean +0.2 deg at t=4 ->
// +6.8 deg at t=72) that loaded LankP to its 44.8 Nm useful cap (49.3% of
// ticks budget-clamped, 29.3 deg tracking error) until the left leg gave up.
// That residual bias is ankle-zero + CoM-model error: it is NOT constant
// (both 07-20 recordings show the solved ankle zeros drifting 4-8 deg WITHIN
// one 50 s run), so no XML constant can cancel it. A leaky integral does not
// need to know the bias -- it nulls whatever steady park it observes.
//
// Recipe (Stephens IROS'07 integral CoP/posture; Caron ICRA'19 leaky DCM
// integral) with the four v1 defects already fixed upstream: leak, support
// frame, nominal 0, quiet gate. See stand_trim_* in Lean_H12_Magpie.xml.
// stand_trim_tau = 0 (the shipped default) forces both to 0 => byte-identical.
static double s_trim_x = 0.0, s_trim_y = 0.0;

namespace {
// Target (post-ramp) reach + brace + posture scales for each named phase. Kept
// in one place so the residual and the transition logic can't drift out of
// sync. Posture is boosted during stand_up because the audit-spec PD gains
// (ankle kp=20, knee kp=200) aren't stiff enough on their own to pull a
// drifted knee back to extension — the Posture cost has to do it via MPC.
//
// ★★★ TRUE BRACE LOAD (2026-07-13). Sum the normal forces of every contact between
// the TABLE and the bracing arm (elbow / wrist links / gripper), straight from
// mjData.
//
// Why not a touch sensor: MuJoCo's touch sensor is STRUCTURALLY unable to do this.
// engine_sensor.c (mjSENS_TOUCH) filters with
//     bodyid = m->site_bodyid[objid];
//     if (con->efc_address >= 0 && (bodyid == conbody[0] || bodyid == conbody[1]))
// -- it only sums contacts involving the site's OWN BODY. The brace site lives on
// the elbow link, so it is blind to the wrist links no matter how large the sensor
// zone is made. That mattered the moment the arm was made solid against the table
// (the table<->wrist/gripper excludes were deleted, 2026-07-13): the brace load
// moved onto the WRIST, and the sensor sat at 0.0 N while ~38 N flowed through the
// contact. Measured: brace_probe Fbrace=0.0 for the entire 18 s hold while
// brace_contact_audit showed geom48[left_wrist_pitch_link] carrying 38.1 N. Every
// brace cost gated on brace_contact_force (support-widening hand_load_frac, the P3
// CoM load transfer, the P4 reach gate, the brace-force shortfall reward) was
// therefore silently DEAD -- the controller could not feel its own brace.
//
// Reading the contacts directly is exact, pose-independent, and immune to which
// link happens to bite first, which is the whole point: the braced forearm+wrist is
// ONE contact surface and the load wanders along it.
inline double TableBraceForce(const mjModel* model, const mjData* data,
                              bool brace_left) {
  int tgeom = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
  if (tgeom < 0) return 0.0;
  int tbody = model->geom_bodyid[tgeom];
  const char* want = brace_left ? "left_" : "right_";
  size_t wlen = std::strlen(want);
  double total = 0.0;
  mjtNum f6[6];
  for (int i = 0; i < data->ncon; i++) {
    const mjContact* c = data->contact + i;
    if (c->efc_address < 0) continue;                 // inactive (within margin, no force)
    int b0 = (c->geom[0] >= 0) ? model->geom_bodyid[c->geom[0]] : -1;
    int b1 = (c->geom[1] >= 0) ? model->geom_bodyid[c->geom[1]] : -1;
    int other = (b0 == tbody) ? b1 : ((b1 == tbody) ? b0 : -1);
    if (other < 0) continue;
    const char* bn = mj_id2name(model, mjOBJ_BODY, other);
    if (!bn || std::strncmp(bn, want, wlen) != 0) continue;
    if (!std::strstr(bn, "elbow") && !std::strstr(bn, "wrist") &&
        !std::strstr(bn, "gripper")) {
      continue;                                        // not the bracing arm
    }
    mj_contactForce(model, data, i, f6);
    if (f6[0] > 0.0) total += f6[0];                   // f6[0] = normal force
  }
  return total;
}

inline void PhaseTargetScales(const std::string& name,
                              double& reach, double& brace_pos,
                              double& posture) {
  reach = 1.0;
  brace_pos = 1.0;
  posture = 1.0;
  // stand_up: pure stabilisation. Reach + brace cost go to 0 so the only
  // signal pulling on the robot is balance + posture. Posture ×3 keeps legs
  // extended at the audit-spec PD (knee + hip_pitch have no other home pull).
  if (name == "stand_up") {
    reach = 0.0; brace_pos = 0.0; posture = 3.0;
  }
  // reach_to_target: "reach an input target" primitive. The chosen hand reaches
  // toward a world point and the feet stay planted with NO brace; the target is
  // the EXTERNAL mocap object (object_pos, clamped to a balance-safe workspace --
  // see Residual()). posture=1.0 (NOT higher): the legs are locked by dedicated
  // terms (Knees Straight / Base Height / Symmetry in the JSON), so a low global
  // Posture leaves the reaching ARM free to extend -- a high Posture parks the
  // limb at its rest pose.
  else if (name == "reach_to_target") {
    reach = 1.0; brace_pos = 0.0; posture = 1.0;
  }
  // forearm_brace_lean: braced lean with the forearm pad on the table. Reach and
  // brace-position both fully active; the contact set is the keyframe's
  // ContactPair list.
  else if (name == "forearm_brace_lean") {
    reach = 1.0; brace_pos = 1.0; posture = 1.0;
  }
  // Any other name falls through at 1.0/1.0/1.0. Only the three names above can
  // occur: they are the complete phase vocabulary of the five live strategies.
}
}  // namespace

// ------------------ Residuals for humanoid lean task ------------
//   Number of residuals:
//      Residual(0): humanoid_bench reward
//      Residual(1): Height: head feet vertical error
//      Residual(2): CoM Velocity
//      Residual(3): joint velocity
//      Residual(4): balance
//      Residual(5): torso forward tilt (NEW - encourages leaning)
//      Residual(6): pelvis tilt (NEW - allows forward lean)
//      Residual(7): posture
//      Residual(8): velocity
//      Residual(9): control
//      Residual(10): object distance (reaching hand)
//      Residual(11): right hand distance to object
//      Residual(12): left hand brace position on table (NEW)
//   Number of parameters:
//      Parameter(0): head height goal
// ----------------------------------------------------------------
void lean::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  double const height_goal = parameters_[0];
  int counter = 0;

  // FOREARM BRACE (2026-07-01): the forearm_brace_lean phase braces the LEFT
  // forearm (elbow_link pad) on the table via an explicit <pair>. It carries NO
  // declared JSON ContactPair (the physics + force are on the pad/sensor, not a
  // cost-side body pair), so mark it arm-contact BY NAME so the Base-Height
  // anchor releases (deep squat allowed) and the support-widening triangle
  // activates -- both otherwise gated on a declared JSON pair.
  const bool is_forearm_brace = (residual_keyframe_.name == "forearm_brace_lean");

  // ----- stage detection: used throughout to gate residual scaling ----- //
  int active_contact_count_early = 0;
  for (const auto& cp : residual_keyframe_.contact_pairs) {
    if (cp.body1 != mjpc::humanoid::kNotSelectedInteract &&
        cp.body2 != mjpc::humanoid::kNotSelectedInteract) {
      active_contact_count_early++;
    }
  }
  // ★ 2026-08-02 THE ARM IS STILL ON THE TABLE DURING THE REACH AND THE RELEASE.
  // `is_forearm_brace` matches the EXACT name "forearm_brace_lean", so
  // any_arm_contact used to go FALSE the instant the reach phase began, even
  // though the forearm was still pressing the slab with 55-60 N. At that single
  // tick three things flipped at once:
  //   * the braced support polygon collapsed to the free-standing FOOT-FOOT LINE
  //     (which has no forward extent at all) while the CoM sat at +0.153, i.e.
  //     PAST the toe -- Balance suddenly reported a huge excursion;
  //   * fwd_scale jumped 0.80 -> 1.0, so that excursion was also priced strictly;
  //   * the Base-Height anti-sink anchor (JSON weight 450) switched ON.
  // That is a cost CLIFF, not a transition -- `target_ramp_sec` ramps the
  // keyframe target, never these booleans. It is the recorded "falls cluster
  // right after the REACH starts", and the rough, un-tendered stand-up the user
  // saw on the render.
  // Widen the gate to every phase where the forearm is on the table, WITHOUT
  // touching `is_forearm_brace` itself -- its other 11 call sites (wrist cock,
  // brace_foot_x, brace_reach_gate, kBraceGateFix) are genuinely lean-only.
  // Honesty is preserved by the load gate downstream: the polygon's third vertex
  // still grows only with MEASURED brace force, so as the arm lifts during the
  // release the support shrinks back to the feet on its own and the planner is
  // obliged to bring the CoM home before it can stand.
  const bool arm_braced_phase =
      is_forearm_brace ||
      residual_keyframe_.name == "forearm_brace_reach" ||
      residual_keyframe_.name == "forearm_brace_release";
  // any_arm_contact: arm is on the table (stand_up has 0 contacts)
  const bool any_arm_contact      = (active_contact_count_early >= 1) || arm_braced_phase;
  // DESIGN (2026-05-26): the leg-lift phase is DROPPED permanently — both feet
  // stay grounded through the whole pipeline (see the lean.h header). The
  // single-support branches that keyed off the old lift phase names were removed
  // 2026-07-31 (permanently unreachable: the live strategies name only stand_up,
  // reach_to_target and forearm_brace_lean). Any future single-support move must
  // come from a WBC/balance decision, not a hard phase-name gate.

  // ---- Phase-aware cost scales (fixes the "hip slam at t=0" bug) ----------
  // The height-based `leaning` scalar below is ~1.0 when the robot is upright,
  // which made the reach-toward-object cost fully active during stand_up.
  // MJPC then planned a forward lunge from frame 0 and slammed the hip into
  // the table. We gate reach + brace residuals on the keyframe name AND
  // smoothly interpolate from the previous phase's scales over kPhaseRampSeconds.
  // That smooth handoff is what stops the robot from snapping forward the
  // instant the next phase's cost gradient switches on — i.e. the WBC-style
  // "blend tasks, don't switch them" behaviour.
  double target_reach_scale     = 1.0;
  double target_brace_pos_scale = 1.0;
  double target_posture_scale   = 1.0;
  PhaseTargetScales(residual_keyframe_.name, target_reach_scale,
                    target_brace_pos_scale, target_posture_scale);

  // Smooth time-ramp from the previous phase's scales to the new ones.
  //
  // We pass the linear progression α = clamp(t/T, 0, 1) through a smoothstep
  // curve s(α) = α² · (3 − 2α) before interpolating. Properties:
  //   • s(0) = 0, s(1) = 1                      → matches endpoints
  //   • s'(0) = s'(1) = 0                       → zero slope at endpoints
  //   • C¹ continuous                            → cost gradient changes gradually
  // The zero slopes are what stops the abrupt shove when a phase begins/ends:
  // with linear lerp the cost gradient jumps from "rising at constant rate"
  // to "constant" instantly, which MJPC reads as an impulsive cost change and
  // plans a snap response. Smoothstep eases in and out symmetrically, which
  // is the canonical "weight-based task transition" used in HQP-style WBC
  // (see e.g. Liu et al., "Generalized hierarchical control" — task priority
  // weights as continuous functions of time, not step changes).
  double time_in_phase = mju_max(0.0, data->time - keyframe_start_time_);
  // CONTROLLED LEAN-IN *AND* RECOVERY (2026-07-01): a heavy humanoid moving INTO or
  // OUT OF a table brace over the default 1.5 s ramp reads as an abrupt lunge/snap
  // (user: "too abrupt... do it slowly in a controlled way" -- both descending AND
  // standing back up). Detect a brace transition in EITHER direction (entering the
  // brace, or LEAVING it for another pose e.g. a live-switch to stand) and stretch
  // the cost-gradient ramp to the ENTERED phase's JSON target_ramp_sec so Brace Pos
  // / Reaching / brace-force / support all ease in/out gently. Non-brace transitions
  // keep the original 1.5 s ramp (brace_transition guards it -> byte-identical).
  int brace_key_id_ramp = mj_name2id(model, mjOBJ_KEY, "forearm_brace_lean");
  bool prev_was_forearm_brace =
      (brace_key_id_ramp >= 0 && prev_posture_key_id_ == brace_key_id_ramp);
  bool brace_transition = is_forearm_brace || prev_was_forearm_brace;
  double phase_ramp_seconds =
      (brace_transition && residual_keyframe_.target_ramp_sec > 1e-9)
          ? residual_keyframe_.target_ramp_sec
          : kPhaseRampSeconds;
  // ★ 2026-07-28 `phase_ramp_sec` numeric: 0 = OFF = BYTE-IDENTICAL (the logic above).
  // >0 overrides the ramp for EVERY transition, not just brace ones.
  // WHY: every surviving failure of the fixed strat-16/21 configs happens 58-61 s, i.e.
  // ~2 s after the 57 s switch -- right as this 1.5 s ramp completes and the full new
  // cost surface lands. That is a HANDOVER TRANSIENT, not a steady-state balance loss
  // (the same configs then hold 240 s when they survive it). Non-brace transitions had
  // NO way to slow this down: `target_ramp_sec` is gated on `brace_transition`, so the
  // 16 and 21 JSONs' own target_ramp_sec was silently ignored. Live-tunable so the
  // handover can be swept without a rebuild.
  {
    int pr_id = mj_name2id(model, mjOBJ_NUMERIC, "phase_ramp_sec");
    double pr = (pr_id >= 0) ? model->numeric_data[model->numeric_adr[pr_id]] : 0.0;
    if (pr > 1e-9) phase_ramp_seconds = pr;
  }
  double alpha_lin     = mju_min(time_in_phase / phase_ramp_seconds, 1.0);
  double alpha         = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  double phase_reach_scale =
      prev_phase_reach_scale_ + alpha * (target_reach_scale -
                                         prev_phase_reach_scale_);
  double phase_brace_pos_scale =
      prev_phase_brace_pos_scale_ + alpha * (target_brace_pos_scale -
                                             prev_phase_brace_pos_scale_);
  double phase_posture_scale =
      prev_phase_posture_scale_ + alpha * (target_posture_scale -
                                           prev_phase_posture_scale_);

  // ----- Pose-library target keyframe ------------------------------------ //
  // A standalone "simple task" strategy (stand / crouch / arms_sideways / …)
  // names its phase after a model <key> keyframe. When a keyframe with that
  // exact name exists, the Posture + Control costs track THAT pose instead of
  // the home keyframe — turning the lean task into a selectable pose-library
  // player. mj_name2id returns -1 when no keyframe matches, so every existing
  // phase whose name is not also a keyframe name falls back to key 0 (home)
  // and behaves exactly as before. The lookup is one hash probe per Residual
  // call — negligible next to the ~10 std::string compares PhaseTargetScales
  // already does.
  // REVERTED 2026-06-08: the squat target-pose ramp (asymmetric stand-up/crouch
  // ramp) is OFF — restored to the original instantaneous SNAP. The ramp ran in
  // this hot residual path for EVERY strategy (incl. single-phase crouch/stand/
  // arms) and was suspected of regressing the live crouch (one-knee-locked slow
  // creep) even though headless held. Snap = exactly the accepted behaviour.
  // The prev_posture_key_id_ plumbing (lean.h, SnapshotEffectiveScales) is left
  // in place but UNUSED; re-enable the ramp here only behind a per-strategy gate
  // that provably never touches single-phase strategies. See [[project_squat_strategy]].
  int posture_key_id =
      mj_name2id(model, mjOBJ_KEY, residual_keyframe_.name.c_str());
  if (posture_key_id < 0) posture_key_id = 0;  // home
  const mjtNum *posture_target = model->key_qpos + posture_key_id * model->nq;
  // one-shot forensic: the hand-off snap survived 6 cost fixes -- verify LIVE which
  // keyframe the residual actually pulls toward (assumed 'stand' knee=0.35 z=0.98)
  static int dbg_last_key = -999;
  if (posture_key_id != dbg_last_key) {
    dbg_last_key = posture_key_id;
    std::fprintf(stderr,
                 "[lean-residual] posture keyframe: name='%s' id=%d z=%.3f kneeL=%.3f kneeR=%.3f\n",
                 residual_keyframe_.name.c_str(), posture_key_id,
                 posture_target[2], posture_target[7 + 3], posture_target[7 + 9]);
  }

  // ----- GENERAL target-pose ramp (re-enabled 2026-06-16, behind the per-strategy
  //       gate the 2026-06-08 revert note above requires) --------------------- //
  // On a phase transition, ease the posture target from the keyframe we LEFT
  // (prev_posture_key_id_, captured at every transition) into the new one over a
  // smoothstep window -- the "blend tasks, don't switch them" idea the weight ramp
  // already uses, applied to the pose itself. Because it lives in the planner's OWN
  // cost, the MJPC GUI, the twin and the real robot glide identically -- unlike a
  // node-side, plant-only command blend, this opens no sim2real gap. Ramp duration
  // = the entered phase's JSON target_ramp_sec when >=0, else kPhaseRampSeconds.
  //   GATED num_phases_ > 1: single-phase strategies (stand/crouch/arms) never take
  //   this branch -> byte-identical. WINDOW-GATED: only runs while alpha<1 (the
  //   ~ramp_dur right after a transition), into a stack buffer, so steady state and
  //   every snap phase (target_ramp_sec==0) pay nothing -- the unconditional version
  //   that ran for EVERY residual call of EVERY strategy is what got reverted.
  mjtNum ramped_posture_target[64];
  if ((num_phases_ > 1 || brace_transition) &&
      prev_posture_key_id_ != posture_key_id && model->nq <= 64) {
    // Enable the pose-target ramp for multi-phase strategies AND either side of a
    // forearm-brace transition (descent glides home->bow; recovery glides bow->stand,
    // not snap). Other single-phase strats (stand/crouch/arms) still snap on a cold
    // load -- prev_posture_key_id_ == posture_key_id there -> this branch is skipped,
    // so the 2026-06-08 revert (no ramp for single-phase stand/crouch/arms) holds.
    double ramp_dur = (residual_keyframe_.target_ramp_sec >= 0.0)
                          ? residual_keyframe_.target_ramp_sec
                          : kPhaseRampSeconds;
    if (ramp_dur > 1e-9 && time_in_phase < ramp_dur) {
      double ra_lin = mju_min(time_in_phase / ramp_dur, 1.0);
      double ra = ra_lin * ra_lin * (3.0 - 2.0 * ra_lin);  // smoothstep
      const mjtNum *from_target =
          model->key_qpos + prev_posture_key_id_ * model->nq;
      for (int i = 0; i < model->nq; i++) {
        ramped_posture_target[i] =
            from_target[i] + ra * (posture_target[i] - from_target[i]);
      }
      posture_target = ramped_posture_target;
    }
  }

  // ---- POSTURE LEG-CHAIN LEVELLER (posture_leg_level, 0 = OFF) ------------ //
  // ★★ 2026-07-27: THE STANDING ORDER TO LEAN FORWARD.
  // A flat-footed, upright-pelvis pose requires the leg PITCH chain to close:
  //     hip_pitch + knee + ankle_pitch == 0
  // MEASURED on the compiled model (build tree), left leg:
  //     stand / stand_up           -0.360 +0.550 -0.190 = +0.0000  ->  0.00 deg
  //     reach_to_target            -0.250 +0.350 -0.210 = -0.1100  -> +6.30 deg FWD
  //     counterbalance_standing    -0.250 +0.350 -0.210 = -0.1100  -> +6.30 deg FWD
  //     stumble_trot               -0.250 +0.350 -0.210 = -0.1100  -> +6.30 deg FWD
  //     stumble_trot_drive         -0.140 +0.350 -0.210 = -0.0000  ->  0.00 deg
  // So with both soles flat on the floor, strat 21 / 16 are told -- at every
  // rollout, for the whole run -- to pitch the pelvis 6.3 deg FORWARD. Posture
  // weight is 12.0 in strat 21 and 12.0 in strat 6: IDENTICAL. The forward pull was
  // never in a weight, which is why ~50 cost-re-weighting runs could not remove it.
  //
  // ★ THE IDENTICAL BUG WAS ALREADY FOUND AND FIXED ONCE, WITH THE IDENTICAL NUMBER.
  // 2026-07-20 (DRIVE-24): "idle-tips-fwd keyframe lean FIXED hipP -0.25 -> -0.14
  // (drive-only)". That is exactly -(knee + ankleP) = -(0.350 - 0.210) = -0.140, the
  // value that closes the chain -- and `stumble_trot_drive` carries it today while its
  // unfixed siblings `stumble_trot`, `reach_to_target` and `counterbalance_standing`
  // still carry -0.250. The fix was simply never propagated.
  //
  // ★ CORROBORATION, independent: across n=30 genuinely-upright lean stages in the
  // 78-run corpus the median-of-run-max forward tilt is 6.3 deg, matching the
  // keyframe's built-in +6.30 deg to two decimals. That IS the observed fixed point.
  //
  // Target-side only -- no weight changes, no base-z change (with hipP -0.14 at base
  // z 1.020 the sole sits at z +0.0059 vs stand's +0.0056). Provably a NO-OP for every
  // keyframe whose chain already closes, which includes stand/stand_up/crouch/squat_*/
  // stumble_trot_drive/jump_launch -- so strategy 6 stays byte-identical even with this
  // numeric ON. It also levels straighten (-0.080) and forearm_brace_lean (-0.050);
  // those are deliberately in scope but have NOT been separately gated -- see the doc.
  // qpos leg layout: L hipY7 hipP8 hipR9 knee10 ankP11 ankR12 | R hipY13 hipP14 ...
  mjtNum posture_level_buf[64];
  {
    int pll_id = mj_name2id(model, mjOBJ_NUMERIC, "posture_leg_level");
    double pll = (pll_id >= 0)
        ? model->numeric_data[model->numeric_adr[pll_id]] : 0.0;
    if (pll > 0.5 && model->nq <= 64) {
      mju_copy(posture_level_buf, posture_target, model->nq);
      posture_level_buf[8]  = -(posture_level_buf[10] + posture_level_buf[11]);
      posture_level_buf[14] = -(posture_level_buf[16] + posture_level_buf[17]);
      posture_target = posture_level_buf;   // function scope: outlives every use below
    }
  }


  // ----- object position ----- //
  double const *object_pos = SensorByName(model, data, "object_pos");

  // ----- Determine which hand reaches and which braces ----- //
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  // torso_pos needed here for AUTO arm-side selection; also used below for
  // brace target computation. Declared once, reused in both places.
  double *torso_pos = SensorByName(model, data, "torso_position");

  // ---- Current heading frame (yaw-only) --------------------------------//
  // Forward = body +x rotated by the live base quaternion, flattened to the
  // horizontal plane and normalized; left = 90 deg CCW of forward. On the REAL
  // robot the base quaternion IS the live IMU orientation (h12_control_node.cc
  // fill_state) and NO yaw-zero calibration exists anywhere in the deploy chain
  // (only pitch/roll offset flags). Expressing the reach target + shoulder
  // anchor + auto hand-pick in THIS frame (rather than a fixed world axis) makes
  // them track WHERE THE ROBOT IS ACTUALLY FACING NOW, so IMU yaw drift no longer
  // swings the reach off to the side (user saw strat 16/21 point the arm to ~3
  // o'clock on hardware). At the home keyframe (yaw 0) forward=(1,0), left=(0,1),
  // so every use below is an EXACT identity -- sim/twin behavior is unchanged.
  double heading_fwd[3];
  {
    double xhat[3] = {1.0, 0.0, 0.0};
    mju_rotVecQuat(heading_fwd, xhat, data->qpos + 3);
    double fh = mju_sqrt(heading_fwd[0] * heading_fwd[0] +
                         heading_fwd[1] * heading_fwd[1]);
    if (fh > 1e-6) { heading_fwd[0] /= fh; heading_fwd[1] /= fh; }
    else { heading_fwd[0] = 1.0; heading_fwd[1] = 0.0; }  // degenerate: looking up
    heading_fwd[2] = 0.0;
  }
  const double heading_lft[2] = {-heading_fwd[1], heading_fwd[0]};
  // Re-express a WORLD-authored reach target so its horizontal bearing+distance
  // are measured from the CURRENT heading instead of the fixed world +x. Height
  // is left world-fixed (pitch/roll ARE IMU-calibrated; only yaw is broken).
  // Pivot = the keyframe base xy (posture_target) so at yaw 0 the result equals
  // the original world target EXACTLY, independent of any base drift.
  auto yaw_relative_target = [&](const double *world_target, double *out) {
    double dx = world_target[0] - posture_target[0];
    double dy = world_target[1] - posture_target[1];
    out[0] = posture_target[0] + dx * heading_fwd[0] + dy * heading_lft[0];
    out[1] = posture_target[1] + dx * heading_fwd[1] + dy * heading_lft[1];
    out[2] = world_target[2];
  };
  // Lateral position of a world point in the robot's frame (+ = robot's LEFT);
  // used for yaw-correct AUTO hand selection (pick the nearer hand) below.
  auto lateral_of = [&](const double *world_pt) {
    return (world_pt[0] - torso_pos[0]) * heading_lft[0] +
           (world_pt[1] - torso_pos[1]) * heading_lft[1];
  };

  // Auto-arm selection for the brace phase (the reach_to_target branch does its
  // own identical pick). reach_hand numeric:
  // 0 = AUTO (mocap target y < torso y -> right hand reaches), 1 = force LEFT,
  // 2 = force RIGHT. The OTHER arm always braces/counterweights.
  int rh_id_sel = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
  int rh_sel = (rh_id_sel >= 0)
      ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id_sel]])
      : 0;
  bool reach_right = (rh_sel == 2) ? true
                   : (rh_sel == 1) ? false
                   : (lateral_of(data->mocap_pos) < 0.0);  // target on robot's right
  double const *reaching_hand = reach_right ? right_hand_pos : left_hand_pos;
  double const *bracing_hand  = reach_right ? left_hand_pos  : right_hand_pos;

  // FOREARM BRACE (2026-07-01): in the forearm_brace_lean phase the brace is the
  // FOREARM (elbow_link pad), NOT the gripper hand. Repoint the brace POSITION
  // target to the forearm pad site so Brace-Pos pulls the forearm (not the hand)
  // onto the table; the forearm is ~0.5 m from the shoulder and CAN reach the
  // 0.77 m table in a deep bow, whereas the hand-brace would land the gripper.
  // Name-gated to forearm_brace_lean; reach_to_target keeps the hand. The gripper
  // is excluded from the table in the model so the pad is the only brace contact.
  // (is_forearm_brace defined at the top of Residual for the any_arm_contact gate.)
  double forearm_site_pos[3];
  if (is_forearm_brace) {
    int fs = mj_name2id(model, mjOBJ_SITE,
                        reach_right ? "left_forearm_brace" : "right_forearm_brace");
    if (fs >= 0) {
      mju_copy3(forearm_site_pos, data->site_xpos + 3 * fs);
      bracing_hand = forearm_site_pos;
    }
  }

  //------------- Reward calculation --------------//
  double const hand_dist_penalty = 1.0;
  double const brace_reward = 0.5;
  double const success = 1000;
  // double const retrieve_reward = 1000;

  // ----- reaching hand position ----- //
  double hand_dist = mju_dist3(reaching_hand, object_pos);

  // ----- Contact forces ----- //
  double *left_contact = SensorByName(model, data, "left_hand_contact");
  double *right_contact = SensorByName(model, data, "right_hand_contact");
  double brace_contact_force = reach_right ? left_contact[0] : right_contact[0];
  double reach_contact_force = reach_right ? right_contact[0] : left_contact[0];
  // FOREARM BRACE: the brace press is the SUM of every table<->bracing-arm contact,
  // read straight from mjData (see TableBraceForce above). It used to read the
  // forearm touch sensor, but a touch sensor only sees contacts on its own site's
  // body -- so once the arm was made solid and the load moved onto the WRIST links,
  // the sensor reported 0.0 N through an entire braced hold while ~38 N was actually
  // flowing, and every cost gated on this value was dead. reach_right => the LEFT
  // arm is the bracing one.
  if (is_forearm_brace) {
    brace_contact_force = TableBraceForce(model, data, /*brace_left=*/reach_right);
  }

  double reward = 0;

  // Bracing position calculation. Reverted Y-clamp (was test 14) → back
  // to bracing_hand[1] (test 12 state). User confirmed test 14 introduced
  // chaotic early-phase behaviour. Y free means no restoring force on
  // lateral position; the eventual ~60s slip seen in test 12 is the
  // known trade-off for accepting this baseline.
  double const *table_pos = SensorByName(model, data, "table_surface_pos");
  // ★ 2026-07-28 `table_surface_pos` is a framepos on the table_top GEOM = its CENTRE
  // (z 0.810), NOT the physical face (z 0.865). THREE separate targets were built on it
  // and all three are therefore 55 mm too low:
  //   * ideal_brace.z  (Brace Pos w120)  -> aims 115 mm INSIDE the wood at depth 0.06
  //   * brace_air_target.z               -> "30 cm above the surface" is really 245 mm
  //   * palm_height_error                -> targets the palm 55 mm INSIDE the wood
  // Same frame confusion `brace_gate_fix` repaired on 2026-07-25, in three places it
  // never touched. `table_face_z` resolves it ONCE, gated by `brace_target_face`
  // (0 = OFF = byte-identical, so every historic verdict stays reproducible).
  // Orientation-safe: support of the box half-extents along WORLD z.
  double table_face_z = table_pos[2];
  {
    int btf_id0 = mj_name2id(model, mjOBJ_NUMERIC, "brace_target_face");
    double btf0 = (btf_id0 >= 0)
        ? model->numeric_data[model->numeric_adr[btf_id0]] : 0.0;
    if (btf0 > 0.5) {
      int tg = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
      if (tg < 0) tg = mj_name2id(model, mjOBJ_GEOM, "table_top");
      if (tg >= 0) {
        const double *Rg = data->geom_xmat + 9 * tg;   // row-major; row 2 = world z
        const double *sg = model->geom_size + 3 * tg;
        table_face_z = data->geom_xpos[3 * tg + 2] +
                       mju_abs(Rg[6]) * sg[0] + mju_abs(Rg[7]) * sg[1] +
                       mju_abs(Rg[8]) * sg[2];
      }
    }
  }

  // ---- SLAB NEAR EDGE, world x (2026-07-29) ---------------------------------
  // Companion to table_face_z above, same orientation-safe support trick but along
  // WORLD X, giving the edge of the slab NEAREST the robot. Used by the
  // `brace_target_slab` gate to aim the brace at a point that is actually ON the
  // surface. NaN-safe sentinel: stays -inf when the gate is off or the geom is
  // missing, and every consumer checks for that.
  double table_near_edge_x = -std::numeric_limits<double>::infinity();
  {
    int tg = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
    if (tg < 0) tg = mj_name2id(model, mjOBJ_GEOM, "table_top");
    if (tg >= 0) {
      const double *Rg = data->geom_xmat + 9 * tg;   // row 0 = world x
      const double *sg = model->geom_size + 3 * tg;
      double half_x = mju_abs(Rg[0]) * sg[0] + mju_abs(Rg[1]) * sg[1] +
                      mju_abs(Rg[2]) * sg[2];
      table_near_edge_x = data->geom_xpos[3 * tg] - half_x;
    }
  }
  // torso_pos declared above (near arm-selection) to keep one definition.

  // ----- Reach target ----------------------------------------------------//
  // The reach residual targets the actual `object_pos` (set by mocap to wherever
  // the task object is). The workspace clamp and the forearm-brace branch below
  // may retarget it into `reach_target_storage`.
  double reach_target_storage[3];
  double const *reach_target = object_pos;
  // FOREARM BRACE (2026-07-01): keep the reaching hand IN THE AIR reaching TOWARD
  // the object, not resting on the table (user: "right arm was doing some bracing
  // too... I don't want it to touch it"). The object sits ON the table (z ~=
  // surface), so an unclamped reach drags the hand straight down to the surface.
  // Point the reach at the object's x,y but hold it at a comfortable airborne
  // height ABOVE the table so the hand hovers / reaches out WITHOUT touching down.
  double brace_air_target[3];
  if (is_forearm_brace) {
    brace_air_target[0] = object_pos[0];
    brace_air_target[1] = object_pos[1];
    brace_air_target[2] = table_face_z + 0.30;  // ~30 cm above the table FACE
                                               // (was the geom CENTRE = 245 mm)
    reach_target = brace_air_target;
  }
  // reach_to_target (Strategy 21): the standalone reach primitive. The target is
  // the EXTERNAL mocap object (object_pos = target_position_, set from a model
  // numeric or, in deploy, a vision/nav stack — see TransitionLocked). Two
  // choices make this a BALANCED reach rather than a lean:
  //   1. Auto-pick the nearer hand (by target y vs torso y) so a target on
  //      either side is reached with the natural arm and no body twist.
  //   2. SPHERICAL workspace clamp: project the target onto the arm-reach sphere
  //      centred at the reaching shoulder. The H1-2 arm rests nearly fully
  //      extended (hand ~0.52 m from the shoulder), so a target even slightly
  //      past that radius made the planner PITCH THE TORSO FORWARD to gain reach
  //      (forward tilt + knee strut) — a lean, not a reach. Projecting onto the
  //      sphere lets the hand extend MAXIMALLY toward an out-of-reach point while
  //      the body stays upright (in-reach => hand on target; out-of-reach => hand
  //      at full extension, still standing). Beyond that envelope is LEAN's / a
  //      step's job — the reach/lean/step hierarchy. A box clamp can't do this:
  //      its forward corner sits outside the arm sphere and still induces the
  //      lean. Pelvis Tilt (JSON) holds the torso upright so the pull extends
  //      the ARM, not the torso.
  else if (residual_keyframe_.name == "reach_to_target") {
    // Input target = the mocap "target" body (data->mocap_pos[0]), set from
    // target_position_ in TransitionLocked (the reach_target numeric, or a
    // vision/nav stack writing mocap_pos at runtime). NOTE: object_pos tracks a
    // SEPARATE, STATIC table "object" body — reading it froze every reach at the
    // table point — so the reach primitive reads the mocap target directly.
    // Yaw-relative ONLY for the PRE-LEAN single-phase strategy (strat 21): its
    // mocap target is a fixed AUTHORED "forward" point, so re-express its bearing
    // from the current heading to beat IMU yaw drift (user saw ~3 o'clock on
    // hardware). Exact identity at yaw 0. In the MULTI-PHASE lean pipeline (strat
    // 32) the vision/nav stack writes the real detected object ALREADY in the
    // planner world frame, so leave it as-is (rotating it would double-rotate).
    double in_target_storage[3];
    if (num_phases_ <= 1)
      yaw_relative_target(data->mocap_pos, in_target_storage);
    else
      mju_copy3(in_target_storage, data->mocap_pos);
    double const *in_target = in_target_storage;
    // Hand selection via the `reach_hand` numeric: 0/absent = AUTO (the nearer
    // hand — target on the robot's right, y below torso, picks the right hand),
    // 1 = force LEFT, 2 = force RIGHT. Forcing a hand lets a future lean/grasp
    // pipeline keep the other arm free (e.g. brace with the left, reach with the
    // right) instead of letting target geometry decide.
    int rh_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
    int rh_mode = (rh_id >= 0)
        ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id]])
        : 0;
    bool reach_right_reach = (rh_mode == 2) ? true
                     : (rh_mode == 1) ? false
                     : (lateral_of(in_target) < 0.0);   // 0/auto (robot's right)
    reaching_hand = reach_right_reach ? right_hand_pos : left_hand_pos;
    // Shoulder anchor = torso_position + FK-measured offset (MJPC frame):
    // (+0.000, +-0.148, +0.219); rest |shoulder->hand| = 0.524.
    // Shoulder anchor laterally offset along the robot's LEFT axis (yaw-correct),
    // not fixed world y: -0.148 = right shoulder, +0.148 = left. Identity at yaw 0.
    double shoulder_lat = reach_right_reach ? -0.148 : 0.148;
    double shoulder[3] = {
        torso_pos[0] + shoulder_lat * heading_lft[0],
        torso_pos[1] + shoulder_lat * heading_lft[1],
        torso_pos[2] + 0.219};
    double v[3];
    mju_sub3(v, in_target, shoulder);
    double r = mju_norm3(v);
    // R=0.50 keeps the elbow a touch bent (rest reach 0.524) so the arm never
    // locks straight into a strut at full extension.
    // reach_radius / reach_drop: LIVE-tunable so the operator dials the reach into
    // the magpie arm's HOLDABLE workspace on the REAL robot (the only faithful
    // verdict). The old 0.50/0.28 put the ~1.8 kg arm at ~0.42 m horizontal ->
    // unholdable -> the arm THRASHED (RMSE 50deg, 97% torque) and swung BACKWARD.
    // Smaller radius + bigger drop = a forward-DOWN, gravity-light pose it can HOLD.
    int rr_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_radius");
    double kReachRadius = (rr_id >= 0)
        ? model->numeric_data[model->numeric_adr[rr_id]] : 0.46;
    if (r > kReachRadius && r > 1e-6) {
      mju_scl3(v, v, kReachRadius / r);          // project onto the sphere
      // GRAVITY GUARD (magpie arms, 2026-06-22). A FAR target (large horizontal
      // distance, modest vertical drop) projects to a NEAR-HORIZONTAL arm: the
      // sphere point sits only ~0.20 m below the shoulder. The gripper-laden
      // magpie forearm cannot HOLD that extension — the elbow hits its torque
      // limit and the arm collapses/swings BACKWARD on the real/twin plant
      // (the agent_server own-sim, with a gentler PD and no gripper-mass stress,
      // holds it and hides the failure — the documented own-sim-over-holds gap).
      // So cap the lift: never command the hand higher than ~34 deg below the
      // shoulder, and rescale the horizontal component to keep the hand on the
      // reach sphere. The arm still MAX-EXTENDS toward the target's bearing, but
      // forward-DOWN (gravity-light, near its natural rest droop) instead of
      // straight out. Deeper reach toward an out-of-reach point is the LEAN's
      // job of the brace phase — the reach/lean/step hierarchy is unchanged.
      int md_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_drop");
      double kMaxLift = (md_id >= 0)
          ? model->numeric_data[model->numeric_adr[md_id]] : 0.36;  // z drop (m); bigger = arm more DOWN = more holdable
      if (v[2] > -kMaxLift) {
        v[2] = -kMaxLift;
        double horiz = mju_sqrt(
            mju_max(1e-9, kReachRadius * kReachRadius - kMaxLift * kMaxLift));
        double rh = mju_sqrt(v[0] * v[0] + v[1] * v[1]);
        if (rh > 1e-6) { v[0] *= horiz / rh; v[1] *= horiz / rh; }
      }
      mju_add3(reach_target_storage, shoulder, v);
    } else {
      mju_copy3(reach_target_storage, in_target);  // already in reach
    }
    reach_target = reach_target_storage;
  }

  double torso_to_table_x = table_pos[0] - torso_pos[0];
  // 2026-05-20: pin the brace LATERALLY under the right shoulder instead of
  // leaving Y free. FK rollout showed the brace hand drifting ~22 cm inboard
  // toward centerline (hand y -0.107 while the right shoulder joint sits at
  // torso_y -0.209) — an inboard brace gives almost no lateral support, so the
  // {L_foot, R_hand} base is narrow and the body tips sideways (user-reported).
  // Targeting torso_y - 0.24 places the palm just to the right of the shoulder
  // joint → a near-vertical force path (plank-like) and a WIDE lateral base
  // coupled with the left foot. The phase_brace_pos_scale ramp still gates this
  // to the brace phases (it's 0 in stand/extend), so non-brace phases are
  // unaffected. Pairs with the contact local_pos2 y = -0.24 in the strategy
  // JSONs (hand + elbow share the same Y → forearm parallel to x, not slanted).
  // ★ 2026-07-27 FRAME FIX — `brace_target_face` (0 = OFF = BYTE-IDENTICAL, 1 = ON).
  // `table_surface_pos` is a framepos on the table_top GEOM, i.e. its CENTRE (z 0.810),
  // NOT the physical face (z 0.865). Same frame confusion that killed the brace gate
  // (fixed 2026-07-25 by `brace_gate_fix`) — it was never fixed HERE. Consequence:
  //   brace_press_depth 0.06 -> ideal_brace.z 0.750 = 115 mm INSIDE the wood
  //   brace_press_depth 0.00 -> ideal_brace.z 0.810 =  55 mm INSIDE the wood
  // so the knob CANNOT express "press at the surface" — its whole range is below the
  // slab. That is why lowering it SATURATED (34 deg) instead of shallowing the bow, and
  // it means the 2026-07-27 refutation of `brace_press_depth` was a refutation of a
  // control that could not reach the value it needed.
  // At `Brace Pos` weight 120 this is a permanent, UN-NULLABLE downward demand: the pad
  // is stopped by the wood at the face, the residual never reaches zero, and the body
  // keeps bowing to chase it until the PELVIS lands on the slab. Measured (brace22_1,
  // direct 6->22): pad seated 79.2 % of frames — the best seating ever recorded — with
  // pelvis 486 N MEDIAN and torso 406 N median, while the only ALLOWED contact, the
  // left forearm pad, carried 50 N. See [[feedback_audit_the_load_path_not_the_pose]].
  // With this ON the depth is measured from the FACE: 0.06 = 60 mm below the surface,
  // 0.0 = exactly at it. Orientation-safe (support along world z), so a tilted or
  // re-placed slab stays correct.
  double brace_press_z =
      table_face_z - GetNumberOrDefault(0.06, model, "brace_press_depth");
  // ---- brace_target_slab (2026-07-29): aim the brace at a point ON THE SLAB ----
  // The legacy x below is `torso + 0.4 * (table - torso)` -- 40 % of the way to the
  // table CENTRE, i.e. deliberately SHORT of the near edge -- and brace_press_z is
  // 20-60 mm BELOW the face. That target sits INSIDE the slab's front face, in front
  // of the surface. It is reachable ONLY in the planner's own sim, which <exclude>s
  // the table from the whole arm chain and lets the arm pass THROUGH the wood while
  // an inert pad reports newtons. On the twin the arm is SOLID, so the residual can
  // never null: the body keeps bowing to chase the buried point until the PELVIS
  // lands on the slab. That is the "drape" -- measured pelvis 486 N / torso 406 N
  // while the only ALLOWED contact, the forearm pad, carried 50 N. It was never a
  // balance failure; it is the same unattainable-target disease as every other lean
  // stage, and it is also why the 07-13 own-sim video looked correct.
  // ON: x = slab NEAR EDGE + `brace_target_inset` (default 0.07), so the pad lands on
  // the surface. ★ inset must exceed the pad capsule's rear-end offset (~46 mm behind
  // its centre) or the rear end hangs over thin air -- lean.xml's header records that
  // 28 mm of overhang alone was enough to take seating to 0 %.
  // Pair this with a SHALLOW `brace_press_depth` (~0.008): the deep 0.06 was tuned in
  // the phantom-arm world where nothing stopped the pad, but on a solid slab the wood
  // does the arresting, so a few mm is enough to build the Brace-Force gradient.
  // 0 = OFF = BYTE-IDENTICAL (the legacy expression below is used verbatim).
  double brace_x_target = torso_pos[0] + 0.4 * torso_to_table_x;
  {
    int bts_id = mj_name2id(model, mjOBJ_NUMERIC, "brace_target_slab");
    double bts = (bts_id >= 0) ? model->numeric_data[model->numeric_adr[bts_id]] : 0.0;
    if (bts > 0.5 && std::isfinite(table_near_edge_x)) {
      brace_x_target = table_near_edge_x +
                       GetNumberOrDefault(0.07, model, "brace_target_inset");
    }
  }
  double ideal_brace[3] = {
      brace_x_target,  // legacy: partway to the table centre; gated: on the slab
      // bracing arm = the OTHER arm (reach_right -> left arm braces, so +0.24).
      torso_pos[1] + (reach_right ? 0.24 : -0.24),
      // 2026-05-22: press TARGET 6 cm BELOW the surface (was -0.02). Under the
      // real-robot (doc) ROM the bracing forearm stalled ~7 cm ABOVE the table:
      // with the target only 2 cm under the surface the downward Brace-Pos pull
      // faded before contact, so no contact + no brace force formed and the lean
      // tipped. A deeper press target sustains the pull through to firm forearm
      // contact (the table collision arrests the hand at the surface and converts
      // the residual press into the brace force the Brace-Force cost rewards).
      // 2026-07-01: KEEP the deep below-surface drive for the forearm brace too --
      // it is what bows the body far enough that the pad nears the table (raising
      // the target slackens the pull -> the body bows LESS -> the pad hovers MORE,
      // verified on the 0.87 probe). The proximity gate that engages load-transfer
      // + force is decoupled below (keyed on the pad's height above the PHYSICAL
      // surface, not this deliberately-unreachable drive target).
      // ★ 2026-07-27: DEPTH IS NOW A NUMERIC. `brace_press_depth` (default 0.06 =
      // BYTE-IDENTICAL) sets how far BELOW the physical face this drive target sits.
      // The 0.06 was chosen (2026-05-22) because "the downward Brace-Pos pull faded
      // before contact" -- but that was measured while `brace_force_prox_gate` was
      // DEAD (it read 0.000 even with the pad perfectly seated; fixed 2026-07-25 by
      // `brace_gate_fix`), so Brace Pos was the ONLY thing pulling the pad down and
      // needed a deep, unreachable target to keep pulling. With the gate restored,
      // Brace Force (w60, ~2232/step uncontacted) supplies that pull directly.
      // MEASURED COST of the deep sink: the pad target sits 115 mm inside the wood,
      // so at w120 the residual never nulls and the body bows to chase it -- the
      // keyframe commands +2.9 deg of pelvis pitch and the plant reaches 42-44 deg.
      // That bow is what drops the RIGHT arm onto the table (right_forearm_pad
      // 103-182 N peak, right_shoulder_yaw 107-146 N) and pushes the CoM to +0.4.
      // Lower this to shallow the bow; the 2026-07-01 note that raising it makes the
      // pad hover MORE was true only under the dead gate.
      // ★ 2026-07-27: now computed above as `brace_press_z` so it can be referenced to
      // the physical FACE instead of the geom centre (`brace_target_face`).
      brace_press_z
  };

  double penalty_hand = hand_dist_penalty * hand_dist;
  double brace_dist = mju_dist3(bracing_hand, ideal_brace);
  double reward_brace = brace_reward * mju_exp(-2.0 * brace_dist);
  double reward_success = (hand_dist < kHandDistThreshold && reach_contact_force > kContactForceThreshold) ? success : 0;
  
  reward = -penalty_hand + reward_brace + reward_success;

  //--------------- End of reward calculation -----------------//

  residual[counter++] = success - reward;

  // -------------- Below are additional residuals -------------- //

  // ----- Height: head feet vertical error ----- //
  // Note: Reduced importance vs push task since leaning lowers head

  // feet sensor positions
  double *foot_right_pos = SensorByName(model, data, "foot_right_pos");
  double *foot_left_pos = SensorByName(model, data, "foot_left_pos");

  double *head_position = SensorByName(model, data, "head_position");
  double head_feet_error =
      head_position[2] - 0.5 * (foot_right_pos[2] + foot_left_pos[2]);
  // TEST #8 (2026-05-17): Restore Height ×0.35 during arm contact
  // (test 4's working setting). Combined with the foot-z anchors added
  // in test 6 (Right Foot Lift penalises lift; Left Leg Anchor enforces
  // both left knee height AND left foot on ground), the squat is the
  // stable solution but now with feet anchored. Tests 6 (×0.7) and 7
  // (×1.0) both failed because the planner can't stand-and-bend with
  // kp_ankle=20 — it either tips forward (×0.7) or backward (×1.0).
  // The squat IS the natural solution given the weak PD.
  double height_scale = any_arm_contact ? 0.35 : 1.0;
  residual[counter++] = height_scale * (head_feet_error - height_goal);

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- CoM xy velocity tracking ----- //
  residual[counter + 0] = com_velocity[0];
  residual[counter + 1] = com_velocity[1];
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  counter += model->nu;

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  // Horizon kept at 0.3 s. A 0.45 s preview was trialed (2026-05-26) to fight the
  // forward-velocity overshoot at brace commit, but a 10-run trace showed it did
  // NOT improve the hold rate (still 8/10) and produced lower-quality holds (one
  // barely-leaning +2 deg, one drifting -8 deg lateral), so it was reverted.
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // CoM fore-aft sim2real correction (2026-06-13). The REAL robot's CoM sits ahead of the
  // model's (un-sysid'd mass), so a model-centered CoM leaves the REAL CoM forward -> the
  // robot leans forward and the ankles range-limit trying to pull it back. `com_x_offset`
  // (model numeric, meters, default 0) biases this balance target: it tells the planner the
  // CoM is this much further FORWARD (world +x = robot-forward for the facing-+x stand), so
  // the planner holds the actual CoM that much BACK. RAISE it if the real robot leans forward;
  // the value that makes it stand upright == the real CoM forward offset (doubles as a sysid
  // measurement). XML-tunable -> no rebuild to change the value; default 0 = exact prior behavior.
  // 2026-07-20: com_x_offset now rides the SUPPORT frame, not world +x. It was
  // written when the stand faced +x and was harmless while shipped at 0.0, but
  // it goes LIVE this session and real headings ran -41..+15 deg tonight -- a
  // world-frame bias would be aimed that far off the robot's actual forward.
  // Same midFeetZUp rule as the trim below / the 2026-07-16 balance_frame fix;
  // byte-identical at heading 0 and while the numeric is 0.
  // (reach_com_back just below is still world-frame -- it is reach-gated, was
  // tuned that way on strat 21, and is out of scope for the stand.)
  {
    int com_off_id = mj_name2id(model, mjOBJ_NUMERIC, "com_x_offset");
    double com_fwd_bias =
        (com_off_id >= 0) ? model->numeric_data[model->numeric_adr[com_off_id]] : 0.0;
    if (com_fwd_bias != 0.0) {
      double fwd[2] = {1.0, 0.0};
      double const *flf = SensorByName(model, data, "foot_left_forward");
      double const *frf = SensorByName(model, data, "foot_right_forward");
      if (flf && frf) {
        double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
        double len = mju_sqrt(fx * fx + fy * fy);
        if (len > 1.0e-6) { fwd[0] = fx / len; fwd[1] = fy / len; }
      }
      capture_point[0] += com_fwd_bias * fwd[0];
      capture_point[1] += com_fwd_bias * fwd[1];
    }
    // reach_com_back (strat 21, 2026-06-23): the REACH adds its own forward-CoM
    // creep ON TOP of the global com_x_offset gap -- the ~4 kg magpie arm reaching
    // forward pulls the CoM toward the ankle's forward limit, and on the REAL chain
    // (12-thread async + un-modeled real CoM) the robot slowly leans forward to a
    // near-topple (the in-process lockstep twin CANNOT show this -- it holds flat).
    // So during a reach, bias the balance target an EXTRA reach_com_back metres
    // forward -> the planner holds the actual CoM that much further BACK (a
    // counterbalance for the forward arm). REACH-GATED: stand/crouch/all other
    // strategies are byte-identical (this only fires for the reach_to_target
    // keyframe). TUNE ON REAL: raise if the reach still leans forward, lower if it
    // leans BACK; the value that stands it upright == the reach's real CoM offset.
    // NOTE: this makes the in-process twin lean BACKWARD (the twin has no real gap),
    // so validate on the twin with this at 0, and
    // dial this on the real robot.
    if (residual_keyframe_.name == "reach_to_target") {
      int rcb_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_com_back");
      if (rcb_id >= 0)
        capture_point[0] += model->numeric_data[model->numeric_adr[rcb_id]];
    }
  }

  // T1 REFERENCE TRIM v2 (ported 2026-07-20; see the s_trim_x file-scope note).
  // Applied along the SUPPORT frame -- the feet's own mean heading, the same
  // midFeetZUp rule as the 2026-07-16 balance_frame fix -- NOT world +x. This
  // matters on real: heading ran -41 deg .. +11 deg across the 07-20 runs, so a
  // world-frame correction would be aimed tens of degrees off the robot's
  // actual forward. (The com_x_offset block above is still world-frame; it is
  // shipped at 0.0 so nothing is live, but it carries the same latent defect.)
  //
  // Semantics mirror com_x_offset: a POSITIVE trim tells the planner the
  // capture point is further forward/left than measured, so the planner holds
  // the real CoM back/right. Both trims are exactly 0.0 unless stand_trim_tau
  // > 0, and the whole block is skipped in that case -> byte-identical.
  // com_y_offset rides the same support-frame axes: the static lateral sibling
  // of com_x_offset, for a sided park that survives an honest roll calibration
  // (lateral mass-model asymmetry, e.g. the one-sided magpie arm). Shipped 0.
  double com_y_bias = 0.0;
  {
    int cy_id = mj_name2id(model, mjOBJ_NUMERIC, "com_y_offset");
    if (cy_id >= 0) com_y_bias = model->numeric_data[model->numeric_adr[cy_id]];
  }
  // STATIC support-frame FORWARD bias (2026-07-22): the fore-aft sibling of
  // com_y_offset. MEASURED need on real (ankleidk npz): ZMP sits +3.5 cm ahead
  // of the model CoM at quiet stance = unmodeled forward mass the planner never
  // compensates -> chronic forward park -> tips. POSITIVE = planner holds the
  // real CoM BACK by this much (same semantics as trim_x). Support-frame, so it
  // stays aimed at the robot's true forward at any heading (runs today sat at
  // +20..+35 deg where the world-frame com_x_offset would be aimed wrong).
  // The leaky trim rides ON TOP for the residual/day-to-day part.
  double com_x_bias = 0.0;
  {
    int cx_id = mj_name2id(model, mjOBJ_NUMERIC, "com_x_offset_support");
    if (cx_id >= 0) com_x_bias = model->numeric_data[model->numeric_adr[cx_id]];
  }
  if (s_trim_x != 0.0 || s_trim_y != 0.0 || com_y_bias != 0.0 ||
      com_x_bias != 0.0) {
    double fwd[2] = {1.0, 0.0}, lat[2] = {0.0, 1.0};
    double const *flf = SensorByName(model, data, "foot_left_forward");
    double const *frf = SensorByName(model, data, "foot_right_forward");
    if (flf && frf) {
      double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
      double len = mju_sqrt(fx * fx + fy * fy);
      if (len > 1.0e-6) {
        fwd[0] = fx / len; fwd[1] = fy / len;
        lat[0] = -fwd[1];  lat[1] = fwd[0];
      }
    }
    double lat_bias = s_trim_y + com_y_bias;
    double fwd_bias = s_trim_x + com_x_bias;
    capture_point[0] += fwd_bias * fwd[0] + lat_bias * lat[0];
    capture_point[1] += fwd_bias * fwd[1] + lat_bias * lat[1];
  }

  // project onto support polygon
  //
  // Phase-aware support polygon for the Balance residual. When the bracing
  // hand is on the table the polygon EXPANDS to include the hand contact —
  // {L_foot, R_foot, R_hand} triangle in both-feet phases, {L_foot, R_hand}
  // line in leg-lift phases. Inside the polygon → pcp = capture_point →
  // residual = 0 (body is supported). Outside → pcp = nearest perimeter
  // point. This is the WBC support-polygon idea: forward lean is free as
  // long as the capture point stays in the convex hull of active contacts.
  //
  // Without this expansion, Balance projects onto the foot-foot line only,
  // and a body advancing forward to lean sees a huge excursion (28cm at a
  // typical lean pose) amplified 10× by edge_amplifier — Balance fights
  // the lean and the planner can't find a stable forward pose.
  //
  // ITER 40 (2026-05-18) left a leg-lift special case (single L_foot
  // projection). That's preserved here for the no-arm-contact sub-case;
  // with arm contact during leg-lift we use the {L_foot, R_hand} line.
  double pcp[3];
  if (any_arm_contact) {
    // L_foot + R_foot + R_hand triangle, with a LOAD-LIMITED hand vertex.
    //
    // MARGIN FIX (2026-05-26): the hand vertex used bracing_hand UNCONDITIONALLY
    // the instant a phase merely DECLARED an arm contact. That told Balance the
    // CoM was supported all the way out to the hand (x≈0.62) BEFORE the hand was
    // pressing, so the planner advanced the CoM forward into support that wasn't
    // bearing load yet and pitched past the brace — a multi-run GUI-cadence trace
    // measured a ~60% FORWARD fall at the brace-commit instant (~13.3 s).
    // Fix mirrors the proven leg-lift load_frac logic above: step the third
    // vertex from the midfoot toward the real hand only by MEASURED brace force /
    // body weight. force≈0 → vertex≈midfoot → triangle collapses to the foot
    // line → CoM must stay over the feet (no forward over-commit → no faceplant);
    // as force builds → vertex extends → forward lean opens up exactly as far as
    // the load justifies. The arm still REACHES the table independently (Reaching
    // Hand Dist / Object Dist), so the hand can touch and start pressing with the
    // CoM back — then this gate opens and the CoM advances into a real brace.
    // No floor here (unlike leg-lift): the whole point is zero forward credit
    // until the hand actually presses. divisor 140 N ≈ a solid brace, cap 0.9.
    //
    // ★ FOOT-EXTENT FIX (2026-08-02). The two foot vertices used
    // `foot_{left,right}_pos`, which is
    //     <framepos objtype="body" objname="left_ankle_roll_link"/>
    // -- the ANKLE JOINT ORIGIN, not the foot. The load-bearing sole runs
    // -0.0835 m (heel) to +0.1348 m (toe) along the foot's forward axis from
    // that origin (mesh vertices within 2 mm of the sole plane, both feet,
    // `stand` keyframe), so the polygon was throwing away 134 mm of real
    // forward support and 84 mm of rear support. The planner believed its
    // support ended AT THE ANKLE and pinned the CoM there.
    //
    // Measured consequence before the fix: the brace keyframe commanded
    // CoM +0.142 m (ankle frame) and the controller held +0.036 -- 25%
    // tracking, sitting on the phantom ankle edge with the 10x edge
    // amplifier holding it. Worse, combined with the load gate below, at a
    // measured 45-60 N the third vertex landed at +0.11, i.e. the "expanded"
    // braced polygon was NARROWER THAN THE BARE FEET -- Balance actively
    // pulled the CoM backward. That is the backward-squat/backward-fall
    // signature (falls 10/20 at t~7.4 s, CoM never leaving the heel half).
    //
    // Same bug class as the 2026-07-25 `table_surface_pos` defect (framepos
    // on the GEOM CENTRE, not the face) that had the brace gate reading
    // 0.000 while perfectly seated. Ask what a framepos is actually ON.
    constexpr double kSoleHeel = -0.079;   // m, along foot forward axis
    constexpr double kSoleToe  = +0.133;   // m (measured +0.1348, held back)
    double flf_x = 1.0, flf_y = 0.0, frf_x = 1.0, frf_y = 0.0;
    {
      double const *flf = SensorByName(model, data, "foot_left_forward");
      double const *frf = SensorByName(model, data, "foot_right_forward");
      if (flf) {
        double n = mju_sqrt(flf[0]*flf[0] + flf[1]*flf[1]);
        if (n > 1.0e-6) { flf_x = flf[0]/n; flf_y = flf[1]/n; }
      }
      if (frf) {
        double n = mju_sqrt(frf[0]*frf[0] + frf[1]*frf[1]);
        if (n > 1.0e-6) { frf_x = frf[0]/n; frf_y = frf[1]/n; }
      }
    }
    // Four sole corners along each foot's own forward axis, so a yawed or
    // staggered stance stays correct (the 2026-07-16 balance_frame lesson).
    double fvx[4] = {foot_left_pos[0]  + kSoleToe  * flf_x,
                     foot_right_pos[0] + kSoleToe  * frf_x,
                     foot_right_pos[0] + kSoleHeel * frf_x,
                     foot_left_pos[0]  + kSoleHeel * flf_x};
    double fvy[4] = {foot_left_pos[1]  + kSoleToe  * flf_y,
                     foot_right_pos[1] + kSoleToe  * frf_y,
                     foot_right_pos[1] + kSoleHeel * frf_y,
                     foot_left_pos[1]  + kSoleHeel * flf_y};
    // Load-gated brace vertex, unchanged in spirit: it grows from the foot
    // centroid toward the real contact only as fast as MEASURED brace force
    // justifies, so there is still zero forward credit before the arm
    // presses. It now starts INSIDE the true foot polygon rather than at a
    // point 134 mm behind the toes, so frac=0 degenerates to exactly the
    // feet instead of to something smaller than the feet.
    double midfoot_x = 0.25 * (fvx[0] + fvx[1] + fvx[2] + fvx[3]);
    double midfoot_y = 0.25 * (fvy[0] + fvy[1] + fvy[2] + fvy[3]);
    double hand_load_frac = mju_min(0.9, brace_contact_force / 140.0);
    double hand_vert_x = midfoot_x + hand_load_frac * (bracing_hand[0] - midfoot_x);
    double hand_vert_y = midfoot_y + hand_load_frac * (bracing_hand[1] - midfoot_y);
    // ⚠ MUST BE A REAL CONVEX HULL, NOT AN ANGLE SORT. The brace vertex STARTS
    // AT THE FOOT CENTROID (hand_load_frac = 0 before the arm presses) and stays
    // inside the foot quad until the force is large enough to push it past the
    // toe line. Angle-sorting a set containing an INTERIOR point produces a
    // non-convex star: the interior vertex is spliced between two hull vertices
    // and carves a dent, so the convex "all left turns" test below reports
    // OUTSIDE for capture points that are genuinely supported. At exactly
    // frac = 0 the vertex IS the centroid and atan2(0,0) = 0, so the ordering is
    // degenerate outright. Andrew's monotone chain drops interior points instead.
    double px_in[5] = {fvx[0], fvx[1], fvx[2], fvx[3], hand_vert_x};
    double py_in[5] = {fvy[0], fvy[1], fvy[2], fvy[3], hand_vert_y};
    // sort lexicographically by (x, y)
    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4 - i; j++) {
        if (px_in[j] > px_in[j+1] ||
            (px_in[j] == px_in[j+1] && py_in[j] > py_in[j+1])) {
          double t = px_in[j]; px_in[j] = px_in[j+1]; px_in[j+1] = t;
          t = py_in[j]; py_in[j] = py_in[j+1]; py_in[j+1] = t;
        }
      }
    }
    double hx[12], hy[12];
    int nh = 0;
    auto cross_ok = [&](double ax, double ay, double bx, double by,
                        double cx2, double cy2) {
      return (bx - ax) * (cy2 - ay) - (by - ay) * (cx2 - ax) <= 1.0e-12;
    };
    for (int i = 0; i < 5; i++) {                       // lower hull
      while (nh >= 2 && cross_ok(hx[nh-2], hy[nh-2], hx[nh-1], hy[nh-1],
                                 px_in[i], py_in[i])) nh--;
      hx[nh] = px_in[i]; hy[nh] = py_in[i]; nh++;
    }
    for (int i = 3, lower = nh + 1; i >= 0; i--) {      // upper hull
      while (nh >= lower && cross_ok(hx[nh-2], hy[nh-2], hx[nh-1], hy[nh-1],
                                     px_in[i], py_in[i])) nh--;
      hx[nh] = px_in[i]; hy[nh] = py_in[i]; nh++;
    }
    int kNV = nh - 1;                                   // last == first
    if (kNV < 3) kNV = nh;                              // degenerate: keep what we have
    double vx[12], vy[12];
    for (int i = 0; i < kNV; i++) { vx[i] = hx[i]; vy[i] = hy[i]; }
    // monotone chain emits CW for a y-up frame; the test below wants CCW.
    {
      double area2 = 0.0;
      for (int i = 0; i < kNV; i++) {
        int j = (i + 1) % kNV;
        area2 += vx[i] * vy[j] - vx[j] * vy[i];
      }
      if (area2 < 0.0) {
        for (int i = 0; i < kNV / 2; i++) {
          double t = vx[i]; vx[i] = vx[kNV-1-i]; vx[kNV-1-i] = t;
          t = vy[i]; vy[i] = vy[kNV-1-i]; vy[kNV-1-i] = t;
        }
      }
    }
    double px = capture_point[0], py = capture_point[1];
    bool inside = true;
    for (int i = 0; i < kNV; i++) {
      int j = (i + 1) % kNV;
      double abx = vx[j] - vx[i];
      double aby = vy[j] - vy[i];
      double apx = px - vx[i];
      double apy = py - vy[i];
      double cross = abx*apy - aby*apx;
      if (cross < 0.0) { inside = false; break; }
    }
    if (inside) {
      pcp[0] = px;
      pcp[1] = py;
      pcp[2] = 1.0e-3;
    } else {
      double best_dist2 = 1.0e9;
      pcp[0] = px; pcp[1] = py; pcp[2] = 1.0e-3;
      for (int i = 0; i < kNV; i++) {
        int j = (i + 1) % kNV;
        double ax = vx[i], ay = vy[i];
        double bx = vx[j], by = vy[j];
        double abx = bx - ax, aby = by - ay;
        double apx = px - ax, apy = py - ay;
        double len2 = abx*abx + aby*aby;
        double t = (len2 > 1e-9)
            ? mju_max(0.0, mju_min(1.0, (apx*abx + apy*aby) / len2))
            : 0.0;
        double qx = ax + t*abx, qy = ay + t*aby;
        double d2 = (px-qx)*(px-qx) + (py-qy)*(py-qy);
        if (d2 < best_dist2) {
          best_dist2 = d2;
          pcp[0] = qx;
          pcp[1] = qy;
          pcp[2] = 1.0e-3;
        }
      }
    }
  } else {
    // No arm contact: existing foot-foot line projection.
    double axis[3], center[3], vec[3];
    mju_sub3(axis, foot_right_pos, foot_left_pos);
    axis[2] = 1.0e-3;
    double length = 0.5 * mju_normalize3(axis) - 0.05;
    mju_add3(center, foot_right_pos, foot_left_pos);
    mju_scl3(center, center, 0.5);
    mju_sub3(vec, capture_point, center);
    double t = mju_dot3(vec, axis);
    t = mju_max(-length, mju_min(length, t));
    mju_scl3(vec, axis, t);
    mju_add3(pcp, vec, center);
    pcp[2] = 1.0e-3;
  }

  // is leaning - modified to be less strict than standing.
  // floored at 0.3 so balance never fully turns off when falling.
  double leaning =
      torso_height / mju_sqrt(torso_height * torso_height + 0.65 * 0.65) - 0.2;
  leaning = mju_max(leaning, 0.3);

  // ITER 22 (2026-05-18): balance scale gated by EXPECTED LOAD on the brace
  // arm, not just "is there contact at all".
  //
  // ITER 38 (2026-05-18): leg-lift phases get FULL balance authority. With
  // only one foot on the ground (right foot lifted), balance is MORE
  // critical, not less — even with strong arm brace, the support polygon
  // collapses from feet-to-feet to a single foot + arm. WBC-style
  // prioritisation says balance dominates during single-support phases.
  // Two-foot lean phases still load-gate so the bracing arm can take real load
  // without balance fighting it.
  // 2026-05-26: balance authority during the braced two-foot phase is now a
  // CONSTANT, DECOUPLED from the brace-force target. It used to be
  // 1 - 0.65*(target/120) — "the harder I intend to brace, the less I enforce
  // balance." That made sense only while the support triangle was
  // unconditionally open to the hand (balance had to be muted or it fought the
  // lean). Now the triangle's hand vertex is LOAD-LIMITED by MEASURED force (see
  // the any_arm_contact branch above), so a supported forward lean is already
  // permitted geometrically and balance no longer needs muting. Worse, the old
  // coupling meant raising brace_force_target (to carry a DEEPER lean) silently
  // CUT balance authority and reopened the forward OVERSHOOT a multi-run
  // GUI-cadence trace caught faceplanting ~60% of rollouts (CoM out to +159 mm).
  // Softening the coefficient 0.65 -> 0.35 took the hold rate 40% -> 80% (10-run
  // trace); pinning it to a constant 0.80 here removes the brace-force coupling
  // so brace force and balance authority can be tuned INDEPENDENTLY — the brace
  // can now be made to carry real load (a vertical hand force forward of the CoM
  // is a nose-up restoring moment) WITHOUT giving back balance authority.
  double braced_balance_scale = 0.80;
  double balance_scale = braced_balance_scale;
  // -------- Directional Balance with quadratic edge amplification ------ //
  // CoM excursion has 3 directions and they're physically asymmetric:
  //   - FORWARD  (+x in world): the brace hand on the table catches this
  //     fall. Allowed to grow more during braced phases — that's the
  //     whole point of bracing. balance_scale (≈0.62 at brace=70 N)
  //     relaxes the penalty here.
  //   - BACKWARD (-x): NOT catchable by the hand brace. A vertical force
  //     on the table provides zero backward restoring moment. Strict
  //     penalty regardless of brace status.
  //   - LATERAL (±y): also NOT catchable by a same-side hand brace
  //     (think: pushing on a table to your right doesn't stop you
  //     falling left). Strict penalty regardless of brace status.
  //
  // Without this decomposition, the previous formulation gave the
  // planner an escape route: amplifier was 10× in ALL directions, so
  // to reduce edge cost the planner could pull CoM BACKWARD (hip flex
  // back) until excursion was small again. Result: the body folded at
  // the hips, sat back, and tipped over backward onto its butt — even
  // though it was correctly bracing the hand forward. By making forward
  // excursion cheap (per balance_scale) and backward/lateral strict, the
  // backward escape disappears: pulling CoM behind the feet now costs
  // more than leaning forward into the brace.
  //
  // Leg-lift: balance_scale=1.0 is forced, so forward also gets strict
  // treatment (no brace, single-foot support — every direction is risky).
  double cp_dx = capture_point[0] - pcp[0];
  double cp_dy = capture_point[1] - pcp[1];
  // FREE-STANDING FIX (2026-06-04): balance_scale (0.80) discounts a FORWARD capture-
  // point excursion because a table hand-brace provides a nose-up restoring moment that
  // catches the forward fall. The free-standing tasks (stand/crouch/arms/lean_l-r) have
  // NO table — any_arm_contact is false — so that discount lets the CoM drift forward
  // unchecked into nothing => the persistent ~15° forward lean that topples. With no brace,
  // penalize forward as strictly as backward (symmetric => keep CoM centered, no escape
  // direction). Table-lean pipeline (any_arm_contact == true) is unchanged.
  double fwd_scale = any_arm_contact ? balance_scale : 1.0;
  double dir_scale_x = (cp_dx > 0.0) ? fwd_scale : 1.0;
  double dir_scale_y = 1.0;
  double eff_dx_full = cp_dx * dir_scale_x;
  double eff_dy = cp_dy * dir_scale_y;
  double balance_excursion =
      mju_sqrt(eff_dx_full * eff_dx_full + eff_dy * eff_dy);
  constexpr double kEdgeInner = 0.05;       // m — amplifier still 1×
  constexpr double kEdgeOuter = 0.10;       // m — amplifier saturated
  constexpr double kEdgePeakAmplifier = 10.0;
  double edge_t =
      mju_min(1.0, mju_max(0.0, (balance_excursion - kEdgeInner) /
                                    (kEdgeOuter - kEdgeInner)));
  double edge_smooth = edge_t * edge_t * (3.0 - 2.0 * edge_t);
  // The 10x edge amplifier is the SUPPORT-POLYGON BARRIER (restoring authority that ramps up as the
  // capture point nears the foot edge) -- NOT a knife-edge bug. Removing it for free-standing (tried
  // 2026-06-05) deleted the balance restoring force -> faceplant in 3s. KEEP it. The correct cause-B
  // is the SYMMETRIC barrier (fwd_scale=1.0 for no-brace, above) WITH this amplifier intact.
  double edge_amplifier = 1.0 + (kEdgePeakAmplifier - 1.0) * edge_smooth;
  double eff_dx = eff_dx_full;

  // Per-axis residual: directional scale already includes balance_scale
  // for forward, 1.0 for backward/lateral. NO outer multiplication by
  // balance_scale (would double-count for forward, and incorrectly
  // relax backward/lateral).
  residual[counter + 0] = eff_dx * leaning * edge_amplifier;
  residual[counter + 1] = eff_dy * leaning * edge_amplifier;
  counter += 2;

  // ----- torso forward tilt (direction-based) ----- //
  // Encourage forward lean to reach object
  double *torso_forward = SensorByName(model, data, "torso_forward");

  // Vector from torso to reach_target. Drives Torso Forward Tilt.
  double reach_dir[3];
  mju_sub3(reach_dir, reach_target, torso_pos);
  mju_normalize3(reach_dir);

  // Want torso forward axis to align with reach direction
  // dot product should be close to 1.
  //
  // Phase-gated by `phase_reach_scale`: during stand_up the reach-toward-object
  // cost is supposed to be zero (we're stabilising, not leaning). Before this
  // gate the residual was always active at weight 3, creating a continuous
  // gradient pulling the torso to face the object. With the audit-spec ankle
  // PD (kp=20), MPC didn't have the authority to suppress that gradient and
  // the robot oscillated forward/back during stand_up.
  double alignment = mju_dot3(torso_forward, reach_dir);
  residual[counter++] = phase_reach_scale * (1.0 - alignment);

  // ----- pelvis tilt ----- //
  // Phase-dependent residual that mixes two sensors:
  //  • pelvis_up[2]      = cos(tilt magnitude) — symmetric in direction
  //  • pelvis_forward[2] = -sin(pitch angle)   — DIRECTIONAL (forward = -,
  //    backward = +)
  // Round 7 fix used `pelvis_up[2] - target` everywhere, which is symmetric
  // for any tilt direction. At a lean target of 0.85 (= 32° tilt) backward
  // tilt achieved the same target as forward tilt, and MPC chose
  // backward to avoid the Hip-Clearance penalty on forward pelvis travel —
  // the user saw the robot stand → plant hand → tip BACKWARD → fall on its
  // back. Round 8 fix: switch the lean-phase branch to pelvis_forward[2]
  // so the target -sin(32°)=-0.530 is only met by forward pitch; backward
  // pitch gives residual +1.060 (cost ~9.6/step ≈ 640/horizon — overwhelming
  // deterrent vs the ~3 Hip-Clearance penalty MPC was previously avoiding).
  // Upright phases keep pelvis_up because the home pose is at pelvis_up=1.0
  // so the residual is already 0 there and roll is also penalized.
  // BISECTION TEST #2 (2026-05-17): reverted to symmetric residual
  // ALWAYS. Pre-R8 form. The directional `pelvis_forward[2] − (−0.530)`
  // was forcing 32° pitch regardless of CoM state; with
  // weak ankle PD (kp=20) this over-committed the lean and let the planner
  // exploit the pelvis-table exclude. Symmetric is permissive: any tilt
  // (forward or backward) costs the same — MPC picks based on other terms.
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  // ITER 31 (2026-05-18): pelvis-tilt residual gives 60° of free forward
  // bow during arm-contact phases, then penalises beyond. Iter 30 made it
  // fully free (0°-90°+) which let the body collapse forward past
  // sustainable balance per user report. 60° is the natural braced-lean
  // depth a human uses on a counter (deep enough for arms to reach,
  // shallow enough that brace arm + ankle PD can hold static balance).
  // Threshold = cos(60°) = 0.5 → residual = max(0, 0.5 - pelvis_up[2]).
  // Forward tilt up to 60° (pelvis_up ≥ 0.5) is free; deeper bow incurs
  // increasing cost.
  // Lean cap: cos(60°) = 0.5, i.e. the pelvis may tilt up to 60° while an arm
  // is on the table. Only braced phases get here (the gate below), which is the
  // whole point of bracing.
  double pelvis_tilt_threshold = 0.5;
  double pelvis_tilt_residual;
  if (any_arm_contact) {
    pelvis_tilt_residual = mju_max(0.0, pelvis_tilt_threshold - pelvis_up[2]);
  } else {
    // FREE-STANDING upright (stand/crouch/arms_*): the legacy residual
    // pelvis_up[2]-1 == cos(tilt)-1 is QUADRATICALLY FLAT near vertical, so a
    // ~17deg counterbalance lean cost almost nothing and the planner parked
    // there (butt-back, arms-forward) on the decoupled twin -> metastable, then
    // toppled. Root-caused 2026-06-05. Replace with sin(tilt) (horizontal
    // magnitude of the pelvis up-vector), which is ~LINEAR near vertical (~6x
    // sharper at 17deg) -> pulls the torso erect. Scaled by the `upright_gain`
    // numeric so it is tunable/reversible WITHOUT a recompile:
    //   gain  > 0 : sharpened linear term * gain  (default 1.0 = the fix)
    //   gain <= 0 : exact legacy cos-flat behaviour (clean A/B baseline)
    // The arm-contact/lean branch above is UNTOUCHED -> table-brace lean intact.
    double upright_gain = GetNumberOrDefault(1.0, model, "upright_gain");
    if (upright_gain <= 0.0) {
      pelvis_tilt_residual = pelvis_up[2] - 1.0;
    } else {
      double sin_tilt =
          mju_sqrt(mju_max(0.0, 1.0 - pelvis_up[2] * pelvis_up[2]));
      pelvis_tilt_residual = upright_gain * mju_max(0.0, sin_tilt);
    }
  }
  // ★ 2026-08-05: DIRECTIONAL BACKWARD PENALTY (numeric-gated).
  // Both branches above are SYMMETRIC in tilt DIRECTION -- they read
  // `pelvis_up[2] = cos(tilt)`, which is 0.8996 at +25.9 deg and 0.9063 at
  // -25 deg. So the residual's MINIMUM sits exactly at vertical, and the
  // restoring gradient vanishes precisely where a robot un-bowing off a brace
  // has to be caught. Momentum then carries it straight through into a
  // backward fall, with nothing opposing it.
  //   MEASURED, bench20_splayfix (n=20): 5 runs ran away through vertical to
  //   -28..-90 deg at 62-352 deg/s peak pitch rate, while EVERY stand-back
  //   stayed under 19 deg/s. 4 of the 5 began inside a 1.6 s window (t=64.8
  //   -66.4), 2-4 s after the forearm left the slab.
  // ⚠ THE HISTORY IS IN THIS FILE: a Round-8 fix made this residual directional
  // (`pelvis_forward[2]`, backward = +) *specifically* because "the user saw the
  // robot stand -> plant hand -> tip BACKWARD -> fall on its back". It was then
  // reverted on 2026-05-17 as BISECTION TEST #2 and never restored.
  // `pelvis_forward[2] = -sin(pitch)`: FORWARD lean is NEGATIVE (-0.437 at the
  // brace keyframe), BACKWARD is POSITIVE (+0.423 at -25 deg) -- FK-verified.
  // Gate: `backward_tilt_gain` numeric. 1.0 = the symmetric legacy form,
  // BYTE-IDENTICAL, so the default changes nothing and the A/B is one numeric
  // edit with no second recompile.
  double backward_tilt_gain =
      GetNumberOrDefault(1.0, model, "backward_tilt_gain");
  if (backward_tilt_gain != 1.0) {
    double *pelvis_forward_dir = SensorByName(model, data, "pelvis_forward");
    // ★ Guard: a null sensor must leave the residual UNTOUCHED, not zero it.
    // A missing <user> sensor silently misaligned every later residual once
    // already (Lean_H12.xml, 2026-07-30).
    if (pelvis_forward_dir && pelvis_forward_dir[2] > 0.0) {
      pelvis_tilt_residual *= backward_tilt_gain;
    }
  }
  residual[counter++] = pelvis_tilt_residual;

  // ----- foot up-vectors: prevent ankle roll ----- //
  double *foot_right_up = SensorByName(model, data, "foot_right_up");
  double *foot_left_up  = SensorByName(model, data, "foot_left_up");
  residual[counter++] = mju_abs(foot_right_up[2] - 1.0);
  residual[counter++] = mju_abs(foot_left_up[2]  - 1.0);

  // ----- waist yaw: stop planner from yawing torso to swing arm ----- //
  // torso_joint is the 13th actuated DOF (nu index 12), home = 0.
  // Confirmed from ResetLocked joint-order print: qpos index 19 = 7 + 12.
  residual[counter++] = data->qpos[7 + 12] - model->key_qpos[7 + 12];

  // ----- hip yaw + roll: prevent planner exploiting hip rotation to
  // reposition the torso/arm during lean (same exploitation mode as waist yaw).
  // Nu indices: L_hip_yaw=0, L_hip_roll=2, R_hip_yaw=6, R_hip_roll=8.
  //
  // 2026-05-19: at the base weight (20) these were overpowered by the
  // balance/reach gradients during the lean — the planner twisted the legs to
  // shuffle the CoM (user: "legs kinda twist on its own"). The leg lift only
  // stays balanced from a SQUARE stance (support foot pointing straight
  // forward, no splay), so we scale the squaring authority up while braced
  // (×2) and harder during leg-lift (×4). Targets are home (key_qpos = facing
  // front), so this drives the legs back to a clean forward stance.
  double hip_square_scale = any_arm_contact ? 2.0 : 1.0;
  // 2026-05-20: FK rollout (monitor/phase_snapshot.py) showed the slant was a
  // single-support phenomenon almost entirely on the STANCE (left) leg (L_hip_yaw
  // to -21.8°, L_hip_roll to -11.5°), while staying within ±4° in two-foot
  // phases. That hip-yaw twist IS the cross-legged look. Single support is gone
  // now, but the squaring authority it motivated is still what keeps
  // square the STANCE hip far harder (×12) during leg-lift but leave the
  // lifting hip at the base scale. User dir: stance leg vertical & square at
  // all times; let MPC find the brace+CoP balance instead of twisting the leg.
  double stance_square_scale = hip_square_scale;
  residual[counter++] = stance_square_scale * (data->qpos[7 + 0] - model->key_qpos[7 + 0]);
  residual[counter++] = stance_square_scale * (data->qpos[7 + 2] - model->key_qpos[7 + 2]);
  residual[counter++] = hip_square_scale * (data->qpos[7 + 6] - model->key_qpos[7 + 6]);
  residual[counter++] = hip_square_scale * (data->qpos[7 + 8] - model->key_qpos[7 + 8]);

  // ----- posture ----- //
  // Reduced weight vs push task to allow more deviation for leaning.
  // Phase-scaled: ×3 during stand_up, ramps down to ×1 entering arm_extend.
  // Why: the Posture cost is the ONLY signal that pulls knee + hip_pitch
  // back to extension. Hip yaw/roll, waist yaw, foot up have dedicated
  // residuals; knee + hip_pitch only get the general 27-dim Posture pull.
  // At weight 0.015 with phase_posture_scale=1 it's too weak — once a knee
  // drifts a few degrees, nothing pulls it back. During stand_up the boost
  // gives Posture 9× more effective cost (quadratic), keeping legs extended.
  mju_sub(&residual[counter], data->qpos + 7, posture_target + 7, model->nu);
  mju_scl(&residual[counter], &residual[counter], phase_posture_scale,
          model->nu);
  // ----- targeted knee + hip-pitch extension anchor (free-standing only) -----
  // The general 27-dim Posture pull is the ONLY signal holding knee + hip_pitch
  // to the keyframe (see comment above), and at its effective weight it is too
  // weak: over the 1s horizon the sampler slowly trades leg extension for a lower
  // CoM and the legs creep open until collapse (the ~30-90s stand/crouch sink,
  // 2026-06-06 research). Amplify ONLY the leg entries (hip_pitch nu-idx 1/7,
  // knee 3/9) inside the Posture residual so the legs hold their keyframe target
  // (straight for stand, 0.7 for crouch) WITHOUT over-constraining the 23 other
  // DOFs -- raising GLOBAL Posture did that and caused the asymmetric crouch.
  // Gated to free-standing (any_arm_contact == false) so the table-lean
  // brace tasks 0-5 are byte-identical. Tunable: <numeric name="leg_extension_gain">
  // (1.0 = off / unchanged; sweep up if it still creeps, down if it over-stiffens).
  if (!any_arm_contact) {
    double leg_gain = GetNumberOrDefault(2.5, model, "leg_extension_gain");
    // Bent-knee holds (crouch/squat) need a STRONGER symmetric leg anchor than
    // straight-knee holds (stand/arms). At the live plan rate the saddle-unstable
    // symmetric crouch breaks into an asymmetric one-leg PROP — one knee drives to
    // its -0.12 ctrl floor while the other over-flexes — unless the bilateral
    // knee+hip-pitch pull is stiff enough to outweigh the balance benefit of
    // propping. Gate on a bent TARGET knee (posture_target L/R knee, nu-idx 3/9 =
    // qpos 7+3 / 7+9, > 0.3 rad) so straight-knee strategies hit neither branch and
    // stay byte-identical to the accepted stand/arms (NO regression). Tunable.
    if (posture_target[7 + 3] > 0.3 || posture_target[7 + 9] > 0.3)
      leg_gain = GetNumberOrDefault(6.0, model, "crouch_leg_extension_gain");
    for (int li : {1, 3, 7, 9}) residual[counter + li] *= leg_gain;
  }
  // PIN THE NON-REACHING ARM (strat 21, 2026-06-23). The reach keeps GLOBAL
  // Posture LOW (12) on purpose so the REACHING arm is free to extend (line ~75,
  // the jab lesson: a high Posture parks a limb at rest). But that ALSO leaves the
  // OTHER arm loose, so the planner throws it around for balance -- the user sees
  // this as the reach "switching arms" / the idle arm flailing. Fix the role
  // assignment right=reach (reach_hand) / left=HELD brace-ready arm: amplify ONLY
  // the non-reaching arm's 7 Posture entries (same per-index pattern as the leg
  // anchor above; arm actuators left 13..19, right 20..26, actuator i == joint i+1)
  // so it is held at its keyframe (rest) pose while the reaching arm + legs (= stand)
  // stay free. reach_brace_hold = boost factor (effective arm Posture = Posture *
  // this), live-tunable. Reach-gated -> all other strategies byte-identical.
  if (residual_keyframe_.name == "reach_to_target") {
    int rh_id2 = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
    int rh_mode2 = (rh_id2 >= 0)
        ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id2]]) : 0;
    bool reaching_right = (rh_mode2 == 2) ? true
                        : (rh_mode2 == 1) ? false
                        : (lateral_of(data->mocap_pos) < 0.0);  // 0/auto (robot's right)
    double kArmHold = GetNumberOrDefault(4.0, model, "reach_brace_hold");
    int arm0 = reaching_right ? 13 : 20;   // pin the OTHER (non-reaching) arm
    for (int j = arm0; j < arm0 + 7 && j < model->nu; j++)
      residual[counter + j] *= kArmHold;
    // LOCK THE REACHING ARM forward (strat 21, 2026-06-24). User: "the arm should
    // NOT go back at all." The planner uses the reaching arm as a BALANCE actuator
    // (folds/throws it to pull CoM back the moment CoM drifts) — that defeats a
    // reach. The faithful twin proves the planner CAN hold the arm; the retract is a
    // cost tradeoff the balancer wins under a high com_back / noisy state. So make the
    // reaching arm OFF-LIMITS to the balancer: amplify ONLY its shoulder pitch/roll/yaw
    // Posture (the throw-back + fold-across-chest DOFs) so balance can never out-pull
    // it — the body/legs must do the balancing instead. Elbow + wrist stay free so the
    // Cartesian reach still extends the hand. reach_arm_lock = boost factor (1 = off).
    double kArmLock = GetNumberOrDefault(1.0, model, "reach_arm_lock");
    if (kArmLock > 1.0) {
      int rarm = reaching_right ? 20 : 13;   // reaching-arm shoulder base nu-idx
      // shoulder PITCH (rarm+0): ONE-SIDED lock. Posture residual = qpos - keyframe(0);
      // the throw-back drives pitch POSITIVE (+, ~+80deg = arm back/up), the FORWARD
      // reach drives it NEGATIVE. So amplify ONLY when residual>0 (going backward) to
      // hard-block the throw-back, and leave residual<0 (forward) FREE so the arm
      // fully EXTENDS. This gives deep reach AND no retraction.
      if (residual[counter + rarm] > 0.0) residual[counter + rarm] *= kArmLock;
      // shoulder ROLL + YAW: symmetric lock — block the fold-across-chest (yaw) and
      // sideways collapse (roll); the forward reach barely uses these.
      residual[counter + rarm + 1] *= kArmLock;
      residual[counter + rarm + 2] *= kArmLock;
    }
  }
  // CROUCH-DEPTH ANCHOR (strat 21 reach, 2026-06-25). On real the reach held
  // 0.29-0.39 m forward for ~73 s, then the body slowly SANK into a deeper crouch
  // (knee 0.35 keyframe -> 0.53) and the RIGHT ELBOW saturated (the ~1.8 kg magpie
  // gripper exceeds the 18 Nm elbow once the shoulder drops) -> the arm folded to
  // 0.09 m. The body never fell (tilt ~1deg); the CROUCH GEOMETRY killed the reach,
  // not balance. The symmetric crouch_leg_extension_gain (4.5x) above did not hold
  // it. Keep the shoulder HIGH so the elbow stays in its holdable envelope: amplify
  // the knee Posture residual ONLY when the knee bends PAST its keyframe (residual>0
  // = sinking deeper); leave straightening (residual<0) FREE so normal compliance is
  // untouched. One-sided, like the arm lock. reach_to_target ONLY (other crouch
  // strategies keep the symmetric gain). reach_knee_anchor = boost factor (1 = off);
  // stacks on the 4.5x. Generic (both knees, side-agnostic).
  if (residual_keyframe_.name == "reach_to_target") {
    double kKneeAnchor = GetNumberOrDefault(1.0, model, "reach_knee_anchor");
    if (kKneeAnchor > 1.0) {
      for (int kn : {3, 9})   // Lknee / Rknee nu-idx (actuator i == joint i+1)
        if (residual[counter + kn] > 0.0) residual[counter + kn] *= kKneeAnchor;
    }
  }
  counter += model->nu;

  // com vel
  double *waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double *torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // ----- move feet ----- //
  double *foot_right_vel = SensorByName(model, data, "foot_right_vel");
  double *foot_left_vel = SensorByName(model, data, "foot_left_vel");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], leaning, 2);
  counter += 2;

  // ----- control ----- //
  mju_sub(&residual[counter], data->ctrl, posture_target + 7,
          model->nu);  // because of pos control (tracks pose-library key)
  counter += model->nu;

  // ----- bracing hand position on table ----- //
  // Reach-to-contact, then FADE OUT (2026-05-22). `ideal_brace.z` sits 6 cm
  // below the surface so the pull is strong enough to bring the forearm down to
  // contact under the tight real-robot ROM. But a below-surface target keeps
  // dragging the body onto the arm AFTER contact -- and the one-sided Brace
  // Force cost lets the planner press arbitrarily hard for free -- giving a
  // ~450 N entry slam and 0<->117 N on/off chatter. So fade the position pull as
  // the MEASURED brace force approaches the per-phase target; past target the
  // Brace Force cost alone holds a steady press. This is force feedback through
  // the position gate: a force spike lowers the gate (less push) and a dropout
  // raises it (more push), which damps the contact oscillation rather than
  // driving it. Pre-contact (force 0) the gate is 1.0, so reaching is unchanged.
  double brace_force_ref  = mju_max(15.0, residual_keyframe_.brace_force_target);
  double brace_force_frac = mju_min(1.0, brace_contact_force / brace_force_ref);
  double brace_pos_gate   = phase_brace_pos_scale * (1.0 - 0.85 * brace_force_frac);
  mju_sub3(&residual[counter], bracing_hand, ideal_brace);
  mju_scl3(&residual[counter], &residual[counter], brace_pos_gate);
  counter += 3;

  // Per-phase brace-force reference.
  // ITER 22 (2026-05-18): ONE-SIDED shortfall residual. The previous symmetric
  // residual `desired - actual` was actively pushing MPC AWAY from any force
  // exceeding the target — at a low target (8 N) the planner was penalised
  // ~100× harder for pushing 30N than for pushing 8N, even though more support
  // is exactly what the body needs. Switching to `max(0, desired - actual)`
  // means: pushing harder than the target is FREE, only under-supporting the
  // brace incurs cost. Combined with the bumped per-phase targets in the
  // strategy JSON this tells MPC "transfer this much of body weight through
  // the arm", matching
  // one-sided contact-force tracking — only under-support costs.
  bool is_active_contact =
      (residual_keyframe_.contact_pairs[0].body1 != mjpc::humanoid::kNotSelectedInteract);
  double target_brace_force = residual_keyframe_.brace_force_target >= 0.0
                                   ? residual_keyframe_.brace_force_target
                                   : (is_active_contact ? 70.0 : 0.0);
  // ITER 28 (2026-05-18): smoothstep ramp the brace_force target across phase
  // boundaries — same machinery as phase_reach_scale etc. Without this, going
  // from stand_up (target=0) into a braced phase (target=60) is a step change
  // in the cost gradient; MPC reads it as "you need 60 N right now" and plans
  // an impulsive arm slam into the table. The 1.5 s smoothstep ramp gives MPC
  // ~100 control cycles to bring the arm into contact and build force
  // gradually, which spreads the contact impulse over enough time that the
  // foot-lift transient (user-flagged in iter 27) becomes negligible.
  double desired_brace_force =
      prev_phase_brace_force_target_ +
      alpha * (target_brace_force - prev_phase_brace_force_target_);
  // PROXIMITY GATE (2026-05-26): only demand brace force once the bracing hand
  // is NEAR its table target. Un-gated, max(0, desired-actual) is a large
  // constant error whenever the forearm isn't in contact, so the planner can
  // only shrink it by pushing FORWARD to chase a contact it can't seat in
  // real time — a forward dive into a fall (user: "leans in to brace, falls
  // forward fast, right leg up"). Brace Pos (above) still pulls the forearm to
  // the table un-gated, so there is no chicken-and-egg: position drives the
  // approach; the force demand ramps in (smoothstep) only over the last ~15 cm.
  double brace_reach_gap = mju_dist3(bracing_hand, ideal_brace);
  // BRACE GATE FIX (2026-07-25, `brace_gate_fix` numeric; 0 = OFF = byte-identical).
  // RESTORES THE DOCUMENTED DESIGN: the ideal_brace comment (~line 1333) already
  // states this gate is "decoupled below (keyed on the pad's height above the
  // PHYSICAL surface, not this deliberately-unreachable drive target)" -- but the
  // line above keys it on the 3-D distance to that very drive target, and the drive
  // target is deliberately BELOW the surface. Measured consequence: table_surface_pos
  // is a framepos on the table_top GEOM FRAME (z 0.820), 50 mm under the physical
  // face (z 0.870), and ideal_brace drops a further 60 mm to z 0.760 -- so a
  // PERFECTLY SEATED pad sits 0.1745 m away, past the 0.15 m span, and this gate is
  // EXACTLY 0.000. The entire Brace-Force demand (weight 60 x ~37 residual =
  // ~2230/step) is therefore multiplied by zero even with the brace seated, which is
  // why the forearm hovered a median 19 mm for all 75 s in 2/2 faithful-chain runs
  // and why three separate weight sweeps moved nothing.
  // Keying on CLEARANCE ABOVE THE FACE gives: seated -> 1.000, 19 mm hover -> 0.956,
  // 150 mm -> 0.000, so the anti-dive purpose of the gate is preserved. The legacy
  // 3-D distance is retained whenever the pad is NOT horizontally over the table, so
  // the gate can never reward pressing into thin air (the documented "chase a contact
  // it can't seat -> forward dive" failure). ideal_brace is deliberately left alone:
  // its deep sub-surface drive is what bows the body, and raising it slackens the
  // pull -> less bow -> MORE hover (verified on the 0.87 probe).
  int bgf_id = mj_name2id(model, mjOBJ_NUMERIC, "brace_gate_fix");
  double kBraceGateFix = (bgf_id >= 0)
      ? model->numeric_data[model->numeric_adr[bgf_id]] : 0.0;
  if (kBraceGateFix > 0.0 && is_forearm_brace) {
    int tt_id = mj_name2id(model, mjOBJ_GEOM, "table_top");
    double const *tsp = SensorByName(model, data, "table_surface_pos");
    if (tt_id >= 0 && tsp) {
      // half-extents of the table slab, from the COMPILED geom (never hardcoded)
      double thx = model->geom_size[3 * tt_id + 0];
      double thy = model->geom_size[3 * tt_id + 1];
      double face_z = tsp[2] + model->geom_size[3 * tt_id + 2];
      int pad_id = mj_name2id(model, mjOBJ_GEOM,
                              reach_right ? "left_forearm_pad" : "right_forearm_pad");
      // capsule radius: a flat-seated pad's SITE sits one radius above the face.
      // half-length grows the footprint so a pad already touching via the near edge
      // is not read as off-table.
      double pad_r  = (pad_id >= 0) ? model->geom_size[3 * pad_id + 0] : 0.035;
      double pad_hl = (pad_id >= 0) ? model->geom_size[3 * pad_id + 1] : 0.046;
      // CONTINUOUS distance from the pad to the table SLAB (not a boolean switch
      // between two unrelated metrics). Equals the pure height gap while the pad is
      // over the slab, and grows smoothly as it leaves -- so there is no cost cliff
      // at the near edge. A boolean over_table test here would step this gate
      // 0.000 -> 0.956 across ~2 mm of pad travel at x = 0.400, and because the
      // Brace-Force residual is a one-sided SHORTFALL, max(0, 50 N - F), crossing
      // that edge before any force exists would COST ~2100/step while staying
      // outside costs 0. The planner's cheapest in-horizon answer would be to hover
      // just short of the edge -- the same hover this fix exists to cure, merely
      // relocated, plus a non-differentiable cost a sampling planner reads as noise.
      double dxo = mju_max(0.0, mju_abs(bracing_hand[0] - tsp[0]) - (thx + pad_hl));
      double dyo = mju_max(0.0, mju_abs(bracing_hand[1] - tsp[1]) - (thy + pad_hl));
      double dzo = bracing_hand[2] - (face_z + pad_r);
      brace_reach_gap = mju_sqrt(dxo * dxo + dyo * dyo + dzo * dzo);
    }
  }
  double bgg = mju_min(1.0, brace_reach_gap / 0.15);
  double brace_force_prox_gate = 1.0 - bgg * bgg * (3.0 - 2.0 * bgg);

  // ---- RECOVERABILITY GATE (2026-07-27) --------------------------------- //
  // `lean_recover_bound` [m]: 0 = OFF = BYTE-IDENTICAL. Shrinks the two big FORWARD
  // demands (Brace Force here, Reaching Hand Dist below) to zero once the capture
  // point has advanced further past the FEET than a feet-only recovery could undo.
  //
  // WHY. The comment on the reach residual just below states the design intent:
  // "Reach gradient is intentionally NOT balance-capped ... Balance's edge_amplifier
  // above is what forces the trade-off". That assumption is QUANTITATIVELY FALSE, and
  // it is the whole reason the braced lean drapes. Measured magnitudes in the active
  // brace phase (h12_lean_brace.json phase[3] forearm_brace_lean):
  //     Brace Force        w 60, SmoothAbs p=15, one-sided max(0, 50 - F)
  //                        -> 60*(sqrt(50^2+15^2)-15) = 2232 / step when uncontacted
  //     Reaching Hand Dist w 80 against a target ~1.0 m out vs a ~0.52 m arm, so the
  //                        error NEVER nulls        ->      ~56 / step, permanently
  //     Balance            w 2.5, incl. the 10x edge amplifier AND the 0.80 braced
  //                        forward discount         ->        ~6 / step at a 0.30 m
  //                                                            over-excursion
  // i.e. ~380 : 1 against balance. The planner is not misbehaving -- it is correctly
  // buying CoM excursion with brace reward. Measured consequence on the A2 twin
  // (2026-07-27, pipeline_arpa_brace_fixed): a genuinely flat 39 s forearm brace
  // (53% of frames, pad tilt 0.01 deg) at CoM margin +0.553 m -- 4x the toe limit
  // +0.133 -- which then FELL FORWARD at 115.9 s.
  //
  // Raising Balance instead does NOT work and has been tried 3x independently: it
  // removes the drape but supplies no feasible alternative, so it falls sooner. The
  // asymmetry has to be fixed on the DEMAND side, which is what this gate does.
  //
  // ★ Deliberately keyed on the FEET-ONLY excursion, never on the load-limited
  // {L_foot, R_foot, hand} triangle used for Balance. That triangle credits its hand
  // vertex from the INSTANTANEOUS measured force (hand_load_frac = min(0.9, F/140)),
  // which is positive feedback: press harder -> licensed further out -> more dependent
  // on pressing. Nothing there requires the CoM to be able to come back if the force
  // lapses, and a lapse is exactly what the forward fall looks like. A brace may still
  // carry real load; it may no longer buy unrecoverable excursion.
  //
  // Forward axis = MEAN OF THE FEET'S OWN HEADINGS (midFeetZUp), matching the
  // 2026-07-16 balance_frame fix and the com_x_offset_support block above.
  // ⚠ NOT the perpendicular to the foot LINE -- that was measured 21 deg wrong
  // (the 2026-07-16 support-frame bug).
  double lean_recover_gate = 1.0;
  int lrb_id = mj_name2id(model, mjOBJ_NUMERIC, "lean_recover_bound");
  double kRecoverBound =
      (lrb_id >= 0) ? model->numeric_data[model->numeric_adr[lrb_id]] : 0.0;
  if (kRecoverBound > 0.0 && any_arm_contact) {
    int lrs_id = mj_name2id(model, mjOBJ_NUMERIC, "lean_recover_span");
    double kRecoverSpan =
        (lrs_id >= 0) ? model->numeric_data[model->numeric_adr[lrs_id]] : 0.0;
    if (kRecoverSpan <= 1.0e-6) kRecoverSpan = 0.10;   // fade over 10 cm by default
    double rg_fwd[2] = {1.0, 0.0};
    double const *rg_flf = SensorByName(model, data, "foot_left_forward");
    double const *rg_frf = SensorByName(model, data, "foot_right_forward");
    if (rg_flf && rg_frf) {
      double fx = rg_flf[0] + rg_frf[0], fy = rg_flf[1] + rg_frf[1];
      double len = mju_sqrt(fx * fx + fy * fy);
      if (len > 1.0e-6) { rg_fwd[0] = fx / len; rg_fwd[1] = fy / len; }
    }
    double mfx = 0.5 * (foot_left_pos[0] + foot_right_pos[0]);
    double mfy = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    // signed forward excursion of the capture point past the midfoot, feet frame
    double cp_fwd_feet = (capture_point[0] - mfx) * rg_fwd[0] +
                         (capture_point[1] - mfy) * rg_fwd[1];
    double over = (cp_fwd_feet - kRecoverBound) / kRecoverSpan;
    over = mju_max(0.0, mju_min(1.0, over));
    // smoothstep, so a sampling planner sees a continuous slope rather than a cliff
    lean_recover_gate = 1.0 - over * over * (3.0 - 2.0 * over);
  }

  // ★ 2026-08-02 THIS WAS A REPULSIVE BARRIER, NOT A REWARD.
  //   Old: gate * max(0, F_des - F). `prox_gate` is 0 FAR and 1 NEAR, so a
  //   planner sitting 150 mm away paid ZERO, and the instant it approached
  //   without load the cost jumped to the full shortfall (~2232/step at w60).
  //   Approaching was punished; staying away was free. Measured consequence:
  //   the planner parked 4.6 mm outside its own gate and never braced, so the
  //   weight was zeroed -- which left NOTHING in the cost stack commanding
  //   contact at all, and every brace since came from the body happening to sag.
  //
  //   Fixed: while far, charge the FULL demand (flat, no cliff); once near,
  //   charge only the remaining force shortfall. The residual is then
  //   MONOTONE NON-INCREASING along approach-then-press: F_des far away,
  //   F_des near-but-unloaded, 0 when the brace carries its target. Closing
  //   the gap never costs more than staying away, and loading always pays.
  double brace_far_cost =
      desired_brace_force * (1.0 - brace_force_prox_gate);
  double brace_shortfall =
      brace_force_prox_gate *
      mju_max(0.0, desired_brace_force - brace_contact_force);
  residual[counter++] =
      lean_recover_gate * (brace_far_cost + brace_shortfall);

  // ------ object distance (reaching hand) ------ //
  // Phase-gated: zero during stand_up so the planner doesn't lunge.
  // Reach gradient is intentionally NOT balance-capped — the planner
  // should keep wanting to reach; Balance's edge_amplifier above is what
  // forces the trade-off (steep edge penalty makes the planner find a
  // counter-balanced posture rather than tipping).
  // Target = `reach_target` (the mocap object — see top of Residual()).
  // FOREARM BRACE load-transfer (P4): sequence the reach behind the brace WITHOUT
  // dead-locking it. The old gate ramped 0.15 -> 1.0 on measured brace FORCE alone;
  // on a surface where the forearm can't fully seat (e.g. a low table the forearm
  // only struts down onto) the force never arrives, so the reach stayed strangled
  // at 15% and the reaching arm never extended (it just relaxed onto the table).
  // Fix: open the reach on brace PROXIMITY *or* force — whichever is greater. As
  // the forearm arrives over the table (brace_force_prox_gate -> 1) the reach
  // releases; it is already positioned to catch the load, so the CoM-overshoot the
  // force-gate guarded against can't happen. Floor 0.40 keeps the arm visibly
  // reaching throughout. Other phases unchanged (gate = 1.0).
  double brace_reach_gate = is_forearm_brace
      ? (0.40 + 0.60 * mju_max(brace_force_prox_gate,
                               mju_min(1.0, brace_contact_force /
                                       mju_max(1.0, desired_brace_force))))
      : 1.0;
  // REACH DEADBAND (2026-07-25, `reach_deadband` numeric [m]; 0 = OFF =
  // byte-identical). A zero-cost, zero-slope BALL of radius tol around the reach
  // target. Both reach terms below are the SAME error vector, and every shipped
  // reach target is DELIBERATELY unreachable (1.4 m out vs a ~0.52 m arm), so |e|
  // never reaches 0 and the cost keeps a permanent forward slope for the entire
  // run. A sampling planner rectifies that constant bias out of control noise into
  // a forward CoM RATCHET: measured +0.364 m of drift over 600 s, ending with the
  // robot draped on the table. Shrinking the error by (1 - tol/|e|) makes the
  // residual EXACTLY zero inside the ball, so no sampled perturbation there is
  // preferred over any other and there is nothing left to rectify; outside, the
  // pull keeps its direction and full slope and is merely offset by tol. The kink
  // at the boundary is smoothed for free by the cost norm itself (type 6 smooth-abs,
  // p = 0.1), whose argument passes through 0 exactly there.
  int rdb_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_deadband");
  double kReachDeadband = (rdb_id >= 0)
      ? model->numeric_data[model->numeric_adr[rdb_id]] : 0.0;
  // ★ 2026-08-10 `reach_side_swing` numeric (0 = OFF = byte-identical): route the
  // reaching hand AROUND THE SIDE of the slab instead of straight at the target.
  // WHY: the straight-line pull drags the hand into the slab front/underside --
  // measured on the restored-0.985 bench, the STRONGEST braces had the LEAST
  // over-slab time (75 s brace, 2 s reach) because the arm fights the table
  // (user-observed; ownsim's arm swung around the side). While the hand is BELOW
  // the surface and INBOARD of the slab's reach-side edge, the target becomes a
  // via-point: no forward pull (via x = hand x), swing OUT past the side edge and
  // UP over the surface; once clear, the true target engages from above the side.
  // Pure function of hand position -> stateless, rollout-thread safe.
  double via_storage[3];
  {
    int ss = mj_name2id(model, mjOBJ_NUMERIC, "reach_side_swing");
    if (ss >= 0 && model->numeric_data[model->numeric_adr[ss]] > 0.0 &&
        is_forearm_brace) {
      int tg = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
      if (tg < 0) tg = mj_name2id(model, mjOBJ_GEOM, "table_top");
      if (tg >= 0 && std::isfinite(table_near_edge_x)) {
        double surf_z = data->geom_xpos[3 * tg + 2] + model->geom_size[3 * tg + 2];
        double half_y = model->geom_size[3 * tg + 1];
        double side = reach_right ? -1.0 : 1.0;   // right arm goes around -y
        const double* h = reaching_hand;
        // 2026-08-12: constants promoted to LIVE numerics (real run 8: the arc
        // engaged too late/low -- operator had to hand-assist past the front
        // edge). Defaults = the retuned earlier/wider/higher arc; absent
        // numerics fall back to these same values.
        auto numor = [&](const char* nm, double dflt) {
          int id = mj_name2id(model, mjOBJ_NUMERIC, nm);
          return id >= 0 ? model->numeric_data[model->numeric_adr[id]] : dflt;
        };
        double eng_below = numor("swing_engage_below", 0.10);  // was 0.03
        double eng_near  = numor("swing_engage_near", 0.25);   // was 0.10
        double swing_out = numor("swing_out", 0.22);           // was 0.18
        double swing_up  = numor("swing_up", 0.20);            // was 0.12
        bool below = h[2] < surf_z + eng_below;
        bool inboard = side * h[1] < half_y + 0.06;  // not yet past the side edge
        bool near_slab = h[0] > table_near_edge_x - eng_near;
        if (below && inboard && near_slab) {
          via_storage[0] = h[0];                   // no forward pull while blocked
          via_storage[1] = side * (half_y + swing_out); // OUT past the side edge...
          via_storage[2] = surf_z + swing_up;      // ...and UP over the surface
          reach_target = via_storage;
        }
      }
    }
  }
  double reach_err[3];
  mju_sub3(reach_err, reaching_hand, reach_target);
  if (kReachDeadband > 0.0) {
    double reach_err_norm = mju_norm3(reach_err);
    // keep = 0 when |e| <= tol also avoids dividing by a vanishing norm.
    double keep = (reach_err_norm > kReachDeadband)
        ? (1.0 - kReachDeadband / reach_err_norm) : 0.0;
    mju_scl3(reach_err, reach_err, keep);
  }
  // lean_recover_gate (== 1.0 unless `lean_recover_bound` > 0) retires the permanent
  // forward pull once the capture point is past feet-only recovery -- see the long
  // note at the Brace Force residual. This is the "balance-cap" the comment above
  // says is intentionally absent; it is absent because Balance was ASSUMED to force
  // the trade-off, and Balance loses that contest ~380:1.
  mju_copy3(&residual[counter], reach_err);
  mju_scl3(&residual[counter], &residual[counter],
           phase_reach_scale * leaning * brace_reach_gate * lean_recover_gate);
  counter += 3;

  // ----- reaching hand distance to object ----- //
  mju_copy3(&residual[counter], reach_err);
  mju_scl3(&residual[counter], &residual[counter],
           phase_reach_scale * brace_reach_gate * lean_recover_gate);
  counter += 3;

  // ----- foot stability: restoring force toward home XY position ----- //
  // Position-based (not velocity-based) so there's a continuous gradient even
  // when the foot is stationary but displaced. Home positions taken from the
  // actual home-pose foot xipos (= body inertial frame, which is what the
  // `framepos objtype="body"` sensor returns — NOT the kinematic xpos). At
  // the H1-2 home keyframe the ankle_roll_link inertial offset places the
  // foot COM at x = 0.2196, not pelvis_x = 0.19. The old 0.19 constant
  // (copied from the pelvis qpos) made the residual +0.03 at home, adding
  // a continuous 3 cm backward pull on both feet. (2026-05-26: leg-lift is
  // DROPPED — both feet stay grounded; the "right foot freed" branch is gone.
  // See the lean.h header.)
  static constexpr double kRightFootHomeXY[2] = {0.2196, -0.163};
  static constexpr double kLeftFootHomeXY[2]  = {0.2196,  0.163};
  // FOREARM BRACE (2026-07-01): anchor the feet ~8 cm FORWARD (x 0.22 -> 0.30) so
  // the support polygon extends toward the table. The planner stalls the bow ~3 cm
  // short of seating the forearm because the CoM (~0.33) reaches the front foot
  // edge (~0.33) and any deeper bow would tip; a forward stance moves that edge to
  // ~0.41, letting it commit the last cm of bow to press the forearm onto the
  // 0.87 m table. Other strategies keep the 0.22 home (is_forearm_brace-gated).
  double brace_foot_x = is_forearm_brace ? 0.30 : kRightFootHomeXY[0];
  // ★★★ 2026-08-09 MEASURED-STANCE PIN (READ ONLY -- this residual runs in
  // parallel rollout threads; the value is captured in TransitionLocked from the
  // real state). NaN => not pinned => hardcoded home => byte-identical.
  if (!std::isnan(foot_pin_x_)) brace_foot_x = foot_pin_x_;


  // Left foot is the primary ground anchor during all lean stages.
  // Scale 4x as soon as the arm contacts the table, 5x during leg lift.
  // This is needed because balance residual would otherwise slide the foot
  // to reposition the COM — the arm provides the forward support instead.
  double left_foot_scale = any_arm_contact ? 4.0 : 1.0;

  // 2026-05-20: FK rollout showed the base of support COLLAPSES during
  // forearm_brace — the right foot creeps inward+forward (y -0.16→-0.03,
  // x 0.22→0.38), shrinking the stance from 33cm to 12cm. Anchor the right
  // foot as firmly as the left while braced (×4) so the wide stance holds.
  // 2026-05-26: leg-lift DROPPED → the right foot is NEVER freed; both feet
  // stay grounded, so the anchor below is unconditional. WBC may still nudge
  // foot placement to hold balance.
  double right_foot_scale = any_arm_contact ? 4.0 : 1.0;
  const double rf_ax = brace_foot_x;
  const double rf_ay = kRightFootHomeXY[1];
  const double lf_ax = brace_foot_x;
  const double lf_ay = kLeftFootHomeXY[1];
  // ★ 2026-08-13 STANCE-WIDTH FLOOR (real flat_9): despite the ±0.163
  // y-anchors at 4x, the feet slid to 213 mm apart during the reach — the
  // anchor is symmetric, so both feet drifting inward the same amount is
  // half-price. One-sided width term: when |yL - yR| < `stance_width_min`,
  // push the feet APART through the same y-residual components.
  // `stance_width_gain` 0/absent = OFF = byte-identical.
  double wpush = 0.0;
  {
    int nwm = mj_name2id(model, mjOBJ_NUMERIC, "stance_width_min");
    int nwg = mj_name2id(model, mjOBJ_NUMERIC, "stance_width_gain");
    double wmin = nwm >= 0
        ? model->numeric_data[model->numeric_adr[nwm]] : 0.0;
    double wgain = nwg >= 0
        ? model->numeric_data[model->numeric_adr[nwg]] : 0.0;
    if (wmin > 0.0 && wgain > 0.0) {
      double width = mju_abs(foot_left_pos[1] - foot_right_pos[1]);
      double deficit = wmin - width;
      if (deficit > 0.0) wpush = wgain * deficit;
    }
  }
  residual[counter++] = right_foot_scale * (foot_right_pos[0] - rf_ax);
  residual[counter++] = right_foot_scale * (foot_right_pos[1] - rf_ay) + wpush;
  residual[counter++] = left_foot_scale * (foot_left_pos[0] - lf_ax);
  residual[counter++] = left_foot_scale * (foot_left_pos[1] - lf_ay) - wpush;

  // ----- hip clearance from table front face ----- //
  // penalise the pelvis entering within 0.08m of the slab's front face.
  // ★ 2026-08-01 THE 0.7 WAS STALE BY THREE TABLE GENERATIONS. The old comment
  // ("body x=0.9, half-size 0.5 -> front face 0.40") described a slab that no
  // longer exists: the real one is body x 1.090, half-x 0.59, front face 0.500.
  // `table_surf[0] - 0.7` = 0.390 put the face 110 mm BEHIND where it is, so
  // the pelvis was penalised for advancing toward open air -- fighting the very
  // forward lean the brace needs. Read the half-size FROM THE MODEL so moving
  // the table (lean_set_table.py) can never desynchronise this again.
  double *pelvis_pos_3d = SensorByName(model, data, "pelvis_position");
  const double *table_surf = SensorByName(model, data, "table_surface_pos");
  // ★ 2026-08-01 model-derived (true half-x 0.59). A/B'd: literal 0.7 gave
  // 2/3 falls, this gives 1/3 -- keep the correct geometry.
  int table_top_gid = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
  double table_half_x = (table_top_gid >= 0) ? model->geom_size[3 * table_top_gid] : 0.59;
  double table_front_x = table_surf[0] - table_half_x;
  double hip_penalty = mju_max(0.0, pelvis_pos_3d[0] - (table_front_x - 0.08));
  residual[counter++] = hip_penalty;

  // ----- leg clearance from table front face ----- //
  // Left: check mid-thigh x (midpoint of pelvis and knee) — thigh is at table
  // height during lean so the knee-only check misses it.
  // Right: knee x only (clearance check; right leg now stays grounded too).
  double *left_knee_pos_3d  = SensorByName(model, data, "left_knee_pos");
  double *right_knee_pos_3d = SensorByName(model, data, "right_knee_pos");
  double left_thigh_mid_x   = 0.5 * (pelvis_pos_3d[0] + left_knee_pos_3d[0]);
  double left_thigh_penalty = mju_max(0.0, left_thigh_mid_x    - (table_front_x - 0.05));
  double right_knee_penalty = mju_max(0.0, right_knee_pos_3d[0] - (table_front_x - 0.06));
  residual[counter++] = left_thigh_penalty;
  residual[counter++] = right_knee_penalty;

  // ----- left leg anchor (left knee straight + left foot planted) --------- //
  // The pipeline: right arm braces while BOTH feet stay grounded — leg-lift is
  // DROPPED (2026-05-26), no leg leaves the floor. Keep the left knee straight
  // (up at ~0.42m) AND foot
  // FIRMLY on the ground during every phase that has any contact load on
  // the table.
  // ITER 22 (2026-05-18): foot-lift tolerance tightened 0.05 → 0.02 (only
  // 2 cm float allowed before penalty). With weight bumped 100 → 250 in
  // the XML, a 5 cm heel-lift now costs 250 × (0.03)² = 0.22 vs 0.06
  // previously — small but the gradient is steeper near zero where it
  // matters for keeping the foot pressed down through the lean.
  // ★★ REPURPOSED 2026-08-03 -- SLOT NAME IS STALE. The <user> sensor is still
  // called "Left Leg Anchor" (renaming it would have to be done in BOTH
  // Lean_H12.xml and Lean_H12_Magpie.xml or the writer/model residual counts
  // desync -- the exact 2026-07-30 bug where every later residual hit the WRONG
  // cost). This slot now means FOOT FLATNESS, BOTH FEET.
  //
  // WHY. The old term was `max(0, foot_left_pos[2] - 0.02)` -- the ANKLE BODY
  // ORIGIN's height. When a heel lifts, the foot rotates about the TOE and the
  // ankle origin barely moves (measured: 0.0445..0.0608 over a whole run, a
  // 1.6 cm wiggle against a 0.02 threshold), so the cost was STRUCTURALLY BLIND
  // to heel-lift -- and it could not tell heel-up from toe-up either. Same class
  // as the `foot_left_pos` framepos bug in the Balance polygon.
  //
  // Measured defect it exists to fix (bench20_ladder run02): the RIGHT heel is
  // off the floor from t~31 to t~47 -- the ENTIRE reach and brace hold -- and
  // BOTH heels lift at t=37-39, exactly at peak right-arm reach (728 mm). On
  // its forefeet the foot is free to rotate about the toe, so the ankle has NO
  // restoring authority at all, and the Balance hull is meanwhile crediting
  // heel area that is not in contact.
  //
  // Signal: the foot's own forward axis z-component. Flat => ~0 (measured
  // -0.0000/-0.0008 at a plain stand); heel-up/toe-down => NEGATIVE (-0.084 L,
  // -0.172 R at t=38); toe-up/rocking-back => POSITIVE (+0.104 at t=70, which is
  // the backward-overshoot mode behind the 3 late falls). Penalise BOTH signs so
  // one term guards the forefoot tip AND the heel rock. 0.02 deadband ~ 1.1 deg.
  {
    double const *lf = SensorByName(model, data, "foot_left_forward");
    double const *rf = SensorByName(model, data, "foot_right_forward");
    constexpr double kFlatTol = 0.02;
    double flat = 0.0;
    if (lf) flat += mju_max(0.0, mju_abs(lf[2]) - kFlatTol);
    if (rf) flat += mju_max(0.0, mju_abs(rf[2]) - kFlatTol);
    // keep the old knee-height guard, which was never the broken half
    residual[counter++] =
        flat + (any_arm_contact ? mju_max(0.0, 0.42 - left_knee_pos_3d[2]) : 0.0);
  }

  // ----- right foot: lift during leg-lift, ground during arm-only stages ----- //
  // TEST #11: gentler leg lift. Target z reduced 0.25 → 0.15 (foot home is
  // ~0.029, so this asks for ~12cm clearance). Combined with weight 60→20
  // in XML and strategy ankle target reduced to local_z=-0.75 (world z≈0.15),
  // the leg lift becomes a small backward foot extension instead of a yoga
  // warrior 3 — small enough that weak ankle PD on the support leg can
  // hold balance during the lift.
  // ITER 40 (2026-05-18): right-foot lift target RAMPS smoothly from 0
  // to 0.03 m over kPhaseRampSeconds. User reported iter 39: body
  // "started leg lift and so it tipped to the side" — the step-change in
  // lift demand at phase entry didn't give MPC time to shift CoM laterally
  // over the left (planted) foot before lifting. With the smoothstep ramp,
  // MPC has 1.5 s to gradually move weight to left foot WHILE the lift
  // target grows from 0 to 0.03 m.
  if (any_arm_contact) {
    residual[counter++] = mju_max(0.0, foot_right_pos[2] - 0.02);
  } else {
    residual[counter++] = 0.0;
  }

  // ----- pelvis forward over midfoot ------------------------------------//
  // Forces ankle/torso-strategy lean instead of hip-flex squat. The
  // planner's failure mode without this: to satisfy reach + brace
  // gradients, it flexes the hip and bends the knees, pushing the
  // pelvis BEHIND the feet (sit-back posture). That setup is one tiny
  // perturbation away from a backward fall onto the butt.
  //
  // Residual = max(0, target_x − pelvis_x), where target_x = midfoot_x +
  // 0.05 m. AT target_x or AHEAD → residual = 0 (free). BEHIND target_x
  // → linear penalty, weight 100, sigma 0.05 → cost grows quadratically.
  //
  // Gated on any_arm_contact (phases 2+). Phase 1 is excluded
  // because the home pose already has pelvis 3 cm behind the feet by
  // design — activating this in phase 1 would create constant forward
  // pull that fights the "stay upright" intent.
  if (any_arm_contact) {
    double midfoot_x = 0.5 * (foot_right_pos[0] + foot_left_pos[0]);
    double pelvis_forward_target = midfoot_x + 0.05;
    if (is_forearm_brace) {
      // FOREARM BRACE load-transfer (P3 LOAD): once the forearm is NEAR the table
      // (brace_force_prox_gate ~1) but not yet loaded, drive the pelvis (= CoM
      // proxy) FORWARD toward the forearm to press weight onto it — this is what
      // turns a hovering/light-touch contact into a real load-bearing brace. Back
      // off as force builds (load_deficit 1->0) so the CoM settles JUST BEHIND the
      // forearm centroid (forearm_x - 0.06), loading it without tipping past it.
      // Self-limiting: at the force target the extra pull vanishes (target ->
      // midfoot+0.05). Reach (P4, gated above) only extends after this loads.
      double load_deficit =
          mju_max(0.0, 1.0 - brace_contact_force / mju_max(1.0, desired_brace_force));
      double loaded_target = forearm_site_pos[0] - 0.06;
      pelvis_forward_target =
          midfoot_x + 0.05 +
          brace_force_prox_gate * load_deficit *
              mju_max(0.0, loaded_target - (midfoot_x + 0.05));
    }
    // 2026-08-04 UPPER BOUND (bench20, n=20). This term was ONE-SIDED: it drove
    // the pelvis forward to `pelvis_forward_target` and then charged NOTHING for
    // going further. Combined with Balance reading ~0 during the brace (the
    // load-gated brace vertex extends the support polygon past the CoM, so the
    // capture point projects to itself), there was no sagittal cost opposing
    // forward CoM travel AT ALL while braced. Measured: all 15 braced runs
    // overshot the +0.050 target, max pelvis excursion 0.068..0.202, and that
    // excursion correlates +0.785 with the CoM at release -- which classifies
    // the stand-back almost perfectly (5/6 recover below CoM 0.165, 0/9 above).
    // So: keep the forward drive, add the missing ceiling. `hi` is clamped to be
    // >= `lo` so a far-forward forearm (loaded_target) can never make the band
    // empty and turn this into a constant cost.
    // ★ 2026-08-06: now a NUMERIC so the ceiling can be tuned per gate without a
    // recompile. Default 0.13 = BYTE-IDENTICAL to the constexpr it replaces.
    //   MEASURED (bench20_splayfix + stress sets, n=38): every one of the 25
    //   runs that stood back entered the stand-back ladder with CoM < 150 mm,
    //   while the stalls entered at 146-188. The offline feasible-region map puts
    //   the feet-only limit at 152 mm INDEPENDENTLY. Three lines, one number.
    //   The stalls are ankle-saturated at 98% BECAUSE they enter past it: knee
    //   and hip un-bow identically in both groups, only the ankle stays stuck
    //   (unwinds 1.3 deg vs 13.4), and straightening about a stuck ankle carries
    //   the CoM FURTHER forward (146 -> 205 mm).
    double kPelvisCapFwd = GetNumberOrDefault(0.13, model, "pelvis_cap_fwd");
    // ⛔ 2026-08-06 THE CAP DID NOT BIND. The 08-04 form was
    //     lo = pelvis_forward_target;  hi = mju_max(lo, midfoot + cap);
    // written that way so a far-forward forearm could never make the band empty.
    // But when the load-transfer pull drives `lo` PAST the cap, `hi` becomes `lo`,
    // the band collapses to a point, and the residual degenerates to
    // |pelvis - lo| -- it then ACTIVELY DRIVES the pelvis forward, past the very
    // ceiling it was added to enforce. Raising the weight amplifies the wrong side:
    // gateP at weight 600 entered the ladder at 174 mm (baseline mean 125).
    // FIX: clamp `lo` to the cap so the ceiling always wins and the band can
    // never invert. The forward drive is preserved up to the ceiling, which is
    // all it was ever meant to do.
    double cap_x = midfoot_x + kPelvisCapFwd;
    double lo = mju_min(pelvis_forward_target, cap_x);
    double hi = cap_x;
    double pelvis_band = mju_max(0.0, lo - pelvis_pos_3d[0]) +
                         mju_max(0.0, pelvis_pos_3d[0] - hi);

    // ★★ 2026-08-06: CAP THE CoM, NOT JUST THE PELVIS.
    // MEASURED (gateQ2 run00): with the pelvis correctly held at 114 mm under a
    // 130 mm cap, the CoM still sat at 169 mm -- the CoM runs 45-88 mm AHEAD of
    // the pelvis because the torso and the extended reaching arm are forward of
    // it. So a pelvis ceiling cannot bound the CoM, and the CoM is the variable
    // that decides recovery: all 25 stand-backs entered the ladder below 150 mm,
    // the stalls entered at 146-188, and the offline feasible-region map puts the
    // feet-only limit at 152 mm independently.
    // (This restates a 2026-08-04 note -- "caps the PELVIS, not the CoM" -- that I
    // then spent two gates re-learning. Cap the variable you actually measured.)
    // `com_cap_fwd` numeric: 0 = OFF = BYTE-IDENTICAL default.
    double com_cap = GetNumberOrDefault(0.0, model, "com_cap_fwd");
    double com_over = 0.0;
    if (com_cap > 0.0) {
      int pid_com = mj_name2id(model, mjOBJ_BODY, "pelvis");
      if (pid_com >= 0) {
        double com_x_now = data->subtree_com[3 * pid_com + 0];
        com_over = mju_max(0.0, com_x_now - (midfoot_x + com_cap));
      }
    }
    residual[counter++] = pelvis_band + com_over;
  } else {
    residual[counter++] = 0.0;
  }

  // ----- contact keyframe residual ----- //
  ContactResidual(model, data, residual, &counter);

  // ----- Joint Velocity Limits ------------------------------------------ //
  // One-sided residual that's zero when |qvel| ≤ 0.85·ω_max and grows
  // linearly beyond. ω_max values are the per-joint velocity limits from
  // the H1-2 URDF (h1_2.urdf <limit velocity="..."/>). Stops MJPC from
  // planning trajectories that demand impossible joint speeds — important
  // for sim2real because a plan that exceeds ω_max will fail on hardware
  // regardless of how good the torque is. 0.85 leaves a 15% safety buffer.
  //
  // Index order = actuator order = the ctrlrange order in h1_2_pos.xml.
  // Same indices work for both H1-2 and H1-2-Hands because both variants
  // expose only these 27 actuated joints (fingers are unactuated in our
  // Hands MJCF).
  // DEPLOYED-ENVELOPE PARITY (2026-06-12): the binding limit on hardware is the
  // SAFETY LAYER's velocity estop (default_safety_full.yaml velocity_ratio x URDF,
  // h12_safety_layer/core/joint_limits.py), which latches and cuts ALL motors --
  // three runs in a row died to mid-recovery arm/leg swings the old URDF-only
  // numbers considered free (RshP estop 4.5 rad/s vs old threshold 0.85*9=7.65).
  // Each entry = min(URDF, estop): hips .20/.25/.20*23, knee .60*14, ankles
  // 3.0*9 (URDF binds), torso .15*23, sho_p/r .50*9, sho_y .50*20, elbow .30*20,
  // wrists .30*31.4. The 0.85 factor below keeps a 15% buffer UNDER the estop.
  static constexpr double kJointVelLimit[27] = {
      // L_hip_yaw, L_hip_pitch, L_hip_roll, L_knee, L_ank_p, L_ank_r
      4.6, 5.75, 4.6, 8.4, 9.0, 9.0,
      // R_hip_yaw, R_hip_pitch, R_hip_roll, R_knee, R_ank_p, R_ank_r
      4.6, 5.75, 4.6, 8.4, 9.0, 9.0,
      // torso
      3.45,
      // L_sho_p, L_sho_r, L_sho_y, L_elbow, L_wr_r, L_wr_p, L_wr_y
      4.5, 4.5, 10.0, 6.0, 9.42, 9.42, 9.42,
      // R_sho_p, R_sho_r, R_sho_y, R_elbow, R_wr_r, R_wr_p, R_wr_y
      4.5, 4.5, 10.0, 6.0, 9.42, 9.42, 9.42,
  };
  // Walk actuators (always 27 here) and look up each joint's qvel index via
  // jnt_dofadr — robust to the Hands variant inserting finger joints later
  // in the qvel layout. MuJoCo stores the actuator's transmission target as
  // actuator_trnid[2*i] (first slot is the joint id for joint-type
  // transmissions, which is all of ours).
  for (int i = 0; i < model->nu && i < 27; i++) {
    int jntid  = model->actuator_trnid[2 * i];
    int dofadr = model->jnt_dofadr[jntid];
    double abs_vel = std::abs(data->qvel[dofadr]);
    double threshold = 0.85 * kJointVelLimit[i];
    residual[counter++] = std::max(0.0, abs_vel - threshold);
  }

  // ----- Support Polygon (lateral excursion off reach axis) ------------ //
  // Penalises capture-point excursion perpendicular to the REACH AXIS
  // (midfoot → reach_target). WBC intent: keep CoM moving along the line
  // the body is actually traveling, catch lateral pushes off that line.
  //
  // Earlier attempt anchored the axis to the bracing hand instead of the
  // reach target — but the hand is off to the side (y ≈ -0.3) while the
  // object is straight ahead (y ≈ 0). The midfoot→hand perpendicular
  // direction has a strong x-component, so the body advancing forward
  // toward the object read as "perpendicular excursion" and got penalised,
  // forcing a knees-bent / arm-bent rest pose instead of an extended
  // reach. The reach axis aligns the safe corridor with the direction the
  // body needs to travel; lateral push (the original bug) still costs.
  //
  // Yaw behaviour: midfoot rotates with the feet, reach_target is fixed in
  // world (= object_pos). Body yaw moves midfoot slightly, which tilts the
  // axis slightly, but CoM also moves with the body — perpendicular offset
  // stays small for pure yaw, so the residual is still effectively
  // yaw-invariant under small rotations.
  //
  // Active only when the bracing arm is on the table (any_arm_contact): with no
  // contact there is no support axis to define.
  double sp_residual = 0.0;
  if (any_arm_contact) {
    double midfoot_x = 0.5 * (foot_left_pos[0] + foot_right_pos[0]);
    double midfoot_y = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    double dx = reach_target[0] - midfoot_x;
    double dy = reach_target[1] - midfoot_y;
    double len = mju_sqrt(dx*dx + dy*dy);
    if (len > 1e-6) {
      // Perpendicular (90° CCW of axis), unit length.
      double perp_x = -dy / len;
      double perp_y =  dx / len;
      double offset = (capture_point[0] - midfoot_x) * perp_x +
                      (capture_point[1] - midfoot_y) * perp_y;
      // 5 cm tolerance band — small drift is fine, hard lateral excursion
      // off the reach axis costs. Matches typical capture-point safety
      // buffers in legged-WBC literature.
      static constexpr double kSupportLateralMargin = 0.05;
      sp_residual = mju_max(0.0, mju_abs(offset) - kSupportLateralMargin);
    }
  }
  residual[counter++] = sp_residual;

  // ----- Body Yaw (xy-projected) ---------------------------------------- //
  // The floating-root quaternion has no direct cost in any joint-level
  // residual, so the planner can rigidly yaw the body via foot-on-floor
  // pivot without paying (Waist Yaw + Hip Yaw target joint angles; whole-
  // body rotation leaves those at home). Torso Forward Tilt catches it
  // but is weight 3 AND mixes in pitch — bumping it would also fight the
  // lean. This residual is yaw-only: project torso_forward and reach_dir
  // onto the xy-plane and check alignment there. Pitch drops out because
  // pitching down shortens torso_forward's xy length without changing
  // its xy direction.
  // ★ 2026-08-02 PORT OF THE STABILIZE `body_yaw_feet_ref` FIX (2026-07-17).
  // As written above, this residual aligns the torso with `reach_dir` =
  // normalize(reach_target - torso) -- a WORLD-frame heading that depends on
  // where the reach target happens to sit. stabilize.cc calls that a "+-11 deg
  // world-frame heading lottery" that "steered the calibrated real stand -22 ->
  // +7 deg and dragged the feet", and fixed it there; lean never got the port.
  // It mattered here because lean ships Body Yaw at weight ZERO in every phase,
  // so nothing has ever anchored the floating-root yaw: measured, the robot
  // twists 20 deg of yaw and 9 deg of roll in the 20 s AFTER the stand-up
  // begins (yaw +2.7 at t=60 -> -17.9 at t=80, worst run -31.6), which leaves
  // the two legs in different configurations -- the left ankle then sits 12 deg
  // off its target while the RIGHT ankle lands exactly, and that asymmetry IS
  // the residual bow that stops the robot standing back up square.
  // Enabling the cost as-is would import the lottery. Face the FEET'S OWN MEAN
  // HEADING instead: placement-invariant, no world anchor, the reference
  // FOLLOWS the stance rather than steering it -- same family as the
  // balance_frame and support-polygon fixes.
  // ⚠ MEASURED 2026-08-02: NEITHER of the two existing references works here.
  //  * FEET reference (the stabilize port): torso yaw ~= feet yaw to within
  //    1-8 deg, i.e. the WHOLE ROBOT PIVOTS ON THE FLOOR, feet included. The
  //    reference rotates WITH the failure, so the cost is blind by
  //    construction. Measured effect: yaw -11.7 -> -12.5 deg. Nothing.
  //    (It is still the right instrument for stabilize, whose failure is the
  //    torso twisting against PLANTED feet -- different disease.)
  //  * LEGACY world reference normalize(reach_target - torso) resolves to
  //    -18.2 deg here, because the reach target sits off to the robot's right.
  //    Anchoring to it would pull the stand FURTHER from square than the
  //    -12.5 deg it already drifts to.
  // What a stand-back actually wants is a FIXED heading: end facing where you
  // started. `body_yaw_ref_deg` supplies exactly that (default 0 = +x = the
  // spawn heading), and it is only ever weighted in the release/stand_up
  // phases, so it cannot fight the reach.
  //   body_yaw_feet_ref: 1 = feet mean heading, 0 = use body_yaw_ref_deg.
  {
    int fr_id = mj_name2id(model, mjOBJ_NUMERIC, "body_yaw_feet_ref");
    bool fr = (fr_id >= 0) &&
              (model->numeric_data[model->numeric_adr[fr_id]] > 0.5);
    double const *flf_y = SensorByName(model, data, "foot_left_forward");
    double const *frf_y = SensorByName(model, data, "foot_right_forward");
    if (fr && flf_y && frf_y) {
      double fx = flf_y[0] + frf_y[0], fy = flf_y[1] + frf_y[1];
      double fl = mju_sqrt(fx * fx + fy * fy);
      if (fl > 1.0e-6) {
        reach_dir[0] = fx / fl;
        reach_dir[1] = fy / fl;
        reach_dir[2] = 0.0;
      }
    } else if (!fr) {
      int ry_id = mj_name2id(model, mjOBJ_NUMERIC, "body_yaw_ref_deg");
      double ry = (ry_id >= 0)
          ? model->numeric_data[model->numeric_adr[ry_id]] * (mjPI / 180.0)
          : 0.0;
      reach_dir[0] = std::cos(ry);
      reach_dir[1] = std::sin(ry);
      reach_dir[2] = 0.0;
    }
  }
  double tf_xy_len = mju_sqrt(torso_forward[0] * torso_forward[0] +
                              torso_forward[1] * torso_forward[1]);
  double rd_xy_len = mju_sqrt(reach_dir[0] * reach_dir[0] +
                              reach_dir[1] * reach_dir[1]);
  double yaw_alignment =
      (tf_xy_len > 1e-6 && rd_xy_len > 1e-6)
          ? (torso_forward[0] * reach_dir[0] +
             torso_forward[1] * reach_dir[1]) / (tf_xy_len * rd_xy_len)
          : 1.0;
  residual[counter++] = 1.0 - yaw_alignment;

  // ----- Body must NOT lean on the table (the ARM braces, not the torso) - //
  // User (2026-05-20): in the unbraced lean (mode 2) the robot was resting its
  // pelvis / lower-torso on the tabletop to hold the lean. That free support
  // is what suppressed the emergent counterbalance-arm posture — with the
  // table holding the body up, the planner had no reason to balance the reach
  // with the free arm (it counterbalanced contactlessly before, at this same
  // Posture weight). The pelvis/torso↔table collision excludes are
  // intentionally OFF so the body can't ghost through the edge (R12/test15),
  // but nothing penalised RESTING on it. Penalise the normal contact force
  // between the pelvis or torso and the table so the only thing allowed to
  // bear load on the table is the bracing arm/elbow — forcing a contactless,
  // CoM-balanced lean.
  //
  // GATED to `!any_arm_contact` for the retired hand/palm brace phases: those
  // deep poses brought the body low near the table and an always-on penalty
  // fought the brace establishing.
  // 2026-07-01: RE-ENABLED for the forearm_brace_lean (user: "the only load-
  // bearing places should be the forearm and the legs, nothing else" — the hip
  // must NOT rest on the table). At 0.87 m the forearm brace keeps the pelvis
  // ~x0.25 (well behind the 0.40 edge) so this penalises ONLY an actual pelvis/
  // torso-table graze (0 in the good pose) without fighting the forearm seat.
  //
  // ★★★★★ 2026-07-27: THE ALLOW-LIST ABOVE WAS THE BUG. Two independent holes, both
  // MEASURED with table_contact_ledger.py on plant traces:
  //   (1) It names only `pelvis` + `torso_link`. NOTHING watched the arms, hips or
  //       grippers. In a direct 6->22 brace the ENTIRE RIGHT ARM landed on the wood
  //       (right_forearm_pad 115 N median, right_shoulder_yaw 78 N, jaws, wrist pad)
  //       while the only ALLOWED contact, left_forearm_pad, carried 7.9 N.
  //   (2) The gate `(!any_arm_contact || is_forearm_brace)` DISABLES it whenever an
  //       arm is already touching and the phase is not the brace itself -- i.e. for
  //       ALL of the reach phases. So in a reach->brace ladder the
  //       robot lay down on the table FOR FREE during reach: pelvis 433-521 N, torso
  //       112-148 N, hip links, gripper -- on a 674 N robot, with the forearm at
  //       22-44 N. By the time the brace phase switches this penalty on (weight 150,
  //       ~62,700/step at 438 N) escaping would mean LIFTING 500 N, so it never does.
  // ⇒ Every brace/lean verdict from 2026-07-24..27 -- pad flatness 0.01 deg, 53%
  //   seating, "430 s no fall, drift -0.001", the "drape", CoM-past-toe, the failed
  //   recovery, and the null result from brace_force_target 50->100 -- was measured on
  //   a robot resting on UNAUTHORISED body parts. "Stable" meant "resting on furniture".
  //
  // FIX: `table_contact_exclusive` numeric. 0 = OFF = BYTE-IDENTICAL (legacy behaviour
  // above). >0 = the requirement as actually stated by the user: "the only load-bearing
  // places are the LEFT FOREARM and the two FEET; nothing else may touch the table."
  // Written by EXCLUSION -- every robot geom in contact with the table is charged
  // EXCEPT an explicit allow-list -- because an allow-list of two bodies is how hole (1)
  // happened. Same residual slot / dim / weight as before, so the gRPC + GUI parameter
  // ladder is untouched; only the QUANTITY changes.
  //
  // Applies in every phase: the only phases that ever needed an escape hatch were the
  // legacy hand/palm brace ones, and none of them survive in a live strategy.
  int tce_id = mj_name2id(model, mjOBJ_NUMERIC, "table_contact_exclusive");
  double kTableExclusive =
      (tce_id >= 0) ? model->numeric_data[model->numeric_adr[tce_id]] : 0.0;

  // ---- MODE 2: PROXIMITY BARRIER (2026-07-27) -----------------------------
  // `table_contact_exclusive >= 2` adds a clearance barrier that rises BEFORE contact.
  // WHY: mode 1 charges contact FORCE, which only exists once the body is already
  // down -- and by then escaping is as expensive as the pose itself, so the planner
  // stays. MEASURED: at weight 150 (≈82,000/step at the observed 567 N) a strat-21 run
  // STILL parked its torso on the slab for ~25% of frames. Pricing arrives too late.
  // This term instead charges (margin - clearance) while the part is still above the
  // face, where the escape is cheap. All observed illegal contact was with
  // `table_top_collision` (the top face), never the legs, so a face-clearance barrier
  // is the right instrument. Guarded set = the bodies actually caught offending;
  // the left forearm/wrist pads are exempt because they are the intended contact.
  // 2026-07-28: was [64] with a silent `n_guard < 64` truncation -- adding bodies to
  // kGuardBodies below would have quietly DROPPED the last ones added and the barrier
  // would have looked covered while missing them. 18 bodies = 41 geoms measured, so 96
  // leaves headroom, and overflow now warns instead of truncating in silence.
  static constexpr int kMaxGuard = 96;
  static int guard_g[kMaxGuard];
  static int n_guard = -1;
  static int tabletop_g = -1;
  if (n_guard < 0) {
    n_guard = 0;
    tabletop_g = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
    static const char* kGuardBodies[] = {
        "pelvis", "torso_link", "left_hip_roll_link", "right_hip_roll_link",
        "left_hip_pitch_link", "right_hip_pitch_link", "left_elbow_link",
        "right_elbow_link", "left_wrist_roll_link", "right_wrist_roll_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
        "left_magpie_gripper", "right_magpie_gripper",
        // 2026-07-28 COVERAGE GAP: the shoulder-yaw links were missing. The contact
        // ledger measured `right_shoulder_yaw_link` (geom51) at 214.9 N peak / 140.7 N
        // median ON THE SLAB in prox21_3 -- it was the ONLY illegal body of 9 that this
        // list did not watch, so the barrier could never see it. Both sides added.
        "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        // Shoulder PITCH links also land: `right_shoulder_pitch_link` was the peak
        // illegal part in mx_22_c (375.6 N). Guard them too -- the rule is written by
        // EXCLUSION, so any arm link that can reach the slab belongs here.
        "left_shoulder_pitch_link", "right_shoulder_pitch_link"};
    for (const char* bn : kGuardBodies) {
      int bid = mj_name2id(model, mjOBJ_BODY, bn);
      if (bid < 0) continue;
      for (int g = 0; g < model->ngeom && n_guard < kMaxGuard; g++) {
        if (model->geom_bodyid[g] != bid) continue;
        const char* gn = mj_id2name(model, mjOBJ_GEOM, g);
        if (gn && (std::strcmp(gn, "left_forearm_pad") == 0 ||
                   std::strcmp(gn, "left_wrist_pad") == 0)) continue;
        // ★ 2026-08-03: the REACHING arm's distal links are exempt from the
        // PROXIMITY barrier (this list) but stay in the CONTACT charge below.
        //
        // The barrier is preventive: it charges any guarded geom that comes within
        // `table_clear_margin` (0.08 m) of the slab face while over the footprint.
        // The right gripper/wrist/elbow were in it -- and the reach target sits at
        // z 1.035 against a face at 0.955, i.e. EXACTLY 0.080 above it. So the
        // barrier (w 150) fired precisely where Reaching Hand Dist (w 80) was
        // pulling, ~2:1 against it. Measured consequence (gateB run01): the right
        // gripper never crossed the slab's near edge (x 0.500), sat 18-21 cm BELOW
        // the face for the whole run, and stayed 0.513 m from a target its arm
        // could reach with 5-11 cm to spare. The reach simply never happened.
        //
        // Exempting only the DISTAL right links keeps the intent intact: the right
        // SHOULDER links stay guarded (they should never be over the slab -- one was
        // the peak illegal part at 375.6 N in mx_22_c), and every right-arm geom is
        // still charged by `body_table_force` if it actually TOUCHES. So the hand may
        // hover over the table to do its job; it still may not rest on it.
        static const char* kReachExempt[] = {
            "right_magpie_gripper", "right_wrist_yaw_link",
            "right_wrist_roll_link", "right_wrist_pitch_link", "right_elbow_link"};
        bool reach_exempt = false;
        for (const char* rn : kReachExempt)
          if (std::strcmp(bn, rn) == 0) { reach_exempt = true; break; }
        if (reach_exempt) continue;
        guard_g[n_guard++] = g;
      }
      if (n_guard >= kMaxGuard) {
        std::fprintf(stderr,
                     "[lean] table_contact_exclusive: guard cache FULL at %d geoms -- "
                     "bodies from '%s' onward are NOT guarded. Raise kMaxGuard.\n",
                     kMaxGuard, bn);
        break;
      }
    }
  }
  double body_table_prox = 0.0;
  if (kTableExclusive >= 2.0 && tabletop_g >= 0) {
    int tpb = model->geom_bodyid[tabletop_g];
    const double* tc = data->xpos + 3 * tpb;   // slab body origin
    double thx = model->geom_size[3 * tabletop_g + 0];
    double thy = model->geom_size[3 * tabletop_g + 1];
    double face_z = data->geom_xpos[3 * tabletop_g + 2] +
                    model->geom_size[3 * tabletop_g + 2];
    int bm_id = mj_name2id(model, mjOBJ_NUMERIC, "table_clear_margin");
    double kMargin = (bm_id >= 0)
        ? model->numeric_data[model->numeric_adr[bm_id]] : 0.08;
    if (kMargin <= 0.0) kMargin = 0.08;
    for (int gi = 0; gi < n_guard; gi++) {
      int g = guard_g[gi];
      const double* gp = data->geom_xpos + 3 * g;
      double rad = model->geom_size[3 * g];      // conservative radius
      // only charge parts that are HORIZONTALLY over the slab footprint
      if (mju_abs(gp[0] - tc[0]) > thx + rad) continue;
      if (mju_abs(gp[1] - tc[1]) > thy + rad) continue;
      double clear = (gp[2] - rad) - face_z;
      if (clear < kMargin) body_table_prox += (kMargin - clear);
    }
  }

  double body_table_force = 0.0;
  if (kTableExclusive > 0.0) {
    int table_bid = mj_name2id(model, mjOBJ_BODY, "table");
    // ALLOW-LIST BY GEOM: only these may bear on the table. The feet never reach it,
    // so they need no entry; if a future scene puts the feet on the table, add them.
    static const char* kAllowed[] = {"left_forearm_pad", "left_wrist_pad"};
    // ★ 2026-07-31: ALLOW THE FOREARM'S REAL SHELL, NOT JUST THE INERT PAD.
    // The pads are proxies; the load-bearing surface on hardware is the forearm
    // itself. Measured: the pad capsule protrudes below the real shell by AT MOST
    // 6.12 mm, and over 44% of the wrist-roll band the shell is 0.5-24.5 mm BELOW
    // it -- so the shell, not the pad, is what actually reaches the wood first.
    // With the table<->arm <exclude>s removed (the faithful model) a name-only
    // allow-list therefore charges the ONLY geoms that can physically touch, and
    // the planner correctly refuses to brace at all: measured 0/3 forearm load,
    // 0 N of contact of any kind, all three runs toppling backward. The arm-chain
    // meshes are UNNAMED, so allow by BODY.
    // Spec unchanged in substance -- brace = forearm + feet; hand and wrist stay
    // illegal. `left_shoulder_yaw_link` is included because its DISTAL bulge is the
    // elbow housing, which is part of the forearm contact patch, not the upper arm.
    int allow_b1 = mj_name2id(model, mjOBJ_BODY, "left_elbow_link");
    int allow_b2 = mj_name2id(model, mjOBJ_BODY, "left_shoulder_yaw_link");
    for (int ci = 0; ci < data->ncon; ci++) {
      const mjContact* con = &data->contact[ci];
      int b1 = model->geom_bodyid[con->geom1];
      int b2 = model->geom_bodyid[con->geom2];
      bool table1 = (b1 == table_bid), table2 = (b2 == table_bid);
      if (table1 == table2) continue;                 // not a robot<->table contact
      int rg = table1 ? con->geom2 : con->geom1;       // the ROBOT-side geom
      const char* gn = mj_id2name(model, mjOBJ_GEOM, rg);
      bool allowed = false;
      if (gn) {
        for (const char* a : kAllowed)
          if (std::strcmp(gn, a) == 0) { allowed = true; break; }
      }
      int rb = model->geom_bodyid[rg];
      if ((allow_b1 >= 0 && rb == allow_b1) || (allow_b2 >= 0 && rb == allow_b2))
        allowed = true;                                 // the forearm's real shell
      if (allowed) continue;
      mjtNum f6[6];
      mj_contactForce(model, data, ci, f6);
      body_table_force += mju_abs(f6[0]);             // normal-force magnitude (N)
    }
  } else if (!any_arm_contact || is_forearm_brace) {
    int pelvis_bid = mj_name2id(model, mjOBJ_BODY, "pelvis");
    int torso_bid  = mj_name2id(model, mjOBJ_BODY, "torso_link");
    int table_bid  = mj_name2id(model, mjOBJ_BODY, "table");
    for (int ci = 0; ci < data->ncon; ci++) {
      const mjContact* con = &data->contact[ci];
      int b1 = model->geom_bodyid[con->geom1];
      int b2 = model->geom_bodyid[con->geom2];
      bool body_side  = (b1 == pelvis_bid || b1 == torso_bid ||
                         b2 == pelvis_bid || b2 == torso_bid);
      bool table_side = (b1 == table_bid || b2 == table_bid);
      if (body_side && table_side) {
        mjtNum f6[6];
        mj_contactForce(model, data, ci, f6);
        body_table_force += mju_abs(f6[0]);  // normal-force magnitude (N)
      }
    }
  }
  // prox term is in METRES; scale it into the same newton-ish range as the force
  // term so one weight governs both (100 N per metre of incursion).
  residual[counter++] = body_table_force + 100.0 * body_table_prox;

  // ----- Knees straight (retired) --------------------------------------- //
  // This pair only ever fired during the leg-lift phase, which was dropped
  // 2026-05-26. Held at zero so the residual slot / dim / weight ladder (and
  // therefore the gRPC + GUI parameter indices) stay exactly where they are.
  residual[counter++] = 0.0;
  residual[counter++] = 0.0;

  // ----- Leg LEFT/RIGHT symmetry (kill the one-knee strut) -------------- //
  // The recurring free-standing failure (stand/crouch/squat) is a one-leg
  // "strut": one knee locks near its lower stop as a passive support while the
  // other balances, then both sink and the robot topples (~70-160 s live). By
  // construction the strut is a large LEFT/RIGHT difference in the sagittal leg
  // joints (knee + hip-pitch). No existing term penalises that DIFFERENCE:
  // Posture penalises each leg's ABSOLUTE deviation from the keyframe, so a
  // matched bent stance and a strut can cost nearly the same in Posture while
  // the strut wins on Balance (offloading one leg to a free passive column is
  // genuinely lower-cost). This quadratic penalty on (L - R) makes the
  // asymmetric strut strictly more expensive than a symmetric stance WITHOUT
  // pinning the absolute pose: the legs may still flex *together* for balance,
  // and tiny micro-asymmetries stay nearly free (quadratic gradient -> 0 at 0)
  // so it does not over-constrain the lateral ankle/hip balance.
  //   knee_L = qpos[7+3], knee_R = qpos[7+9]; hipPitch_L = qpos[7+1],
  //   hipPitch_R = qpos[7+7] (verified vs body-tree joint order + Knees
  //   Straight above). Only the sagittal pair: ankle is the primary balance
  //   actuator (asymmetry there is normal), and hip-roll/yaw are mirror-signed
  //   (the widened stance is +/-0.12), so neither belongs in an (L - R) term.
  // GATED to free-standing (!any_arm_contact): leg-lift / lean / retrieve
  // legitimately stand on one leg, so symmetry is zeroed there. With the XML
  // default weight 0 the term is OFF for every strategy that does not opt in
  // via its JSON weight map ("Symmetry": w), so all other tasks stay
  // byte-identical (zero weight AND/OR zero residual = zero cost).
  if (!any_arm_contact) {
    residual[counter++] = data->qpos[7 + 3] - data->qpos[7 + 9];  // knee L-R
    residual[counter++] = data->qpos[7 + 1] - data->qpos[7 + 7];  // hipPitch L-R
    // anklePitch L-R: the SAGITTAL ankle. joint_forensics on the cold-start
    // backward fall (strat 19) showed the one-leg strut hides HERE — L/R
    // ankle_pitch diverged 13.9deg while knee diverged only 5.2deg. The
    // original term excluded "ankle" reasoning about the LATERAL ankle_roll
    // (asymmetry there is normal); ankle_PITCH is fore/aft and must stay
    // symmetric for a square stance, so penalising its (L-R) closes the
    // strut's last escape. Quadratic => tiny asymmetries stay ~free.
    residual[counter++] = data->qpos[7 + 4] - data->qpos[7 + 10]; // anklePitch L-R
  } else {
    // ★ 2026-08-13 LEFT-SINK FIX (real flat_9/10): with symmetry hard-zeroed
    // during arm contact, the planner leans toward the brace by SINKING the
    // left knee (kneeL-kneeR hit +0.45..+0.50: run 9 froze the un-bow on one
    // buried leg; run 10 ran away into a leftward fall at dive commit).
    // Keep DEADBANDED knee + hip-pitch (L-R) terms alive while braced:
    // `brace_knee_sym` gain (0 = OFF = old hard-zero), `brace_knee_sym_db`
    // deadband (rad, default 0.30) keeps transient leg-lift shuffles and
    // small equilibrium asymmetry free while outlawing the sustained sink.
    // Ankle pitch stays exempt during contact (feet do the balance work).
    double ksym = 0.0, kdb = 0.30;
    int nks = mj_name2id(model, mjOBJ_NUMERIC, "brace_knee_sym");
    if (nks >= 0) ksym = model->numeric_data[model->numeric_adr[nks]];
    int nkd = mj_name2id(model, mjOBJ_NUMERIC, "brace_knee_sym_db");
    if (nkd >= 0) kdb = model->numeric_data[model->numeric_adr[nkd]];
    auto shrink = [&](double v) {
      double ex = mju_abs(v) - kdb;
      return ex > 0.0 ? (v > 0.0 ? ex : -ex) : 0.0;
    };
    if (ksym > 0.0) {
      residual[counter++] =
          ksym * shrink(data->qpos[7 + 3] - data->qpos[7 + 9]);
      residual[counter++] =
          ksym * shrink(data->qpos[7 + 1] - data->qpos[7 + 7]);
      residual[counter++] = 0.0;
    } else {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
  }

  // ----- Base-height anchor (free-standing anti-sink) ------------------- //
  // After the strut is removed (Symmetry), the residual failure is a SYMMETRIC
  // CoM SINK: the planner lowers the base for stability (Balance rewards a low
  // CoM) with nothing holding standing height, so stand slowly crouches and
  // topples (~76s live), and the squat sinks instead of re-rising on its
  // ascent. Anchor the base height: penalise the base dropping BELOW the active
  // phase keyframe's base z (posture_target[2]) -- a CoM-height proxy that does
  // NOT freeze individual joints (the planner picks HOW to hold height), unlike
  // stiffening the knee posture (leg_extension_gain 5.5 OVER-stiffened and broke
  // stand in the headless gate). One-sided (max(0, target - z)): only the SINK
  // is penalised, never being tall, so it adds no cost while held. Scaled x10 so
  // a metre-deviation maps to a joint-angle-like residual (sane JSON weights).
  //   STAND: posture_target[2] = home key z = 1.028 = correct standing height.
  //   CROUCH/SQUAT: all keyframes share the nominal base z 1.028 (the real
  //   bent-leg height comes from FK, not qpos[2]), so this anchor is WRONG for
  //   them -> kept OFF via JSON weight 0 (crouch already holds; squat-ascent
  //   needs a per-phase FK-derived target, a follow-on). Gated free-standing;
  //   brace/lean use the existing Height term -> byte-identical.
  if (!any_arm_contact) {
    double bh_err = posture_target[2] - data->qpos[2];
    residual[counter++] = 10.0 * (bh_err >= 0.0 ? bh_err : 0.0);
  } else {
    // ★ 2026-08-13 DRAPE FIX: this anchor used to be HARD 0 while the pad
    // touched -- w450 x 0.0 -- so the flat-brace press could fold the legs
    // and sink the base to z 0.30-0.55 at zero cost ("drape", 5/10 in
    // final10; braces at 400-860 N carrying the folded body). The brace
    // keyframes carry real base-z targets (lean 0.964 / mid 0.885), so keep
    // the SAME one-sided anchor alive during contact with a slack band
    // (`brace_sink_slack`, m): normal brace depth (z 0.87-0.95) stays free,
    // only the fold below target-slack is charged. slack <= 0/absent = OFF =
    // the old hard-0 behavior.
    double sink_res = 0.0;
    int nsl = mj_name2id(model, mjOBJ_NUMERIC, "brace_sink_slack");
    double slack = nsl >= 0
        ? model->numeric_data[model->numeric_adr[nsl]] : 0.0;
    if (slack > 0.0) {
      double bh_err = (posture_target[2] - slack) - data->qpos[2];
      if (bh_err > 0.0) sink_res = 10.0 * bh_err;
    }
    residual[counter++] = sink_res;
  }

  // ----- Centroidal angular momentum (hip / Horak-Nashner strategy) ----- //
  // Regulating whole-body angular momentum about the CoM toward zero gives the
  // sampling planner the active whole-body ("hip") balance strategy that the
  // ankle (capture-point Balance) and the static L/R Symmetry term do not: it
  // penalises the TRANSIENT rotational mode that lets an incipient tip grow into
  // the one-leg passive-column "strut", so the planner instead nulls momentum
  // with a coordinated hip/trunk rotation and can RETRIEVE to symmetric when
  // nudged. Pairs with Balance (linear-momentum/ankle) and Symmetry (static
  // pose) -- it is the dynamic rotational complement, NOT a replacement.
  // SAMPLING-LEGAL: penalises only the STATE L_cm over the rollout horizon --
  // no feedback law, no Jacobian, no dL/dt. data->subtree_angmom is ALREADY
  // populated by the forward pass: mj_sensorVel calls mj_subtreeVel because the
  // model has <subtreelinvel> sensors (the Balance term above reads
  // torso_subtreelinvel), and that one call fills subtree_linvel AND
  // subtree_angmom for every body. So the read is O(0) and thread-safe
  // (read-only of this rollout thread's own mjData; NO mj_forward). "pelvis" is
  // the floating-base root body, so its subtree == the whole robot and
  // subtree_angmom[3*pelvis] == the centroidal angular momentum [Lx,Ly,Lz]
  // about the whole-body CoM (world frame, kg.m^2/s). Scaled x0.1 so a fall-rate
  // momentum (~10) maps to an O(1) residual. Gated free-standing so
  // brace/lean/retrieve stay byte-identical; default XML weight 0 (opt-in via
  // strategy JSON key "Angular Momentum").
  if (!any_arm_contact) {
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pelvis_id < 0) pelvis_id = 1;  // floating-base root fallback (always exists)
    const mjtNum *angmom = data->subtree_angmom + 3 * pelvis_id;
    double Lx_tgt = 0.0, Ly_tgt = 0.0;
    // 2026-08-04 REVIVED. This was a hard-coded 0.0, so all three residuals were
    // identically zero while h12_simple_forearm_brace.json carried
    // "Angular Momentum": 5.0 in ALL EIGHT phases -- the JSON weight multiplied a
    // zeroed residual and the term has been inert for the entire gate series.
    // Same bug, same fix as stabilize.cc:2651 (which was trot-gated there): set the
    // carrier to 1.0 and let the JSON weight do the scaling. That fix produced the
    // first stable non-collapsing real stand (91.7 s, 2026-06-30).
    // WHY IT MATTERS HERE: regulating centroidal angular momentum IS the hip
    // strategy, and Koolen's capturability analysis identifies the reaction-mass
    // model -- the one WITH angular momentum -- as having a LARGER capture region
    // than the ankle-only one. Our 2*tau/W bound is the ankle-only bound, which is
    // exactly what a zeroed angmom term describes. With full parity the ankle lost
    // authority, so the hip strategy is the lever that is supposed to cover it.
    // Gate is unchanged (!any_arm_contact), so this switches on for the later
    // standback rungs and the final stand -- where the un-bow has to complete.
    double angmom_w = 1.0;
    residual[counter++] = angmom_w * 0.1 * (angmom[0] - Lx_tgt);
    residual[counter++] = angmom_w * 0.1 * (angmom[1] - Ly_tgt);
    residual[counter++] = angmom_w * 0.1 * angmom[2];
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

  // ----- Lateral CoM centering (the frontal-plane balance Balance OMITS) ----- //
  // ROOT CAUSE of the systematic rightward lean (2026-06-09 live trace): the
  // capture-point Balance term projects the capture point onto the support
  // polygon, which for free-standing is the {L_foot, R_foot} LINE. A lateral
  // CoM shift TOWARD a foot stays ON that line -> projects to itself -> residual
  // ~ 0 -> UNPENALISED. So Balance constrains fore-aft excursion and polygon
  // EXIT, but lateral CoM CENTERING between the feet is structurally free. A
  // sub-threshold sideways seed then drifts (unopposed) until the CoM sits over
  // one foot: that leg loads and its knee buckles while the unloaded leg locks
  // straight into the passive "strut" -- a positive feedback that the live trace
  // shows as roll +0.6 -> +3 deg with the right hip_roll torque 2.3x the left.
  // This term closes that gap: penalise the torso-subtree CoM's LATERAL (y)
  // offset from the foot midpoint (same CoM proxy + foot sensors Balance already
  // uses, all fetched above). Quadratic -> gradient 0 at centre, so it opposes a
  // SUSTAINED drift without fighting a legitimate transient lateral push-recovery
  // or the normal ankle/hip-roll micro-corrections. x10 so a ~metre maps to a
  // joint-angle-like residual (sane JSON weights, matches Base Height). Gated
  // free-standing (leg-lift / lean / retrieve legitimately stand off-centre);
  // XML default weight 0 -> OFF unless a strategy opts in via JSON "Lateral
  // Center", so every other task stays byte-identical.
  // ★★★ 2026-08-09 LATERAL CENTERING DURING THE BRACE (`lateral_center_braced`
  // numeric; 0 = OFF = byte-identical). MEASURED on the twin (bench20 v5+v6,
  // n=32-40): the BRACE-SIDE (left) foot moves 189 mm median laterally vs 61 mm
  // for the right, and it UNLOADS 56% median across the brace while the right
  // foot picks the load up. Unload predicts slide (Spearman rho +0.31, p=0.041;
  // the high-unload half slides 112 mm further, p=0.029). That is EXACTLY the
  // strut/buckle feedback this term was written to kill -- but the
  // `!any_arm_contact` gate switches it OFF the instant the forearm seats, i.e.
  // precisely when the brace starts pulling load off the ipsilateral leg. The
  // gate's original rationale (leg-lift / lean / retrieve legitimately stand
  // off-centre) still holds for those keyframes, which simply carry weight 0.
  // Fore-aft lean is untouched: this residual is the y axis only.
  // MODES (`lateral_center_braced`): 0 = OFF = byte-identical (residual zeroed on
  // arm contact, the shipped behaviour). 1 = live during the brace, target = foot
  // midpoint. 2 = live during the brace, target = the ROLL-EQUILIBRIUM point.
  //
  // ★★★ WHY 2 EXISTS -- mode 1 was benched (n=5) and made the slide WORSE
  // (|dy_left| 275 mm vs 189 mm baseline), because the midpoint target is simply
  // WRONG while a brace carries load. Measured on 20 braced frames: CoM y - mid
  // sat at -2 mm, i.e. the CoM was ALREADY centred, so mode 1 had nothing to
  // correct. Roll equilibrium about the foot midpoint with three contacts,
  //     F_b*y_b + F_L*y_L + F_R*y_R = W*y_c,
  // set F_L = F_R over a symmetric stance (y_L = -y_R) and the equal-load CoM is
  //     y_c* = F_b * (y_b - y_mid) / W                     [+19 mm, measured]
  // NOT the midpoint. Driving y_c to 0 therefore commands a 25 mm roll error that
  // pushes ~52 N off the brace-side foot -- most of the measured +-81 N split
  // (left 187 N vs right 350 N) -- and an unloaded foot has no friction budget,
  // so it slides. Model sanity on the same frames: contacts sum 670 N vs W 674 N,
  // 3-point moment residual 8 N.m (ankle roll torque / finite sole width).
  double lat_mode = 0.0;
  {
    int lcb = mj_name2id(model, mjOBJ_NUMERIC, "lateral_center_braced");
    if (lcb >= 0) lat_mode = model->numeric_data[model->numeric_adr[lcb]];
  }
  // ★ 2026-08-09 MODE 2 REFINEMENT: only steer laterally while the brace is
  // GENUINELY LOADED. Without this, mode 2 degenerates into mode 1 exactly where
  // mode 1 hurts: through `forearm_brace_release` and the `standback_r*` rungs the
  // arm may still graze the slab while carrying ~0 N, so the equilibrium shift
  // collapses to 0 and the residual reverts to the midpoint target -- steering the
  // CoM at the moment the stand-back needs it left alone. n=20 vs the
  // estimator-in-loop control, mode 2 traded falls for feet (never-fell 2/20 vs
  // 5/20) and the stand-back is this pipeline's known-marginal phase.
  bool brace_loaded = any_arm_contact && brace_contact_force > 5.0;
  if (!any_arm_contact || (lat_mode >= 2.0 ? brace_loaded : lat_mode > 0.0)) {
    double midfoot_y = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    double lat_tgt = midfoot_y;
    if (lat_mode >= 2.0 && brace_loaded) {
      int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
      double Wn = (pid >= 0 ? model->body_subtreemass[pid] : 0.0) *
                  mju_abs(model->opt.gravity[2]);
      // pelvis subtree = ROBOT ONLY; mj_getTotalmass would also count the TABLE
      // (2308 N vs the robot's 674 N) and shrink the correction by 3.4x.
      if (Wn > 1.0) {
        double shift = brace_contact_force * (bracing_hand[1] - midfoot_y) / Wn;
        // clamp to half the stance half-width: past that the "equal load" target
        // would sit outside the feet and stop being a balance point at all.
        double lim = 0.5 * mju_abs(foot_left_pos[1] - foot_right_pos[1]) * 0.5;
        lat_tgt += mju_max(-lim, mju_min(lim, shift));
      }
    }
    residual[counter++] = 10.0 * (subcom[1] - lat_tgt);
  } else {
    residual[counter++] = 0.0;
  }

  // --- Brace Wrist Cock (dim 1, 2026-07-29) -----------------------------------
  // WHY THIS COST EXISTS. In the forearm brace the LEFT FOREARM must be the only
  // non-foot part bearing load (hand and wrist are BOTH illegal: loading the wrist
  // can pop the gripper open, and the wrist load path binds at 52 N because it puts
  // all three wrist joints in front of the elbow, versus 110 N through the forearm).
  // At the `forearm_brace_lean` keyframe the ordering is already correct -- forearm
  // pad bottom 0.862 vs wrist pad bottom 0.874, so the forearm lands 12 mm first.
  // MEASURED over 18 twin runs the ordering INVERTS: the elbow tracks ~12 deg
  // straighter and the shoulder ~10 deg lower than commanded, dropping the wrist
  // BELOW the forearm, and `left_wrist_pad` then takes essentially all the load
  // (370-1474 contact frames) while the forearm hovers ~23 mm clear.
  // Cocking `left_wrist_yaw` to -50 deg lifts wrist clearance 8.7 -> 40.3 mm and the
  // gripper 19.7 -> 71.9 mm while leaving forearm seating IDENTICAL, because the
  // forearm pad sits on `left_elbow_link` -- PROXIMAL to the wrist, so a wrist DOF
  // cannot move it. Putting that angle in the keyframe was NOT enough: it was tracked
  // to only -8 deg of -50, because NO existing term owns this DOF. `Brace Pos` (w120)
  // constrains the forearm SITE POSITION and leaves wrist yaw free, and `Posture` is
  // too diffuse -- raising it to 120 made things WORSE (the pad stopped reaching the
  // slab at all). Hence a dedicated, phase-gated term rather than another weight.
  // ⚠ Name-gated to `forearm_brace_lean` ONLY: the reach_to_target phases brace
  // with the HAND and must keep their wrist free.
  // Default `brace_wrist_cock_deg` 0 => residual identically 0 => BYTE-IDENTICAL.
  {
    double wrist_cock_err = 0.0;
    int bwc_id = mj_name2id(model, mjOBJ_NUMERIC, "brace_wrist_cock_deg");
    double cock_deg =
        (bwc_id >= 0) ? model->numeric_data[model->numeric_adr[bwc_id]] : 0.0;
    if (is_forearm_brace && cock_deg != 0.0) {
      // reach_right == true means the RIGHT arm reaches, so the LEFT arm braces.
      int j = mj_name2id(model, mjOBJ_JOINT,
                         reach_right ? "left_wrist_yaw_joint"
                                     : "right_wrist_yaw_joint");
      if (j >= 0) {
        double tgt = cock_deg * (M_PI / 180.0);
        // Clamp the target into the joint's own range so a bad numeric cannot ask
        // for an unreachable angle -- that is the "unattainable target" failure
        // mode that killed every other lean stage (residual never nulls, its
        // permanent slope rectifies into a body drift).
        tgt = mju_min(model->jnt_range[2 * j + 1],
                      mju_max(model->jnt_range[2 * j], tgt));
        wrist_cock_err = data->qpos[model->jnt_qposadr[j]] - tgt;
      }
    }
    residual[counter++] = wrist_cock_err;
  }

  // --- Brace Arm Plane (dim 1, 2026-08-03) ------------------------------------
  // Keep shoulder -> elbow -> wrist roughly COPLANAR and pointing forward, so the
  // brace load line runs along the arm instead of across it.
  //
  // MEASURED DEFECT (bench20_ladder run02, torso frame): during the reach the
  // forearm splays 20-30 deg out to the robot's LEFT (heading +29.6 deg at t=35)
  // while the elbow tucks up to 35 mm INBOARD of the shoulder -- a "chicken wing".
  // Two consequences: (1) the wrist wanders to y=+0.288 against a slab edge at
  // +0.297, i.e. 9 mm from sliding off the SIDE of the table, which is the same
  // failure as 2026-08-02 creeping back; (2) the contact force then acts on a
  // LATERAL moment arm about the shoulder, and shoulder_roll is already the
  // largest single term in the load path (0.1510 N.m per N for a forearm
  // contact). Straightening the arm shrinks that arm and raises the 198 N ceiling.
  //
  // Two terms, both in the TORSO frame so torso yaw does not leak in:
  //   splay  = |atan2(dy, dx)| of the elbow->wrist vector   (forearm points fwd)
  //   tuck   = lateral offset of the elbow from the shoulder->wrist chord
  // Deadbands keep the natural brace pose free; only the excursion is charged.
  // ★ Read the BODY FRAMES directly -- there are no left_shoulder_pos /
  // left_elbow_pos / left_wrist_pos sensors in this model (they belong to the
  // H12_Hands variant). Using SensorByName here would return null, the guard
  // would zero the term, and the cost would be silently INERT.
  // ⚠ These are body ORIGINS, which for these links sit at the driving joint --
  // fine for a heading/coplanarity measure, but do NOT reuse them as surface
  // points (the `foot_left_pos`-is-the-ankle-origin trap).
  {
    int b_sh = mj_name2id(model, mjOBJ_BODY, "left_shoulder_yaw_link");
    int b_el = mj_name2id(model, mjOBJ_BODY, "left_elbow_link");
    int b_wr = mj_name2id(model, mjOBJ_BODY, "left_wrist_yaw_link");
    int b_to = mj_name2id(model, mjOBJ_BODY, "torso_link");
    double plane_err = 0.0;
    if (b_sh >= 0 && b_el >= 0 && b_wr >= 0 && b_to >= 0) {
      double const *T = data->xpos + 3 * b_to;
      double const *R = data->xmat + 9 * b_to;
      double sl[3], el_[3], wl[3], tmp[3];
      mju_sub3(tmp, data->xpos + 3 * b_sh, T); mju_mulMatTVec(sl, R, tmp, 3, 3);
      mju_sub3(tmp, data->xpos + 3 * b_el, T); mju_mulMatTVec(el_, R, tmp, 3, 3);
      mju_sub3(tmp, data->xpos + 3 * b_wr, T); mju_mulMatTVec(wl, R, tmp, 3, 3);

      constexpr double kSplayTol = 0.17;   // rad, ~10 deg of free heading
      constexpr double kTuckTol  = 0.03;   // m, 3 cm of free lateral offset
      double splay = std::atan2(wl[1] - el_[1], wl[0] - el_[0]);
      plane_err += mju_max(0.0, mju_abs(splay) - kSplayTol);

      // lateral distance of the elbow from the shoulder->wrist chord, in xy
      double cx = wl[0] - sl[0], cy = wl[1] - sl[1];
      double cn = mju_sqrt(cx * cx + cy * cy);
      if (cn > 1.0e-6) {
        double px = el_[0] - sl[0], py = el_[1] - sl[1];
        double lat = mju_abs((px * cy - py * cx) / cn);
        plane_err += mju_max(0.0, lat - kTuckTol);
      }

      // ★ SLAB CONTAINMENT (2026-08-04). The two terms above straighten the arm
      // but say nothing about WHERE it lands, and straightening turned out to
      // push the contact OUTBOARD: the forearm now lies parallel to x at the
      // shoulder's lateral offset instead of angling across the slab. Measured
      // over the brace window, contact y went 0.194 -> 0.289 and 0.180 -> 0.332
      // (slab edge +0.297) WITH this cost, versus 0.217 -> 0.135 and 0.232 ->
      // 0.142 without it. Base lateral drift (+0.07 m left) stacks on top.
      // So charge the bracing elbow for leaving the slab footprint, in WORLD y
      // against the actual table geom -- the objective is "stay on the table",
      // which neither an arm-frame nor a torso-frame term can express.
      int g_tab = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
      int b_el2 = mj_name2id(model, mjOBJ_BODY, "left_elbow_link");
      if (g_tab >= 0 && b_el2 >= 0) {
        double ty = data->geom_xpos[3 * g_tab + 1];
        double half_y = model->geom_size[3 * g_tab + 1];
        constexpr double kEdgeKeepout = 0.06;   // m of slab we refuse to use
        double dy = mju_abs(data->xpos[3 * b_el2 + 1] - ty);
        plane_err += mju_max(0.0, dy - (half_y - kEdgeKeepout));
      }

      // ★ BRACE FLAT (2026-08-13). Real runs 14/15: at contact the commanded
      // forearm is inclined 24-33 deg wrist-end-up, so only the elbow-end
      // CORNER of the pad meets the table -- right at the front edge -- and
      // it chatters (23-25 contact-LOST resets/run) and slips. Nothing above
      // constrains PITCH: splay/tuck live in the torso xy-plane, Brace Pos is
      // a point target a corner can satisfy. Sim never punishes this (twin
      // seats at +7..+21 deg and still pulls 100-160 N from a corner), so the
      // term must be model-state, not contact-mediated: charge the WORLD
      // elevation of elbow->wrist once the pad is near slab height. Deadband
      // `brace_flat_tol` (rad, default 0.09 ~ 5 deg; keyframe design is +3).
      // Activation ramps in over the last 15 cm of descent so the swing-down
      // arc far above the table stays free.
      if (g_tab >= 0) {
        int g_pad2 = mj_name2id(model, mjOBJ_GEOM, "left_forearm_pad");
        int b_wr2 = mj_name2id(model, mjOBJ_BODY, "left_wrist_roll_link");
        if (g_pad2 >= 0 && b_el2 >= 0 && b_wr2 >= 0) {
          int nflat = mj_name2id(model, mjOBJ_NUMERIC, "brace_flat_tol");
          double flat_tol = nflat >= 0
              ? model->numeric_data[model->numeric_adr[nflat]] : 0.09;
          // ★ iter-2 (gate_braceflat 1-5): at gain 1 the press wins — seats
          // start +6..+12 deg then the Brace Force gradient (|r| to 37 at
          // w40) rolls them to +20..+23 deg; a 0.1-0.3 rad excess inside a
          // shared-w300 term is ~78 cost units, not enough. `brace_flat_gain`
          // scales the excess so flatness competes with the press.
          int ngain = mj_name2id(model, mjOBJ_NUMERIC, "brace_flat_gain");
          double flat_gain = ngain >= 0
              ? model->numeric_data[model->numeric_adr[ngain]] : 1.0;
          double surf_z2 = data->geom_xpos[3 * g_tab + 2] +
                           model->geom_size[3 * g_tab + 2];
          double pad_z2 = data->geom_xpos[3 * g_pad2 + 2];
          constexpr double kFlatEngage = 0.25;  // m above slab: ramp start
          double act = mju_clip(
              1.0 - (pad_z2 - surf_z2) / kFlatEngage, 0.0, 1.0);
          if (act > 0.0 && flat_tol > 0.0) {
            double fx = data->xpos[3 * b_wr2 + 0] - data->xpos[3 * b_el2 + 0];
            double fy = data->xpos[3 * b_wr2 + 1] - data->xpos[3 * b_el2 + 1];
            double fz = data->xpos[3 * b_wr2 + 2] - data->xpos[3 * b_el2 + 2];
            double elev = std::atan2(fz, mju_sqrt(fx * fx + fy * fy));
            plane_err +=
                flat_gain * act * mju_max(0.0, mju_abs(elev) - flat_tol);
          }
        }
      }
    }
    residual[counter++] = plane_err;
  }

  // // ========== FOREARM BRACING (H12_Hands only - OPTIONAL) ========== //
  // // Check if we have elbow sensors (indicates H12_Hands model)
  // bool has_elbow_sensors = false;
  // for (int i = 0; i < model->nsensor; i++) {
  //   if (std::string(model->names + model->name_sensoradr[i]) == "left_elbow_pos") {
  //     has_elbow_sensors = true;
  //     break;
  //   }
  // }
  
  // if (has_elbow_sensors) {
  //   // Get elbow positions for bracing arm
  //   double const *left_elbow_pos = SensorByName(model, data, "left_elbow_pos");
  //   double const *right_elbow_pos = SensorByName(model, data, "right_elbow_pos");
  //   double const *bracing_elbow = left_reaches ? right_elbow_pos : left_elbow_pos;
    
  //   // Get elbow orientation (z-axis should point along forearm)
  //   double const *left_elbow_zaxis = SensorByName(model, data, "left_elbow_zaxis");
  //   double const *right_elbow_zaxis = SensorByName(model, data, "right_elbow_zaxis");
  //   double const *bracing_elbow_zaxis = left_reaches ? right_elbow_zaxis : left_elbow_zaxis;
    
  //   // Ideal forearm position: between palm and elbow, on table surface
  //   double ideal_forearm[3] = {
  //       (bracing_palm[0] + bracing_elbow[0]) * 0.5,
  //       bracing_hand[1],
  //       table_pos[2]  // On table surface
  //   };
    
  //   // Distance from ideal forearm contact point
  //   mju_sub3(&residual[counter], bracing_elbow, ideal_forearm);
  //   counter += 3;
    
  //   // Forearm orientation: want forearm parallel to table (z-axis perpendicular to table normal)
  //   // If forearm is flat, dot product with table normal should be ~0
  //   double table_normal[3] = {0, 0, 1};
  //   double forearm_alignment = mju_abs(mju_dot3(bracing_elbow_zaxis, table_normal));
  //   residual[counter++] = forearm_alignment;  // Should be close to 0 when parallel
    
  //   // Optional: Check for forearm contact (if you add touch sensor)
  //   // This would require adding a touch sensor to the elbow_link geoms
  // }
  // // ========== END FOREARM BRACING ========== //

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  // Per-sensor diagnostic: prints once when the mismatch fires so we can see
  // exactly which residual is short. mju_warning is non-fatal (vs mju_error
  // which calls exit() and freezes headless agent_server with "Press Enter").
  if (user_sensor_dim != counter) {
    static bool printed = false;
    if (!printed) {
      printed = true;
      mju_warning(
          "Lean residual length %d != user_sensor_dim %d. Per-sensor:",
          counter, user_sensor_dim);
      for (int i = 0; i < model->nsensor; i++) {
        if (model->sensor_type[i] == mjSENS_USER) {
          const char *nm = mj_id2name(model, mjOBJ_SENSOR, i);
          mju_warning("  user sensor %d: %s dim=%d", i,
                      nm ? nm : "?", model->sensor_dim[i]);
        }
      }
    }
    // Pad with zeros so the planner doesn't crash; this is the fallback path
    // until the missing residual is tracked down.
    while (counter < user_sensor_dim) {
      residual[counter++] = 0.0;
    }
  }
}

void lean::ResidualFn::ContactResidual(const mjModel *model, const mjData *data,
                                       double *residual, int *counter) const {
  using mjpc::humanoid::kNotSelectedInteract;
  using mjpc::humanoid::kNumberOfContactPairsInteract;
  // Per-pair age factor: newly-appeared pairs ramp in over kPhaseRampSeconds
  // so the planner doesn't see the target gradient as a step change. Old
  // pairs (continuously active across the phase boundary) get factor 1.0.
  double t_in_phase = mju_max(0.0, data->time - keyframe_start_time_);
  double phase_t_norm = mju_min(1.0, t_in_phase / kPhaseRampSeconds);
  double phase_smoothstep =
      phase_t_norm * phase_t_norm * (3.0 - 2.0 * phase_t_norm);
  for (int i = 0; i < kNumberOfContactPairsInteract; i++) {
    const mjpc::humanoid::ContactPair& contact = residual_keyframe_.contact_pairs[i];
    if (contact.body1 != kNotSelectedInteract &&
        contact.body2 != kNotSelectedInteract &&
        contact.body1 < model->nbody &&
        contact.body2 < model->nbody) {
      double dist[3] = {0.};
      contact.GetDistance(dist, data);
      double age_factor = contact_pair_is_new_[i] ? phase_smoothstep : 1.0;
      for (int j = 0; j < 3; j++) {
        residual[(*counter)++] = age_factor * mju_abs(dist[j]);
      }
    } else {
      for (int j = 0; j < 3; j++) residual[(*counter)++] = 0;
    }
  }
}

// -------- Transition for humanoid_bench lean task -------- //
// ------------------------------------------------------------ //
//
// ITER 26 (2026-05-18): removed the iter-23 equality-weld toggle. Single-
// point/SE3 pins on the foot felt unnatural (sway-spring behaviour); foot
// anchoring is now done by real physics (gravcomp 0.97 → 0.90 gives ~49 N
// of net body weight on each foot, enough friction to hold against the
// soft cost gradients without artificial pins).
void lean::TransitionLocked(mjModel *model, mjData *data) {
  // ---- T1 REFERENCE TRIM v2 (ported from stabilize.cc 1708253, 2026-07-20) - //
  // Runs on the REAL state once per plan (never inside a rollout), so the
  // integrator sees measured physics, not the sampler's imagination.
  //
  // EXACT-NAME GATE: the stand keyframe only. The trim's whole premise is a
  // both-feet quiet park; a trot/walk/drive keyframe has no such park and its
  // twin tuning was never done against a moving reference. Same scoping rule
  // the 07-19 stumble anchor port used, and the reason a flag's verdict is
  // scoped to the strategy it was measured on.
  {
    const bool trim_strategy =
        (residual_.residual_keyframe_.name == "stand_up");
    int tt_id = mj_name2id(model, mjOBJ_NUMERIC, "stand_trim_tau");
    double ttau = (tt_id >= 0)
        ? model->numeric_data[model->numeric_adr[tt_id]] : 0.0;
    double const *tup = SensorByName(model, data, "torso_up");
    double const *cvel = SensorByName(model, data, "waist_lower_subcomvel");
    int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
    static double s_trim_t = -1.0;      // last tick time, for dt
    static double s2_ex = 0.0, s2_ey = 0.0;   // trim's own 4 s DC EMA
    if (trim_strategy && ttau > 0.0 && tup && cvel && pid >= 0) {
      auto Num = [&](const char *name, double dflt) {
        int id = mj_name2id(model, mjOBJ_NUMERIC, name);
        return (id >= 0) ? model->numeric_data[model->numeric_adr[id]] : dflt;
      };
      double tdelay   = Num("stand_trim_delay", 0.0);
      double tmax_pos = Num("stand_trim_max", 0.08);
      double tmax_neg = Num("stand_trim_neg_max", tmax_pos);
      double tleak    = Num("stand_trim_leak", 60.0);
      double tquiet   = Num("stand_trim_quiet", 0.03);
      double tlat_max = Num("stand_trim_lat_max", 0.02);
      double tnom     = Num("stand_trim_nominal_x", 0.0);
      const bool trim_armed = (tdelay <= 0.0) || (data->time >= tdelay);

      // SUPPORT frame from the feet's own mean heading (must match the axes the
      // Residual applies the trim along; degenerate feet -> world axes).
      double fwd[2] = {1.0, 0.0}, lat[2] = {0.0, 1.0};
      {
        double const *flf = SensorByName(model, data, "foot_left_forward");
        double const *frf = SensorByName(model, data, "foot_right_forward");
        if (flf && frf) {
          double fx = flf[0] + frf[0], fy = flf[1] + frf[1];
          double len = mju_sqrt(fx * fx + fy * fy);
          if (len > 1.0e-6) {
            fwd[0] = fx / len; fwd[1] = fy / len;
            lat[0] = -fwd[1];  lat[1] = fwd[0];
          }
        }
      }
      const mjtNum *com = data->subtree_com + 3 * pid;
      double zc = mju_max(0.5, com[2]);
      double tau_c = mju_sqrt(zc / 9.81);
      // capture excursion in the SUPPORT frame; nominal 0 = upright (v1's bug
      // was using the strat-20 lean nominal 0.06 here, which drove the stand's
      // CoM 6 cm toward the toe).
      double exf = zc * ((tup[0] * fwd[0] + tup[1] * fwd[1]) - tnom) +
                   tau_c * (cvel[0] * fwd[0] + cvel[1] * fwd[1]);
      double eyl = zc * (tup[0] * lat[0] + tup[1] * lat[1]) +
                   tau_c * (cvel[0] * lat[0] + cvel[1] * lat[1]);
      double dt = (s_trim_t >= 0.0) ? mju_max(0.0, data->time - s_trim_t) : 0.0;
      s_trim_t = data->time;
      double a2 = (dt > 0.0) ? mju_min(1.0, dt / 4.0) : 1.0;  // first call snaps
      s2_ex += a2 * (exf - s2_ex);
      s2_ey += a2 * (eyl - s2_ey);
      // QUIET gate: only a STEADY park integrates. During a push, a catch, or
      // an operator hand on the chest the instantaneous excursion leaves its DC
      // -> freeze. (v1 wound those transients into the reference = the 07-14
      // hunt. It also means an ASSISTED run must not be used to judge the trim:
      // the gate cannot see an external force, only the motion it suppresses.)
      const bool quiet = std::fabs(exf - s2_ex) < tquiet &&
                         std::fabs(eyl - s2_ey) < tquiet;
      if (trim_armed && quiet) {
        s_trim_x += (dt / ttau) * s2_ex;
        s_trim_y += (dt / ttau) * s2_ey;
      } else if (!trim_armed) {
        s_trim_x = 0.0; s_trim_y = 0.0;
      }
      // LEAK (Caron'19): decay toward 0. A true constant bias re-wins every
      // tick (steady state trim = park * tleak/ttau, residual park =
      // trim * ttau/tleak -- at the 60/15 defaults a 4 cm need leaves ~1 cm
      // park); transient garbage has no source and self-unwinds.
      if (tleak > 0.0 && dt > 0.0) {
        double decay = 1.0 - mju_min(1.0, dt / tleak);
        s_trim_x *= decay;
        s_trim_y *= decay;
      }
      s_trim_x = mju_min(tmax_pos, mju_max(-tmax_neg, s_trim_x));
      s_trim_y = mju_min(tlat_max, mju_max(-tlat_max, s_trim_y));
      static double last_print = -1.0e9;
      if (data->time - last_print > 5.0) {
        last_print = data->time;
        // stderr on purpose: agent_server/deploy stdout is BLOCK-buffered when
        // redirected, and the line vanishes if the process is killed unflushed.
        std::fprintf(stderr,
                     "[trim] t=%.1f park(fwd%+.3f lat%+.3f) m trim(x%+.3f "
                     "y%+.3f) m armed=%d quiet=%d\n",
                     data->time, s2_ex, s2_ey, s_trim_x, s_trim_y,
                     trim_armed ? 1 : 0, quiet ? 1 : 0);
      }
    } else {
      // OFF (or a non-stand strategy): hard-zero everything, including the EMA
      // state, so a later arm starts clean instead of inheriting a stale park.
      s_trim_x = 0.0; s_trim_y = 0.0;
      s2_ex = 0.0; s2_ey = 0.0; s_trim_t = -1.0;
    }
  }

  // ---- DEBUG: print leg stability diagnostics every ~0.5 s ---- //
  static int debug_tick = 0;
  static const bool lean_dbg = [] {
    const char* e = std::getenv("LEAN_DEBUG");
    return e != nullptr && e[0] != '0';   // set but "0" = explicitly off
  }();
  if (lean_dbg && ++debug_tick % 33 == 0) {  // ~0.5 s; set LEAN_DEBUG=1 to enable
    double *foot_right_up = SensorByName(model, data, "foot_right_up");
    double *foot_left_up  = SensorByName(model, data, "foot_left_up");
    double *torso_up      = SensorByName(model, data, "torso_up");
    double torso_h        = SensorByName(model, data, "torso_position")[2];
    double leaning        = torso_h / mju_sqrt(torso_h * torso_h + 0.65 * 0.65) - 0.2;

    // largest joint velocity (which DOF is flailing)
    int worst_dof = 0;
    double worst_vel = 0;
    for (int i = 0; i < model->nu; i++) {
      double v = mju_abs(data->qvel[6 + i]);
      if (v > worst_vel) { worst_vel = v; worst_dof = i; }
    }
    const char *worst_name = mj_id2name(model, mjOBJ_JOINT, worst_dof);

    double const *obj_pos_dbg = SensorByName(model, data, "object_pos");
    double const *lh = SensorByName(model, data, "left_hand_pos");
    double const *rh = SensorByName(model, data, "right_hand_pos");
    double ld = mju_dist3(lh, obj_pos_dbg);
    double rd = mju_dist3(rh, obj_pos_dbg);
    double *pelvis_pos_dbg   = SensorByName(model, data, "pelvis_position");
    const double *tsurf_dbg  = SensorByName(model, data, "table_surface_pos");
    double table_front_x_dbg = tsurf_dbg[0] - 0.7;
    printf("[LEAN DBG t=%.2f] leaning=%.3f | torso_up_z=%.3f | "
           "rfoot_up_z=%.3f lfoot_up_z=%.3f | "
           "obj_y=%.3f L_dist=%.2f R_dist=%.2f | "
           "pelvis_x=%.3f(table_front=%.3f) | "
           "worst_dof=%d(%s) vel=%.2f rad/s\n",
           data->time, leaning, torso_up[2],
           foot_right_up[2], foot_left_up[2],
           obj_pos_dbg[1], ld, rd,
           pelvis_pos_dbg[0], table_front_x_dbg,
           worst_dof, worst_name ? worst_name : "?", worst_vel);

    // Print any contacts involving the table
    int tbl_geom = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
    for (int c = 0; c < data->ncon; c++) {
      const mjContact& con = data->contact[c];
      if (con.geom1 == tbl_geom || con.geom2 == tbl_geom) {
        int other_geom = (con.geom1 == tbl_geom) ? con.geom2 : con.geom1;
        int other_body = model->geom_bodyid[other_geom];
        const char *bn = mj_id2name(model, mjOBJ_BODY, other_body);
        printf("  [TABLE CONTACT] body=%d(%s) geom=%d pos=(%.3f,%.3f,%.3f)\n",
               other_body, bn ? bn : "?", other_geom,
               con.pos[0], con.pos[1], con.pos[2]);
      }
    }
    // ---- Biomechanics trace: confirms the Height-vs-squat hypothesis ----
    // Want to know whether MPC is choosing the lean-without-squat strategy
    // (knees ≈ 0, hip pitch ≈ 0, pelvis translates forward, CoM goes past
    // midfoot) vs the squat-and-lean strategy (knees + hip pitch flexing,
    // pelvis stays back). qpos indices follow the actuator order in
    // lean.cc::kJointVelLimit: L_hip_pitch=1, L_knee=3, R_hip_pitch=7,
    // R_knee=9 → qpos[7+i].
    double L_hip_pitch_deg = data->qpos[7 + 1] * 180.0 / M_PI;
    double L_knee_deg      = data->qpos[7 + 3] * 180.0 / M_PI;
    double R_hip_pitch_deg = data->qpos[7 + 7] * 180.0 / M_PI;
    double R_knee_deg      = data->qpos[7 + 9] * 180.0 / M_PI;
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    double com_x = (pelvis_id >= 0)
                       ? data->subtree_com[3 * pelvis_id + 0]
                       : 0.0;
    double *foot_R = SensorByName(model, data, "foot_right_pos");
    double *foot_L = SensorByName(model, data, "foot_left_pos");
    double midfoot_x = 0.5 * (foot_R[0] + foot_L[0]);
    double com_ahead = com_x - midfoot_x;  // >0 = CoM forward of feet
    double head_z = SensorByName(model, data, "head_position")[2];
    double foot_avg_z = 0.5 * (foot_R[2] + foot_L[2]);
    double head_feet = head_z - foot_avg_z;
    double *lhand = SensorByName(model, data, "left_hand_pos");
    double *rhand = SensorByName(model, data, "right_hand_pos");
    const std::string& phase_name = residual_.residual_keyframe_.name;
    double t_in_phase = data->time - residual_.keyframe_start_time_;
    // Strategy name (which JSON is loaded) — printed every debug tick so the
    // user can identify which view is running without consulting the slider
    // index → name mapping.
    const auto strat_names_dbg = GetStrategyNames();
    const char *strat_name =
        (current_strategy_ >= 0 &&
         current_strategy_ < (int)strat_names_dbg.size())
            ? strat_names_dbg[current_strategy_].c_str()
            : "?";
    int strat_phase_idx = motion_strategy_.GetCurrentKeyframeIndex();
    int strat_phase_count = motion_strategy_.GetKeyframesCount();
    // Added foot_R_z, foot_L_z to detect if a foot is lifting off ground
    // (user reports "leg lifting before braced arm position" — verify).
    printf("  [STRAT %d:%s | kf %d/%d]\n", current_strategy_, strat_name,
           strat_phase_idx + 1, strat_phase_count);
    printf("  [BIO  t=%.2f phase=%s(%.2fs)] "
           "L_hipP=%6.1f L_knee=%6.1f R_hipP=%6.1f R_knee=%6.1f deg | "
           "footR_z=%.3f footL_z=%.3f | "
           "com_x=%.3f midfoot_x=%.3f Δ=%+.3f | "
           "head_feet=%.3f | Lhand_x=%.3f Rhand=%.2f,%.2f,%.2f\n",
           data->time, phase_name.c_str(), t_in_phase,
           L_hip_pitch_deg, L_knee_deg, R_hip_pitch_deg, R_knee_deg,
           foot_R[2], foot_L[2],
           com_x, midfoot_x, com_ahead,
           head_feet, lhand[0],
           rhand[0], rhand[1], rhand[2]);
  }
  // ---- END DEBUG ---- //

  // keep mocap target updated for the existing hand-reach residuals.
  // If the model declares a `reach_target` numeric (3 values) the target is
  // EXTERNALLY SUPPLIED — the Strategy-21 reach primitive, or a vision/nav stack
  // writing that numeric. Pin the mocap object there and never auto-respawn it.
  // With no such numeric the legacy retrieve-game behavior holds (random object
  // that re-spawns once the reaching hand touches it).
  int reach_tgt_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_target");
  // LIVE reach target (2026-07-02): if the gRPC-settable "Reach Active" param is
  // on, use the live (X,Y,Z) point supplied by the core_ws ROS2 bridge instead of
  // the static `reach_target` numeric. The bridge has already converted the
  // perception PoseStamped (pelvis frame) into MJPC world via the node's base pose,
  // so these are a stable world point (see lean.h). Guarded by parameters.size()
  // so models without these params fall back to the legacy numeric path.
  const bool reach_live =
      (int)parameters.size() > kLeanReachZParameterIndex &&
      parameters[kLeanReachActiveParameterIndex] > 0.5;
  if (reach_live) {
    target_position_ = {parameters[kLeanReachXParameterIndex],
                        parameters[kLeanReachYParameterIndex],
                        parameters[kLeanReachZParameterIndex]};
  } else if (reach_tgt_id >= 0) {
    const mjtNum *rt = model->numeric_data + model->numeric_adr[reach_tgt_id];
    target_position_ = {rt[0], rt[1], rt[2]};
  } else {
    double const *object_pos = SensorByName(model, data, "object_pos");
    double const *left_hand_pos_tr = SensorByName(model, data, "left_hand_pos");
    double hand_dist = mju_dist3(left_hand_pos_tr, object_pos);
    if (hand_dist < 0.05) {
      std::random_device rd;
      std::mt19937 gen(rd());
      // TEST #16 (2026-05-18): target x range 1.4-1.6 → 1.2-1.4. Closer to
      // robot so static reach is within natural-posture range; combined with
      // kp_ankle 20→40 should let stand-and-lean be stable.
      std::uniform_real_distribution<> dis_x(1.2, 1.4);
      std::uniform_real_distribution<> dis_y(-0.3, 0.3);
      target_position_ = {dis_x(gen), dis_y(gen), 0.83};
      printf("New target position: %f, %f, %f\n", target_position_[0],
             target_position_[1], target_position_[2]);
    }
  }
  mju_copy3(data->mocap_pos, target_position_.data());

  // strategy-based contact keyframe progression
  const auto kStrategyNames = GetStrategyNames();
  int requested_strategy =
      (int)std::round(parameters[kLeanStrategyParameterIndex]);
  requested_strategy = std::max(
      0, std::min(requested_strategy, (int)kStrategyNames.size() - 1));

  // Helper: diff old vs new keyframe contact-pair activity and mark which
  // pairs just appeared (active now, inactive before). ContactResidual
  // uses these flags to ramp the new pair's residual contribution from 0
  // → full over kPhaseRampSeconds so the planner doesn't see an
  // instantaneous gradient toward the new contact target.
  auto MarkNewlyAppearedContacts =
      [&](const mjpc::humanoid::ContactKeyframe& old_kf,
          const mjpc::humanoid::ContactKeyframe& new_kf) {
        for (int i = 0; i < mjpc::humanoid::kNumberOfContactPairsInteract;
             ++i) {
          bool old_active = (old_kf.contact_pairs[i].body1 !=
                             mjpc::humanoid::kNotSelectedInteract);
          bool new_active = (new_kf.contact_pairs[i].body1 !=
                             mjpc::humanoid::kNotSelectedInteract);
          residual_.contact_pair_is_new_[i] = (new_active && !old_active);
        }
      };

  // Helper: snapshot the scales currently in effect (mid-ramp possible) so
  // the next phase ramps smoothly out of them. The lerp matches what
  // Residual() is doing, so prev_* always equals what the rollouts are
  // actually seeing at the moment of transition.
  auto SnapshotEffectiveScales = [&]() {
    double r_t, b_t, p_t;
    PhaseTargetScales(residual_.residual_keyframe_.name, r_t, b_t, p_t);
    double dt = mju_max(0.0, data->time - residual_.keyframe_start_time_);
    double alpha_lin =
        mju_min(dt / ResidualFn::kPhaseRampSeconds, 1.0);
    // Match the smoothstep used in Residual() so the snapshot equals what
    // the rollouts were actually seeing at the moment of transition.
    double alpha = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
    residual_.prev_phase_reach_scale_ =
        residual_.prev_phase_reach_scale_ +
        alpha * (r_t - residual_.prev_phase_reach_scale_);
    residual_.prev_phase_brace_pos_scale_ =
        residual_.prev_phase_brace_pos_scale_ +
        alpha * (b_t - residual_.prev_phase_brace_pos_scale_);
    residual_.prev_phase_posture_scale_ =
        residual_.prev_phase_posture_scale_ +
        alpha * (p_t - residual_.prev_phase_posture_scale_);
    // ITER 28: snapshot brace_force_target the same way so the next phase's
    // ramp starts from the actual mid-ramp value rather than snapping.
    double bf_t = residual_.residual_keyframe_.brace_force_target >= 0.0
                      ? residual_.residual_keyframe_.brace_force_target
                      : 0.0;
    residual_.prev_phase_brace_force_target_ =
        residual_.prev_phase_brace_force_target_ +
        alpha * (bf_t - residual_.prev_phase_brace_force_target_);
    // Capture the posture keyframe id of the phase we are LEAVING so Residual()
    // ramps the target pose OUT of it into the new phase (parallels the scale
    // snapshots above). Transitions here fire post-settle (sustain >> ramp) so
    // the leaving keyframe == the effective target; mid-ramp scrubs accept a
    // small target discontinuity, same as the pre-existing scale snapshot.
    int leaving_key = mj_name2id(model, mjOBJ_KEY,
                                 residual_.residual_keyframe_.name.c_str());
    residual_.prev_posture_key_id_ = (leaving_key < 0) ? 0 : leaving_key;
  };

  bool cold_start = !motion_strategy_.HasKeyframes();
  if (cold_start || requested_strategy != current_strategy_) {
    // A live strategy switch (not the first load) eases into the new task:
    // snapshot the scales + weights the rollouts currently see, so the new
    // strategy ramps OUT of them over kPhaseRampSeconds instead of snapping.
    // Reuses the exact machinery the manual Phase scrubber uses, so a
    // task-select button press transitions calmly rather than lurching.
    if (!cold_start) {
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
    }
    current_strategy_ = requested_strategy;
    motion_strategy_.ClearKeyframes();
    motion_strategy_.LoadStrategy(kStrategyNames[current_strategy_],
                                  kLeanStrategyFilePath);
    // Gate the target-pose ramp (Residual): >1 phase => cyclic strategy gets the
    // smooth transitions; 1 => single-phase, the ramp stays disabled (byte-identical).
    residual_.num_phases_ = motion_strategy_.GetKeyframesCount();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    MarkNewlyAppearedContacts(residual_.residual_keyframe_,
                              motion_strategy_.GetCurrentKeyframe());
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    residual_.keyframe_start_time_ = data->time;
    // ★ Pin the Foot Stability anchor to the MEASURED stance on entry to a
    // STEPPING keyframe (trot/walk/drive). The residual's home constants are
    // odometric-frame and that frame drifts, so anchoring to where the feet
    // ACTUALLY are at hand-over is the whole fix (lower-body: cost 225 -> 0.00
    // on real, 2026-07-18). Feet are planted at entry, so this is a clean
    // capture. Non-stepping keyframes (incl. strat 6 stand) never touch it.
    // ⚠ NO COLD-START PIN. The first version pinned HERE, at strategy entry --
    // i.e. before the state estimate is initialised -- and only null-checked the
    // sensor pointers. On the real robot that DRAGGED IT SIDEWAYS at startup
    // (uninitialised read => both anchors near the origin => the weight-8 cost
    // hauls both feet toward it). The anchor now starts at the legacy constants
    // and is only ever pinned by the drive-idle path below, which is gated on
    // settle time AND plausibility. Left as a note so nobody re-adds it.
    if (cold_start) {
      // First load: no history to ramp from. Snap to the first phase's
      // targets (prev scales = 0, posture = 1.0 "no-boost", brace_force = 0),
      // exactly as before. stand_up's targets are also (0, 0) so there is no
      // visible ramp on boot.
      residual_.prev_phase_reach_scale_        = 0.0;
      residual_.prev_phase_brace_pos_scale_    = 0.0;
      residual_.prev_phase_posture_scale_      = 1.0;
      residual_.prev_phase_brace_force_target_ = 0.0;
      residual_.prev_posture_key_id_           = 0;  // home: gentle boot ramp
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
      prev_phase_weights_ = next_phase_weights_;  // snap (no history)
    } else {
      // Live switch: prev_* scales + prev_phase_weights_ already hold the
      // current effective values (snapshotted above). Only set the new
      // targets; ApplyRampedWeights + the Residual() scale ramp carry the
      // transition over kPhaseRampSeconds (alpha keyed off keyframe_start_time_).
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    }
    ApplyRampedWeights(model, data);
    return;
  }

  // ----- Manual phase scrubber ---------------------------------------- //
  // residual_Phase parameter:
  //   -1  → auto-advance (existing behaviour, runs below)
  //   0..N-1 → hold at that keyframe index regardless of progress;
  //            auto-advance disabled. Body state is preserved across
  //            manual jumps — only the active keyframe changes, with
  //            weights smoothstep-ramping into the new phase (the same
  //            machinery the auto-advance uses).
  int requested_phase =
      (int)std::round(parameters[kLeanPhaseParameterIndex]);
  bool manual_phase_mode = (requested_phase >= 0);
  if (manual_phase_mode) {
    int n_kf = motion_strategy_.GetKeyframesCount();
    requested_phase = std::max(0, std::min(requested_phase, n_kf - 1));
    int current_kf_idx = motion_strategy_.GetCurrentKeyframeIndex();
    if (requested_phase != current_kf_idx) {
      // Manual phase jump — same snapshot+ramp dance as auto-advance, so
      // weights blend smoothly from the old phase's effective values into
      // the new phase's targets over kPhaseRampSeconds. Body qpos/qvel
      // are untouched; only the active keyframe (contact targets, brace
      // force target, weight overrides) changes.
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.UpdateCurrentKeyframe(requested_phase);
      MarkNewlyAppearedContacts(residual_.residual_keyframe_,
                                motion_strategy_.GetCurrentKeyframe());
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    }
    // Skip auto-advance entirely while in manual mode.
  } else {
    const mjpc::humanoid::ContactKeyframe& current_kf =
        motion_strategy_.GetCurrentKeyframe();
    const double total_distance =
        motion_strategy_.CalculateTotalKeyframeDistance(
            data, mjpc::humanoid::ContactKeyframeErrorType::kNorm);

    // ★ 2026-08-07 COMMIT ABORT-AND-RETRY. `lean_commit_retry` numeric:
    // 0 = OFF = BYTE-IDENTICAL. >0 = seconds of detected STALL before the
    // strategy regresses to keyframe 0 (stand) and re-attempts, capped at 3
    // retries. WHY: the lean commit fails ~25% on the twin bench in a
    // RECOVERABLE way — the bow hangs quasi-statically at tilt 12-28° with
    // the forearm pad NOT on the table (toe-margin stall), then topples
    // seconds later. While stalled the CoM is still over the feet, so
    // commanding the stand keyframe recovers it; riding the stall ends in a
    // fall. The built-in time-limit reset cannot catch this because
    // total_distance is identically 0 (all contact pairs -1). Detector is
    // physical: in the lean/mid phases, >=6 s in phase AND torso tilt in
    // the stall band AND no pad↔table contact, sustained `retry` seconds.
    // Static-local state: the deploy node owns exactly one lean task
    // instance and calls Transition serially; bench-scoped pragmatism.
    {
      static double stall_since = -1.0;
      static int retries_used = 0;
      static double last_time = -1.0;
      static double tilt_prev = -1.0, tilt_rate_ema = 10.0;
      if (data->time < last_time - 1.0) {  // sim restarted → clear state
        stall_since = -1.0; retries_used = 0;
        tilt_prev = -1.0; tilt_rate_ema = 10.0;
      }
      double dt_step = data->time - last_time;
      last_time = data->time;
      int rt_id = mj_name2id(model, mjOBJ_NUMERIC, "lean_commit_retry");
      double retry_sec =
          rt_id >= 0 ? model->numeric_data[model->numeric_adr[rt_id]] : 0.0;
      const std::string& kf_name = current_kf.name;
      bool commit_phase = (kf_name == "forearm_brace_lean" ||
                           kf_name == "forearm_brace_mid");
      if (retry_sec > 0.0 && commit_phase && retries_used < 3) {
        // torso tilt from the base quaternion
        const double* q = data->qpos + 3;
        double tilt = 2.0 * mju_acos(mju_min(mju_abs(q[0]), 1.0));
        // pad↔table contact scan
        int pad_gid = mj_name2id(model, mjOBJ_GEOM, "left_forearm_pad");
        bool pad_on = false;
        for (int ci = 0; ci < data->ncon; ci++) {
          const mjContact& con = data->contact[ci];
          if (con.geom1 == pad_gid || con.geom2 == pad_gid) {
            pad_on = true;
            break;
          }
        }
        // tilt RATE (EMA over ~1 s): a HEALTHY approach descends at 2-4
        // deg/s through the same band with the pad still off — first gate
        // build aborted good commits mid-approach (4/5 false positives,
        // seats 2/5). The stall is quasi-static: |rate| < 0.8 deg/s.
        if (tilt_prev >= 0.0 && dt_step > 1e-6 && dt_step < 1.0) {
          double r = (tilt - tilt_prev) / dt_step;
          double a = mju_min(1.0, dt_step / 1.0);
          tilt_rate_ema = (1.0 - a) * tilt_rate_ema + a * r;
        }
        tilt_prev = tilt;
        double in_phase =
            data->time - motion_strategy_.GetCurrentKeyframeStartTime();
        bool stalled = in_phase > 6.0 && !pad_on &&
                       tilt > 12.0 * mjPI / 180.0 &&
                       tilt < 28.0 * mjPI / 180.0 &&
                       mju_abs(tilt_rate_ema) < 0.8 * mjPI / 180.0;
        if (!stalled) {
          stall_since = -1.0;
        } else if (stall_since < 0.0) {
          stall_since = data->time;
        } else if (data->time - stall_since > retry_sec) {
          retries_used++;
          stall_since = -1.0;
          std::printf(
              "[lean-retry] COMMIT STALL in '%s' (tilt %.1f deg, pad off, "
              "%.1fs) — regressing to stand, attempt %d/3\n",
              kf_name.c_str(), tilt * 180.0 / mjPI, retry_sec, retries_used);
          SnapshotEffectiveScales();
          SnapshotCurrentWeightsAsPrev();
          motion_strategy_.Reset();
          MarkNewlyAppearedContacts(residual_.residual_keyframe_,
                                    motion_strategy_.GetCurrentKeyframe());
          residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
          motion_strategy_.SetCurrentKeyframeStartTime(data->time);
          motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
          residual_.keyframe_start_time_ = data->time;
          PrepareNextPhaseWeights(residual_.residual_keyframe_);
          ApplyRampedWeights(model, data);
          return;
        }
      }
    }

    if (data->time - motion_strategy_.GetCurrentKeyframeStartTime() >
            current_kf.time_limit &&
        total_distance > current_kf.target_distance_tolerance) {
      // Time-limit reset (strategy restarts from keyframe 0). Save the
      // scales that were just in effect so the next ramp blends from them.
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.Reset();
      MarkNewlyAppearedContacts(residual_.residual_keyframe_,
                                motion_strategy_.GetCurrentKeyframe());
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    } else if (total_distance <= current_kf.target_distance_tolerance &&
               data->time -
                       motion_strategy_.GetCurrentKeyframeSuccessStartTime() >
                   current_kf.success_sustain_time &&
               [&] {
                 // ★ 2026-08-12 BRACE-CONTACT-GATED ADVANCE (user-approved
                 // spec): brace-side rungs (lean/mid/reach) may only advance
                 // after the forearm pad has been in believed table contact
                 // for >= `brace_contact_verify` s continuously. 0/absent =
                 // OFF = byte-identical. Loss >= `brace_contact_loss` s
                 // resets the counter (freeze: the ladder holds while the
                 // pad re-seats — Brace Pos stays active). Loss >=
                 // `brace_contact_freeze_cap` s => give up: handled by the
                 // retry machinery via the stall path (pad off + banded
                 // tilt), which this freeze feeds by construction.
                 int bcv = mj_name2id(model, mjOBJ_NUMERIC,
                                      "brace_contact_verify");
                 double vsec = bcv >= 0
                     ? model->numeric_data[model->numeric_adr[bcv]] : 0.0;
                 const std::string& kfn =
                     motion_strategy_.GetCurrentKeyframe().name;
                 bool brace_side = (kfn == "forearm_brace_lean" ||
                                    kfn == "forearm_brace_mid" ||
                                    kfn == "forearm_brace_reach");
                 if (vsec > 0.0 && brace_side) {
                   int pad_gid2 =
                       mj_name2id(model, mjOBJ_GEOM, "left_forearm_pad");
                   bool on = false;
                   for (int ci = 0; ci < data->ncon; ci++) {
                     const mjContact& con = data->contact[ci];
                     if (con.geom1 == pad_gid2 || con.geom2 == pad_gid2) {
                       on = true;
                       break;
                     }
                   }
                   // ★ 2026-08-13 NEAR-CONTACT MARGIN (real flat_2): the
                   // REAL pad sat planted for ~65 s (pitch 30 rock-steady)
                   // while the BELIEVED pad hovered ~2 cm above the believed
                   // surface (attitude error at 30 deg pitch + unmodeled
                   // rubber), so believed contact flickered 134x and the
                   // ladder never advanced on a solid real brace. If
                   // `brace_contact_zmargin` (m) > 0, count the pad as ON
                   // whenever its lowest point is within that margin of the
                   // slab top while horizontally over the slab. Flat/INCLINED
                   // still gates edge grazes. 0/absent = strict contact scan.
                   if (!on) {
                     int nzm = mj_name2id(model, mjOBJ_NUMERIC,
                                          "brace_contact_zmargin");
                     double zmar = nzm >= 0
                         ? model->numeric_data[model->numeric_adr[nzm]] : 0.0;
                     int g_tab3 = mj_name2id(model, mjOBJ_GEOM,
                                             "table_top_collision");
                     if (zmar > 0.0 && pad_gid2 >= 0 && g_tab3 >= 0) {
                       const double* pp = data->geom_xpos + 3 * pad_gid2;
                       double prad = model->geom_size[3 * pad_gid2];
                       const double* tc = data->geom_xpos + 3 * g_tab3;
                       double thx = model->geom_size[3 * g_tab3 + 0];
                       double thy = model->geom_size[3 * g_tab3 + 1];
                       double surf = tc[2] + model->geom_size[3 * g_tab3 + 2];
                       // ★ flat_4 telemetry: gap was +11 mm but over=0 on
                       // EVERY sample — the relative tag anchor preserves
                       // the robot-vs-table offset latched at bring-up, so
                       // believed x can sit ~5-10 cm behind truth and the
                       // footprint test starves the gate. Believed z is the
                       // trustworthy axis (~1 cm); slack the x/y footprint
                       // by `brace_contact_xyslack` (m, default 0.10).
                       int nxs = mj_name2id(model, mjOBJ_NUMERIC,
                                            "brace_contact_xyslack");
                       double xys = nxs >= 0
                           ? model->numeric_data[model->numeric_adr[nxs]]
                           : 0.10;
                       bool over = mju_abs(pp[0] - tc[0]) < thx + xys &&
                                   mju_abs(pp[1] - tc[1]) < thy + xys;
                       double gap = (pp[2] - prad) - surf;
                       if (over && gap < zmar) on = true;
                       // ★ measure, don't guess: print the believed gap so
                       // real runs tell us the actual belief offset.
                       static int gap_note = 0;
                       if (++gap_note % 100 == 1)
                         std::printf("[lean-gate] believed pad gap %+.0f mm "
                                     "(over=%d margin %.0f mm)\n",
                                     gap * 1000.0, (int)over, zmar * 1000.0);
                     }
                   }
                   // ★ 2026-08-13 FLAT-GATED VERIFY: a corner graze is not a
                   // seat (real 14/15: 24-33 deg inclined edge contact
                   // chattered 23-25x and never verified). Contact only
                   // counts while forearm elevation < `brace_flat_gate`
                   // (rad; 0/absent = off).
                   bool inclined = false;
                   int nfg = mj_name2id(model, mjOBJ_NUMERIC,
                                        "brace_flat_gate");
                   double fgate = nfg >= 0
                       ? model->numeric_data[model->numeric_adr[nfg]] : 0.0;
                   if (on && fgate > 0.0) {
                     int b_el3 =
                         mj_name2id(model, mjOBJ_BODY, "left_elbow_link");
                     int b_wr3 = mj_name2id(model, mjOBJ_BODY,
                                            "left_wrist_roll_link");
                     if (b_el3 >= 0 && b_wr3 >= 0) {
                       double fx = data->xpos[3 * b_wr3 + 0] -
                                   data->xpos[3 * b_el3 + 0];
                       double fy = data->xpos[3 * b_wr3 + 1] -
                                   data->xpos[3 * b_el3 + 1];
                       double fz = data->xpos[3 * b_wr3 + 2] -
                                   data->xpos[3 * b_el3 + 2];
                       double elev =
                           std::atan2(fz, mju_sqrt(fx * fx + fy * fy));
                       if (mju_abs(elev) > fgate) {
                         inclined = true;
                         on = false;
                       }
                     }
                   }
                   static double bc_since = -1.0, bc_lost_since = -1.0;
                   int bcl = mj_name2id(model, mjOBJ_NUMERIC,
                                        "brace_contact_loss");
                   double lsec = bcl >= 0
                       ? model->numeric_data[model->numeric_adr[bcl]] : 1.0;
                   if (on) {
                     bc_lost_since = -1.0;
                     if (bc_since < 0.0) bc_since = data->time;
                   } else {
                     if (bc_lost_since < 0.0) bc_lost_since = data->time;
                     if (data->time - bc_lost_since > lsec) bc_since = -1.0;
                   }
                   if (bc_since < 0.0 ||
                       data->time - bc_since < vsec) {
                     static int bc_note = 0;
                     if (++bc_note % 100 == 1)
                       std::printf("[lean-gate] HOLD %s: pad contact %s "
                                   "(need %.1fs verified)\n",
                                   kfn.c_str(),
                                   on ? "verifying"
                                      : (inclined ? "INCLINED" : "LOST"),
                                   vsec);
                     return false;
                   }
                 }
                 // ★ 2026-08-13 RELEASE PITCH GATE (bench20_flatstack 1-10
                 // forensics): with Balance 40 the un-bow now WORKS (38->26,
                 // 34->20 deg) but the pad leaves the table at 25-38 deg and
                 // the last stretch is ankle-only -- infeasible from that
                 // depth (the #51 toe-line bound), so every run stalls at
                 // 17-26 deg and topples forward 1-10 s later. Hold the
                 // release-side rungs until believed base pitch is below a
                 // per-rung bound, so the ARM keeps pressing and walks the
                 // pitch down into ankle-recoverable range before letting
                 // go (counter push-off). Numerics standback_pitch_release/
                 // _r1/_r2 in rad; 0/absent = OFF = prior behavior.
                 {
                   const char* pnum = nullptr;
                   if (kfn == "forearm_brace_release")
                     pnum = "standback_pitch_release";
                   else if (kfn == "standback_r1")
                     pnum = "standback_pitch_r1";
                   else if (kfn == "standback_r2")
                     pnum = "standback_pitch_r2";
                   if (pnum != nullptr) {
                     int pid = mj_name2id(model, mjOBJ_NUMERIC, pnum);
                     double plim = pid >= 0
                         ? model->numeric_data[model->numeric_adr[pid]] : 0.0;
                     if (plim > 0.0) {
                       const double* q = data->qpos;  // free joint quat wxyz
                       double sinp = 2.0 * (q[3] * q[5] - q[6] * q[4]);
                       sinp = mju_clip(sinp, -1.0, 1.0);
                       double bpitch = std::asin(sinp);
                       if (bpitch > plim) {
                         static int rp_note = 0;
                         if (++rp_note % 100 == 1)
                           std::printf(
                               "[lean-gate] HOLD %s: pitch %.2f > %.2f rad "
                               "— keep pressing\n",
                               kfn.c_str(), bpitch, plim);
                         return false;
                       }
                     }
                   }
                 }
                 return true;
               }() &&
               [&] {
                 // ★ 2026-08-06 `phase_advance_quiet_vel` numeric: 0 = OFF =
                 // BYTE-IDENTICAL. >0 = do not advance while the base sways
                 // faster than this (m/s, horizontal): the dwell clock can
                 // expire mid-sway and the next phase then inherits that
                 // velocity — on the twin bench ~half the lean commits
                 // entered on an adverse sway sample and toppled (stall→
                 // backward overshoot / asymmetric bow). Holding for a calm
                 // sample costs at most a sway half-period (~0.5 s).
                 int qv = mj_name2id(model, mjOBJ_NUMERIC,
                                     "phase_advance_quiet_vel");
                 if (qv < 0) return true;
                 double lim = model->numeric_data[model->numeric_adr[qv]];
                 if (lim <= 0.0) return true;
                 bool quiet = mju_sqrt(data->qvel[0] * data->qvel[0] +
                                       data->qvel[1] * data->qvel[1]) <= lim;
                 // ★ 2026-08-11 `phase_advance_quiet_wvel`: 0 = OFF =
                 // byte-identical. >0 = ALSO hold while the base ANGULAR
                 // rate (roll/pitch, rad/s) exceeds this. WHY: 80-run
                 // forensics — pre-release base roll RATE was the earliest
                 // failure predictor (AUC 0.71); the linear gate above is
                 // blind to it. Sway is periodic (~1 s) so waiting for a
                 // rate zero-crossing costs at most half a period.
                 int qw = mj_name2id(model, mjOBJ_NUMERIC,
                                     "phase_advance_quiet_wvel");
                 double wlim = qw >= 0
                     ? model->numeric_data[model->numeric_adr[qw]] : 0.0;
                 if (wlim > 0.0) {
                   quiet = quiet &&
                       mju_sqrt(data->qvel[3] * data->qvel[3] +
                                data->qvel[4] * data->qvel[4]) <= wlim;
                 }
                 // ★ 2026-08-10 `phase_advance_upright_deg`: 0 = OFF =
                 // byte-identical. >0 = ALSO hold the advance while the torso
                 // is pitched more than this (deg). WHY: at seat 17/20 (the
                 // restored-0.985 bench) all THREE failed commits launched
                 // TILTED BACK (+3..+16 deg, base drifted -0.04..-0.16 m) --
                 // calm but mis-postured, which the quiet gate cannot see; the
                 // fixed-length lean then undershoots (face-plant) or recoils
                 // (backward fall). All 17 good runs launched upright.
                 // ESCAPE HATCH: only holds for `phase_advance_upright_wait`
                 // seconds (default 12) past the dwell, then advances anyway --
                 // nothing recenters a parked robot, so an unconditional gate
                 // could hang the ladder forever.
                 int ud = mj_name2id(model, mjOBJ_NUMERIC,
                                     "phase_advance_upright_deg");
                 double udeg = ud >= 0
                     ? model->numeric_data[model->numeric_adr[ud]] : 0.0;
                 if (udeg <= 0.0) return quiet;
                 double over = data->time -
                     motion_strategy_.GetCurrentKeyframeSuccessStartTime() -
                     current_kf.success_sustain_time;
                 double cap = GetNumberOrDefault(
                     12.0, model, "phase_advance_upright_wait");
                 if (over > cap) return quiet;   // escape hatch
                 const double* q = data->qpos;
                 double pitch_deg = 57.29578 * mju_asin(mju_max(-1.0,
                     mju_min(1.0, 2.0 * (q[3] * q[5] - q[6] * q[4]))));
                 return quiet && mju_abs(pitch_deg) <= udeg;
               }()) {
      // Normal phase advance — this is the path that fires after stand_up
      // succeeds. Snapshot first so the new ramp starts from the old scales.
      SnapshotEffectiveScales();
      SnapshotCurrentWeightsAsPrev();
      motion_strategy_.NextKeyframe();
      MarkNewlyAppearedContacts(residual_.residual_keyframe_,
                                motion_strategy_.GetCurrentKeyframe());
      residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
      motion_strategy_.SetCurrentKeyframeStartTime(data->time);
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
      residual_.keyframe_start_time_ = data->time;
      PrepareNextPhaseWeights(residual_.residual_keyframe_);
    } else if (total_distance > current_kf.target_distance_tolerance) {
      // Re-arm the success clock: outside tolerance -- sustain must be
      // CONSECUTIVE.
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    }
  }

  // ★★★ 2026-08-09 CAPTURE THE MEASURED STANCE for the Foot Stability anchor.
  // `foot_anchor_measured` numeric: 0 = OFF = byte-identical (hardcoded x=0.30).
  // WHY: on the twin the feet sit at x ~= 0.005 at harness release, so the 0.30
  // home is a standing 29 cm error the cost drags the feet toward all ladder --
  // 200-250 mm of measured drift in benches v5/v6, and raising the weight made
  // it WORSE, exactly as a drag would.
  {
    int fam = mj_name2id(model, mjOBJ_NUMERIC, "foot_anchor_measured");
    double on = fam >= 0 ? model->numeric_data[model->numeric_adr[fam]] : 0.0;
    const std::string& kfn = motion_strategy_.GetCurrentKeyframe().name;
    if (on <= 0.0) {
      residual_.foot_pin_x_ = std::numeric_limits<mjtNum>::quiet_NaN();
    } else if (std::isnan(residual_.foot_pin_x_) && kfn == "forearm_brace_lean") {
      double const *lp = SensorByName(model, data, "foot_left_pos");
      double const *rp = SensorByName(model, data, "foot_right_pos");
      if (lp && rp && lp[2] < 0.08 && rp[2] < 0.08) {
        residual_.foot_pin_x_ = 0.5 * (lp[0] + rp[0]);
        std::fprintf(stderr, "[lean-foot] stance PINNED at x=%.3f "
                     "(hardcoded home was 0.30)\n", residual_.foot_pin_x_);
      }
    }
  }

  // ★★★ 2026-08-10 PHASE-SCHEDULED CEM VARIANCE FLOOR (`std_min_state_gated`
  // numeric: 0 = OFF = byte-identical -> LiveStdMinOverride stays -1). WHY: the
  // global floor trades SURVIVAL against PRECISION with opposite slopes (n=20
  // arms, estimator-in-loop: floor 0.01 -> 5/20 never-fell but a crisp ladder;
  // 0.05 -> 15/20 never-fell, p~=0.001, but the robot survives by DANCING --
  // seat 15->7, up to 1.3 m of foot travel, LOW hover endings; 0.02 -> 1/20,
  // no survival at all). So: WIDE floor (`std_min_wide`) in the fall-prone
  // free/transition phases, TIGHT floor (`std_min_tight`) in the seated phases
  // (mid/reach/release) where sustained pad contact is what the noise breaks.
  {
    int sg = mj_name2id(model, mjOBJ_NUMERIC, "std_min_state_gated");
    if (sg >= 0 && model->numeric_data[model->numeric_adr[sg]] > 0.0) {
      double wide = 0.05, tight = 0.01;
      int wi = mj_name2id(model, mjOBJ_NUMERIC, "std_min_wide");
      int ti = mj_name2id(model, mjOBJ_NUMERIC, "std_min_tight");
      if (wi >= 0) wide = model->numeric_data[model->numeric_adr[wi]];
      if (ti >= 0) tight = model->numeric_data[model->numeric_adr[ti]];
      const std::string& kf = motion_strategy_.GetCurrentKeyframe().name;
      // ★ v2 (2026-08-10): release REMOVED from the tight set. With v1 (tight =
      // mid/reach/release) the 8 remaining falls all clustered at t=117-145 =
      // exactly the release->standback segment: release is a PUSH-OFF transition,
      // not a hold, and starving it of search width is what dropped it. Survival
      // 12/20 + seat 12/20 in v1 -- first arm to recover both sides of the trade.
      // ★ v3 (2026-08-10): stand_up ADDED to the tight set. v2 (wide everywhere
      // but mid/reach) fixed survival (15/20, release cluster gone) but feet
      // stayed 0/20 with 0.4-1.5 m of drift -- the scoring window is dominated
      // by the FINAL stand phase (t~139-240 s), which sat at the wide floor, so
      // the robot spent ~100 s wandering. Standing needs no width: strat-6
      // free-stands to the episode cap at the default 0.01 floor. Wide is now
      // ONLY the dynamic transitions (lean / release / standback rungs).
      bool seated_phase = (kf == "forearm_brace_mid" ||
                           kf == "forearm_brace_reach" ||
                           kf == "stand_up");
      double target = seated_phase ? tight : wide;
      // ★ v4 (2026-08-10): SMOOTHSTEP the floor between phase targets instead of
      // stepping. v3's step 0.01->0.05 at lean entry killed 20/20 runs at
      // t=70-90: 5x noise injected at the exact commit moment. (v1/v2 never saw
      // this because their stand phase was already wide -- no step -- but their
      // "survival" was partly survival-by-never-committing: dancing robots
      // missing the lean.) The ramp keeps the first seconds of each transition
      // at the previous phase's crispness and grows width for the descent.
      double ramp = 2.5;
      int ri = mj_name2id(model, mjOBJ_NUMERIC, "std_min_ramp_sec");
      if (ri >= 0) ramp = model->numeric_data[model->numeric_adr[ri]];
      if (std_min_target_ < 0.0) {          // first tick after gate-on
        std_min_target_ = target;
        std_min_ramp_from_ = target;
        std_min_ramp_t0_ = data->time;
      } else if (target != std_min_target_) {
        std_min_ramp_from_ = std_min_live_.load();
        std_min_target_ = target;
        std_min_ramp_t0_ = data->time;
      }
      double a = ramp > 0.0
                     ? mju_min(1.0, (data->time - std_min_ramp_t0_) / ramp)
                     : 1.0;
      a = a * a * (3.0 - 2.0 * a);
      std_min_live_.store(std_min_ramp_from_ +
                          a * (std_min_target_ - std_min_ramp_from_));
    } else {
      std_min_live_.store(-1.0);
      std_min_target_ = -1.0;
    }
  }

  // Apply the smoothstep weight ramp every tick. This also propagates the
  // current target weights into weight[] when no phase advance fired, which
  // is what makes the live GUI cost sliders mirror the JSON values.
  ApplyRampedWeights(model, data);
}

// ---------------- Live per-phase weight blending --------------------- //
// All loops over weight[] must be bounded by `weight_names.size()` (== the
// actual number of user-sensor cost terms), NOT `weight.size()` — the Task
// base class resizes `weight` to `kMaxCostTerms` (128) regardless of the
// real term count, while `weight_names` is sized to `num_term`. Indexing
// `weight_names` past num_term reads uninitialised std::strings and
// segfaults when the JSON map then calls .find(garbage).

// Snapshot the XML default weights from sensor user data. Called once per
// Reset so the fallback for "JSON omitted this residual" stays in sync with
// the static XML weights (which Task::Reset itself populated into weight[]).
void lean::SnapshotXmlDefaultWeights(const mjModel *model) {
  const std::size_t n = weight_names.size();
  xml_default_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) xml_default_weights_[i] = weight[i];
}

// Build next_phase_weights_ from this phase's JSON weight map. Missing keys
// fall back to the XML default snapshot. Unknown keys (typos) are silently
// skipped — same forgiving behaviour as interact.cc.
void lean::PrepareNextPhaseWeights(const mjpc::humanoid::ContactKeyframe &kf) {
  const std::size_t n = weight_names.size();
  if (next_phase_weights_.size() != n) next_phase_weights_.assign(n, 0.0);
  if (xml_default_weights_.size() != n) xml_default_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::string &name = weight_names[i];
    auto it = kf.weight.find(name);
    next_phase_weights_[i] =
        (it != kf.weight.end()) ? it->second : xml_default_weights_[i];
  }
}

void lean::SnapshotCurrentWeightsAsPrev() {
  const std::size_t n = weight_names.size();
  prev_phase_weights_.assign(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) prev_phase_weights_[i] = weight[i];
}

// Lerp weight[] from prev → next using the same smoothstep curve the
// residual uses for its phase scales, so the rollouts' cost surface evolves
// continuously across phase boundaries.
// 2026-07-28: shared resolver for the phase ramp so the WEIGHT ramp and the residual
// scale ramp cannot drift apart. `phase_ramp_sec` numeric: 0 = OFF = byte-identical.
static double LeanPhaseRampSeconds(const mjModel *model, double fallback) {
  int pr_id = mj_name2id(model, mjOBJ_NUMERIC, "phase_ramp_sec");
  double pr = (pr_id >= 0) ? model->numeric_data[model->numeric_adr[pr_id]] : 0.0;
  return (pr > 1e-9) ? pr : fallback;
}

void lean::ApplyRampedWeights(const mjModel *model, const mjData *data) {
  const std::size_t n = weight_names.size();
  if (prev_phase_weights_.size() != n || next_phase_weights_.size() != n) {
    return;
  }
  double dt = mju_max(0.0, data->time - residual_.keyframe_start_time_);
  double alpha_lin = mju_min(
      dt / LeanPhaseRampSeconds(model, ResidualFn::kPhaseRampSeconds), 1.0);
  double alpha = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  for (std::size_t i = 0; i < n; ++i) {
    weight[i] = prev_phase_weights_[i] +
                alpha * (next_phase_weights_[i] - prev_phase_weights_[i]);
  }
}

void lean::ResetLocked(const mjModel *model) {
  // Capture the XML default weights AFTER Task::Reset has populated weight[]
  // from sensor user data. These act as the fallback for any residual the
  // current phase's JSON doesn't override, and as the prev/next initialisers
  // before the first phase transition. All three vectors are sized to
  // weight_names.size() (== num_term), NOT weight.size() (== kMaxCostTerms).
  SnapshotXmlDefaultWeights(model);
  prev_phase_weights_ = xml_default_weights_;
  next_phase_weights_ = xml_default_weights_;
  if (!residual_.residual_keyframe_.weight.empty()) {
    PrepareNextPhaseWeights(residual_.residual_keyframe_);
  }

  // Reach target on reset: an external `reach_target` numeric pins it
  // deterministically (Strategy-21 reach / vision input); otherwise the legacy
  // random spawn (matches the same gate in TransitionLocked above).
  int reset_reach_tgt_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_target");
  const bool reset_reach_live =
      (int)parameters.size() > kLeanReachZParameterIndex &&
      parameters[kLeanReachActiveParameterIndex] > 0.5;
  if (reset_reach_live) {
    target_position_ = {parameters[kLeanReachXParameterIndex],
                        parameters[kLeanReachYParameterIndex],
                        parameters[kLeanReachZParameterIndex]};
  } else if (reset_reach_tgt_id >= 0) {
    const mjtNum *rt = model->numeric_data + model->numeric_adr[reset_reach_tgt_id];
    target_position_ = {rt[0], rt[1], rt[2]};
  } else {
    std::random_device rd;
    std::mt19937 gen(rd());
    // TEST #16 (2026-05-18): target x range 1.4-1.6 → 1.2-1.4 (matches
    // the same fix in TransitionLocked above). Missing this caused the
    // FIRST target on every reset to spawn far at [1.4, 1.6] — robot
    // couldn't reach without losing balance, and user saw the tipping.
    std::uniform_real_distribution<> dis_x(1.2, 1.4);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.83};
    printf("New target position: %f, %f, %f\n", target_position_[0],
           target_position_[1], target_position_[2]);
  }

  // DEBUG: Print joint order
  printf("\nJoint order for qpos:\n");
  for (int i = 0; i < model->njnt; i++) {
    const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
    int qpos_adr = model->jnt_qposadr[i];
    printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  }
}


// ============================================================================
// ComputeMetrics — phase-aware monitoring metrics for the Research GUI /
// headless analyzer. Reads the current keyframe + sensor stack; no rollout
// hot-path work. See QUANTIFICATION_PLAN.html for the 10 metrics surfaced
// here (reach, CoP, ICP, brace force, saturation, ...).
// ============================================================================
void lean::ComputeMetrics(const mjModel *model, const mjData *data,
                          std::map<std::string, double> *metrics,
                          std::string *phase_name) const {
  if (metrics) metrics->clear();
  if (phase_name) phase_name->clear();
  if (!metrics) return;

  // ----- Phase identity -------------------------------------------------- //
  const auto &kf = residual_.residual_keyframe_;
  if (phase_name) *phase_name = kf.name;

  // Phase ramp progress (matches Residual()'s smoothstep)
  double time_in_phase =
      mju_max(0.0, data->time - residual_.keyframe_start_time_);
  double alpha_lin =
      mju_min(time_in_phase / ResidualFn::kPhaseRampSeconds, 1.0);
  double alpha_smooth = alpha_lin * alpha_lin * (3.0 - 2.0 * alpha_lin);
  (*metrics)["phase_time"] = time_in_phase;
  (*metrics)["phase_alpha_linear"] = alpha_lin;
  (*metrics)["phase_alpha"] = alpha_smooth;

  // Brace-force target — TWO values exposed so plots can show what's
  // actually happening on the controller side:
  //   brace_force_target_final = the keyframe's destination value
  //                              (what the phase is heading toward).
  //   brace_force_target       = the smoothstep-ramped value the
  //                              residual is currently chasing.
  // Without this distinction the plot's target line jumps in steps at
  // each phase boundary even though the planner is actually pursuing a
  // smooth ramp over kPhaseRampSeconds.
  bool kf_active = (kf.contact_pairs[0].body1 !=
                     mjpc::humanoid::kNotSelectedInteract);
  double target_final = kf.brace_force_target >= 0.0
                            ? kf.brace_force_target
                            : (kf_active ? 70.0 : 0.0);
  double ramped_target = residual_.prev_phase_brace_force_target_ +
      alpha_smooth * (target_final -
                       residual_.prev_phase_brace_force_target_);
  (*metrics)["brace_force_target"] = ramped_target;
  (*metrics)["brace_force_target_final"] = target_final;

  // ----- Sensor reads (bail out if any missing) -------------------------- //
  double *left_hand = SensorByName(model, data, "left_hand_pos");
  double *right_hand = SensorByName(model, data, "right_hand_pos");
  double *right_contact = SensorByName(model, data, "right_hand_contact");
  double *left_contact = SensorByName(model, data, "left_hand_contact");
  double *torso = SensorByName(model, data, "torso_position");
  double *foot_left = SensorByName(model, data, "foot_left_pos");
  double *foot_right = SensorByName(model, data, "foot_right_pos");
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *com_vel = SensorByName(model, data, "torso_subtreelinvel");
  if (!left_hand || !right_hand || !torso || !foot_left || !foot_right ||
      !subcom) {
    return;
  }

  // Right arm always braces, left arm always reaches (lean.cc convention).
  double *reaching_hand = left_hand;
  double *bracing_hand = right_hand;
  double brace_force_normal = right_contact ? right_contact[0] : 0.0;
  double reach_contact_force = left_contact ? left_contact[0] : 0.0;

  // ----- M1: reach-from-pelvis (horizontal) ------------------------------ //
  // Using torso as pelvis proxy — closest sensor we have.
  double rx = reaching_hand[0] - torso[0];
  double ry = reaching_hand[1] - torso[1];
  (*metrics)["reach_from_pelvis"] = mju_sqrt(rx * rx + ry * ry);

  // ----- M2: reach-beyond-foot-edge -------------------------------------- //
  // Forward distance the palm projects past the front-most foot edge.
  // +0.10 m toe offset because foot_*_pos is at the ankle site, not the toe.
  double front_edge_x = mju_max(foot_left[0], foot_right[0]) + 0.10;
  (*metrics)["reach_beyond_foot_edge"] = reaching_hand[0] - front_edge_x;

  // ----- Reach-to-target (Strategy 21) ----------------------------------- //
  // Surface the live reach for the deploy monitor: the auto-picked reaching hand
  // (by target side), the INPUT target (mocap "target" body = data->mocap_pos —
  // NOT the static object_pos), and the residual error to the requested point
  // (to the RAW target, so a far/out-of-reach request reads its true shortfall).
  if (kf.name == "reach_to_target") {
    double const *tgt = data->mocap_pos;
    int rh_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
    int rh_mode = (rh_id >= 0)
        ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id]])
        : 0;
    bool rr = (rh_mode == 2) ? true
            : (rh_mode == 1) ? false
            : (tgt[1] < torso[1]);
    double *rh = rr ? right_hand : left_hand;
    (*metrics)["reach_tgt_x"] = tgt[0];
    (*metrics)["reach_tgt_y"] = tgt[1];
    (*metrics)["reach_tgt_z"] = tgt[2];
    (*metrics)["reach_hand_side"] = rr ? 1.0 : 0.0;  // 1=right, 0=left
    double ex = rh[0] - tgt[0], ey = rh[1] - tgt[1], ez = rh[2] - tgt[2];
    (*metrics)["reach_err"] = mju_sqrt(ex * ex + ey * ey + ez * ez);
  }

  // ----- M3: CoM excursion from midfoot ---------------------------------- //
  double midfoot_x = 0.5 * (foot_left[0] + foot_right[0]);
  double midfoot_y = 0.5 * (foot_left[1] + foot_right[1]);
  double com_dx = subcom[0] - midfoot_x;
  double com_dy = subcom[1] - midfoot_y;
  (*metrics)["com_excursion_sagittal"] = com_dx;
  (*metrics)["com_excursion_lateral"] = com_dy;
  (*metrics)["com_excursion"] = mju_sqrt(com_dx * com_dx + com_dy * com_dy);
  (*metrics)["com_x"] = subcom[0];
  (*metrics)["com_y"] = subcom[1];
  (*metrics)["com_z"] = subcom[2];

  // ----- M4: CoP / ZMP from floor contacts ------------------------------- //
  // Floor geom is conventionally named "floor" in MJCF; lookup once.
  // Approximation: world vertical component of each contact = f_normal * n_z,
  // where n_z = c.frame[2] (z-component of contact normal). Tangential
  // components contribute negligible vertical force on a flat floor.
  int floor_geom_id = mj_name2id(model, mjOBJ_GEOM, "floor");
  double cop_x_num = 0.0, cop_y_num = 0.0, cop_denom = 0.0;
  double foot_force_total = 0.0;
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact &c = data->contact[i];
    bool involves_floor = (floor_geom_id >= 0) &&
                          (c.geom[0] == floor_geom_id ||
                           c.geom[1] == floor_geom_id);
    if (!involves_floor) continue;
    double f6[6];
    mj_contactForce(model, data, i, f6);
    double n_z = c.frame[2];
    double fz = f6[0] * n_z;
    if (fz > 0.0) {
      cop_x_num += c.pos[0] * fz;
      cop_y_num += c.pos[1] * fz;
      cop_denom += fz;
      foot_force_total += fz;
    }
  }
  if (cop_denom > 0.0) {
    (*metrics)["cop_x"] = cop_x_num / cop_denom;
    (*metrics)["cop_y"] = cop_y_num / cop_denom;
  } else {
    (*metrics)["cop_x"] = midfoot_x;
    (*metrics)["cop_y"] = midfoot_y;
  }
  (*metrics)["foot_force_total"] = foot_force_total;

  // ----- M5: brace force ------------------------------------------------- //
  // Bracing hand normal contact (right). The Hands variant has an additional
  // right_palm_contact sensor — surface it separately when present.
  (*metrics)["brace_force"] = brace_force_normal;
  (*metrics)["reach_hand_contact_force"] = reach_contact_force;
  double *right_palm_contact =
      SensorByName(model, data, "right_palm_contact");
  if (right_palm_contact) {
    (*metrics)["palm_contact_force"] = right_palm_contact[0];
  }
  // Force-distribution ratio: how much vertical reaction is carried by the
  // bracing hand vs the feet. Diagnostic for "is the brace actually loaded".
  double total_vertical = foot_force_total + brace_force_normal;
  (*metrics)["brace_force_ratio"] =
      (total_vertical > 1e-6) ? brace_force_normal / total_vertical : 0.0;

  // ----- M6: friction slack on bracing-hand contacts --------------------- //
  // s = μ|F_normal| − ||F_tangential||. Positive = inside friction cone.
  // We iterate contacts and track the minimum slack across any contact that
  // involves NOT-the-floor (palm/elbow on table). For a flat-floor task this
  // is good enough because the floor contacts are foot-on-floor (high slack).
  double min_brace_slack = std::numeric_limits<double>::infinity();
  bool any_brace_contact = false;
  for (int i = 0; i < data->ncon; ++i) {
    const mjContact &c = data->contact[i];
    bool involves_floor = (floor_geom_id >= 0) &&
                          (c.geom[0] == floor_geom_id ||
                           c.geom[1] == floor_geom_id);
    if (involves_floor) continue;
    double f6[6];
    mj_contactForce(model, data, i, f6);
    double fn = mju_abs(f6[0]);
    double ft = mju_sqrt(f6[1] * f6[1] + f6[2] * f6[2]);
    double mu = c.friction[0];
    double slack = mu * fn - ft;
    if (slack < min_brace_slack) min_brace_slack = slack;
    any_brace_contact = true;
  }
  if (any_brace_contact) {
    (*metrics)["brace_friction_slack"] = min_brace_slack;
  } else {
    (*metrics)["brace_friction_slack"] = 0.0;
  }

  // ----- M7: torque saturation (max fraction across actuators) ----------- //
  // |ctrl[i] − midpoint| / half-range, max over actuators. Uses MuJoCo's
  // actuator_forcerange (set by ctrlrange / forcerange in MJCF / URDF).
  double max_torque_frac = 0.0;
  for (int i = 0; i < model->nu; ++i) {
    double lo = model->actuator_forcerange[2 * i];
    double hi = model->actuator_forcerange[2 * i + 1];
    if (hi > lo) {
      double range = 0.5 * (hi - lo);
      double mid = 0.5 * (hi + lo);
      double frac = mju_abs(data->ctrl[i] - mid) / range;
      if (frac > max_torque_frac) max_torque_frac = frac;
    }
  }
  (*metrics)["torque_saturation_max"] = max_torque_frac;

  // ----- M8: max joint velocity (raw — % vs URDF limits in analyzer) ----- //
  double max_vel = 0.0;
  for (int i = 6; i < model->nv; ++i) {  // skip free-joint base
    double v = mju_abs(data->qvel[i]);
    if (v > max_vel) max_vel = v;
  }
  (*metrics)["joint_velocity_max"] = max_vel;

  // ----- Instantaneous Capture Point ------------------------------------ //
  // ξ = r_CoM + ṙ_CoM · √(z_CoM / g). Cheap predictive stability margin.
  if (com_vel) {
    double z = mju_max(subcom[2], 0.1);
    double omega = mju_sqrt(9.81 / z);
    (*metrics)["icp_x"] = subcom[0] + com_vel[0] / omega;
    (*metrics)["icp_y"] = subcom[1] + com_vel[1] / omega;
  }

  // ----- Support-polygon excursion (B6) ---------------------------------- //
  // Signed forward distance from CoM ground projection to front foot edge.
  // Positive = CoM is forward of the foot polygon (only safe with brace).
  (*metrics)["com_beyond_foot_edge"] = subcom[0] - front_edge_x;

  // ----- ZMP/CoP support-polygon excursion (B6b) ------------------------- //
  // Same sign convention as com_beyond_foot_edge, but for the MEASURED ZMP
  // (cop_x from M4) instead of the CoM. Positive = CoP forward of the front
  // foot edge. Because cop_x is force-weighted over FLOOR contacts only, the
  // ZMP stays pinned inside the feet even while the CoM pushes out over the
  // table edge during a brace -- so the gap between the two margins is exactly
  // the load the bracing hand is carrying.
  (*metrics)["cop_beyond_foot_edge"] = (*metrics)["cop_x"] - front_edge_x;

  // ----- Foot positions exposed for SP top-down panel -------------------- //
  (*metrics)["foot_left_x"] = foot_left[0];
  (*metrics)["foot_left_y"] = foot_left[1];
  (*metrics)["foot_left_z"] = foot_left[2];
  (*metrics)["foot_right_x"] = foot_right[0];
  (*metrics)["foot_right_y"] = foot_right[1];
  (*metrics)["foot_right_z"] = foot_right[2];

  // ----- Bracing-hand position exposed for SP-with-hand panel ------------ //
  (*metrics)["brace_hand_x"] = bracing_hand[0];
  (*metrics)["brace_hand_y"] = bracing_hand[1];
  (*metrics)["brace_hand_z"] = bracing_hand[2];
  (*metrics)["reach_hand_x"] = reaching_hand[0];
  (*metrics)["reach_hand_y"] = reaching_hand[1];
  (*metrics)["reach_hand_z"] = reaching_hand[2];

  // ----- Target (object) position + palm-to-target distance. Saved on
  // every row so post-analysis can compute exact reach success without
  // having to recover the random spawn position from mjpc stdout.
  double *object_pos_m = SensorByName(model, data, "object_pos");
  if (object_pos_m) {
    (*metrics)["target_x"] = object_pos_m[0];
    (*metrics)["target_y"] = object_pos_m[1];
    (*metrics)["target_z"] = object_pos_m[2];
    double dx = reaching_hand[0] - object_pos_m[0];
    double dy = reaching_hand[1] - object_pos_m[1];
    double dz = reaching_hand[2] - object_pos_m[2];
    (*metrics)["palm_to_target"] = mju_sqrt(dx*dx + dy*dy + dz*dz);
  }
}

}  // namespace mjpc