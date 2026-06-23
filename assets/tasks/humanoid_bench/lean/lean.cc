#include "mjpc/tasks/humanoid_bench/lean/lean.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc {

namespace {
// Target (post-ramp) reach + brace + posture scales for each named phase. Kept
// in one place so the residual and the transition logic can't drift out of
// sync. Posture is boosted during stand_up because the audit-spec PD gains
// (ankle kp=20, knee kp=200) aren't stiff enough on their own to pull a
// drifted knee back to extension — the Posture cost has to do it via MPC.
//
// New two-step sequence: arm_plant → lean_forward. Splits the old merged
// brace_and_lean phase so the bracing hand makes contact (with a small brace
// force target supplied per-keyframe via JSON) BEFORE the torso commits to
// the full forward lean. arm_plant tolerates only a small torso tilt so MPC
// prioritises getting the hand on the table; lean_forward then unlocks the
// full Torso Forward Tilt residual while the planted hand stays put.
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
  // arm_plant: get the bracing hand onto the table first. Torso tilt is
  // softly allowed (reach=0.2) just enough for arm geometry, while the
  // bracing-hand position residual is fully active. Brace force target (8N)
  // is supplied per-keyframe via JSON brace_force_target.
  else if (name == "arm_plant") {
    reach = 0.2; brace_pos = 1.0; posture = 1.0;
  }
  // lean_forward: hand is now planted — unlock the full Torso Forward Tilt
  // residual so the body commits to the lean. Bracing-hand position stays
  // fully active to keep the hand on the table. Brace force target (15N)
  // supplied per-keyframe via JSON.
  else if (name == "lean_forward") {
    reach = 1.0; brace_pos = 1.0; posture = 1.0;
  }
  // arm_extend_standing: body upright, right arm reaches forward to the
  // table edge contact target. The strategy JSON also boosts Pelvis Tilt
  // and zeroes Torso Forward Tilt so the body resists tipping while the
  // arm extends. Posture×1.5 keeps the legs locked extended.
  else if (name == "arm_extend_standing") {
    reach = 1.0; brace_pos = 0.0; posture = 1.5;
  }
  // counterbalance_standing: dedicated STANDING counterbalance skill (Strategy
  // 16). Like arm_extend_standing the left arm reaches forward and the feet
  // stay planted, BUT the reach target is overridden to SHOULDER height (see
  // Residual()) so the arm extends horizontally instead of reaching down toward
  // the table-edge brace point — that downward, unreachable target was what
  // made the arm tuck/crunch into the torso. Posture×1.0 (NOT 1.5) leaves the
  // free (right) arm loose so it can swing back as an emergent counterweight.
  else if (name == "counterbalance_standing") {
    reach = 1.0; brace_pos = 0.0; posture = 1.0;
  }
  // reach_to_target: standalone "reach an input target" primitive (Strategy 21).
  // Like arm_extend_standing the chosen hand reaches toward a world point and
  // the feet stay planted with NO brace, but the target is the EXTERNAL mocap
  // object (object_pos, clamped to a balance-safe workspace — see Residual())
  // instead of a fixed forward pose. posture=1.0 (NOT 1.5): the legs are locked
  // by dedicated terms (Knees Straight / Base Height / Symmetry in the JSON),
  // so a low global Posture leaves the reaching ARM free to extend (the jab
  // lesson — a high Posture parks the limb at its rest pose).
  else if (name == "reach_to_target") {
    reach = 1.0; brace_pos = 0.0; posture = 1.0;
  }
  // lean_with_arm_no_brace: arm stays out from the previous phase but the
  // body now commits to a forward lean WITHOUT making table contact. This
  // is the "lean-and-catch-yourself-just-in-time" beat; the contact lands
  // in the next phase. Pelvis Tilt + Height gates allow the lean (see
  // Residual()).
  else if (name == "lean_with_arm_no_brace") {
    reach = 1.0; brace_pos = 0.0; posture = 1.0;
  }
  // forearm_brace_lean: deeper braced lean with the elbow now on the
  // table in addition to the hand. Behaves like lean_forward from a
  // scaling standpoint; the extra elbow contact is just another
  // ContactPair in the keyframe.
  else if (name == "forearm_brace_lean") {
    reach = 1.0; brace_pos = 1.0; posture = 1.0;
  }
  // Legacy phase names kept for backwards compatibility — if a strategy still
  // uses them, they ramp in like before.
  else if (name == "brace_and_lean") {
    reach = 1.0; brace_pos = 1.0; posture = 1.0;
  }
  else if (name == "arm_extend") { reach = 0.3; brace_pos = 1.0; posture = 1.0; }
  else if (name == "lean_plant") { reach = 0.7; brace_pos = 1.0; posture = 1.0; }
  // lean_reach / lean_reach_ext / leg_lift_arm_plant / deep_reach → 1.0/1.0/1.0.
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

  // ----- stage detection: used throughout to gate residual scaling ----- //
  int active_contact_count_early = 0;
  for (const auto& cp : residual_keyframe_.contact_pairs) {
    if (cp.body1 != mjpc::humanoid::kNotSelectedInteract &&
        cp.body2 != mjpc::humanoid::kNotSelectedInteract) {
      active_contact_count_early++;
    }
  }
  // any_arm_contact: arm is on the table (stand_up has 0 contacts)
  const bool any_arm_contact      = (active_contact_count_early >= 1);
  // is_lean_no_brace_phase: the "lean forward without bracing" beat. No
  // contacts but the body IS supposed to tilt, so several gates (Pelvis
  // Tilt, Height, Foot Stability) need to behave as if the arm were on
  // the table even though it isn't yet. Keyed on phase name, not contact
  // count, because the contact count is genuinely zero here.
  const bool is_lean_no_brace_phase =
      (residual_keyframe_.name == "lean_with_arm_no_brace");
  const bool arm_contact_or_lean = any_arm_contact || is_lean_no_brace_phase;
  // ITER 36 (2026-05-18): is_leg_lift detection is PHASE-NAME based (keyed on
  // the active keyframe name), not contact-count based.
  //
  // DESIGN (2026-05-26): the leg-lift phase is DROPPED permanently — both feet
  // stay grounded through the whole pipeline (see the lean.h header). NO
  // strategy JSON contains "leg_lift_arm_plant" / "deep_reach" anymore, so
  // is_leg_lift_stage_early is now ALWAYS FALSE and every `is_leg_lift_stage`
  // branch below is DORMANT (kept as vestigial history, not removed) — the
  // grounded branch (both feet anchored) always runs. Any future single-
  // support move must come from a WBC/balance decision, not this hard gate.
  const bool is_leg_lift_stage_early =
      (residual_keyframe_.name == "leg_lift_arm_plant" ||
       residual_keyframe_.name == "deep_reach");

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
  double alpha_lin     = mju_min(time_in_phase / kPhaseRampSeconds, 1.0);
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
  // pipeline phase ("stand_up", "arm_plant", …) falls back to key 0 (home)
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
  if (num_phases_ > 1 && prev_posture_key_id_ != posture_key_id &&
      model->nq <= 64) {
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

  // ==================== STUMBLE: gait clock + swing-leg JOINT reference ===== //
  // Strategy 20 "stumble" stepping. (Full rationale at the Gait/Step Place
  // residual terms near the end of this function.) The gait CLOCK and the swing-
  // leg JOINT reference are computed HERE -- before the Posture cost reads
  // posture_target -- so the swing leg's hip/knee/ankle TARGETS fold up each
  // cycle and the strong, well-behaved Posture cost DRIVES the foot off the
  // ground. That turns stepping into a TRACKING problem (which sampling MPC
  // solves trivially, like every pose strategy tracking a keyframe) instead of a
  // SEARCH problem -- the planner will not spontaneously cross the cost barrier
  // of a half-formed step out of the stable-stand local minimum at calm
  // exploration. The Cartesian Gait/Step Place terms at the END of the function
  // refine foot height + xy placement, reusing g_amp/g_bump_* computed here.
  // NAME-GATED on "stumble": every other strategy (0-19) skips this block and
  // its two end-of-function residual terms default to weight 0 -> byte-identical.
  mjtNum stumble_posture_target[64];  // private swing-leg reference (stumble only)
  const bool is_stumble = (residual_keyframe_.name.rfind("stumble", 0) == 0);
  // gait parameters (constexpr -> tune by edit+rebuild; nav drives only kDesVel*)
  constexpr double kCadenceHz  = 0.8;   // steps/s per foot (slower -> swing fits fewer
                                        // spline knots, so it steps at a stand-tolerable
                                        // bandwidth: spline 8 destabilises even a plain
                                        // stand, spline<=5 holds. 2026-06-18.)
  constexpr double kDutyRatio  = 0.70;  // stance fraction (more double-support = steadier)
  constexpr double kStepHeight = 0.022; // swing-foot peak Cartesian clearance [m] (gentle shuffle)
  constexpr double kAmpRampSec = 4.5;   // ease the gait in over the first 4.5 s of the phase
  constexpr double kDesVelX    = 0.0;   // desired CoM velocity [m/s] world x -- NAV HOOK
  constexpr double kDesVelY    = 0.0;   // desired CoM velocity [m/s] world y -- NAV HOOK
  // swing-leg JOINT-space lift offsets [rad] at full bump (added to the keyframe
  // target): fold the leg up so the foot clears ground. hip flexes back, knee
  // bends, ankle holds the sole level. This is what BOOTSTRAPS the step; the
  // leg hip_pitch/knee entries are additionally leg-gain-amplified by the
  // Posture cost, so even a modest Posture weight tracks the swing crisply.
  constexpr double kSwingHip   = 0.11;  // hip_pitch: more negative -> thigh lifts (gentler)
  constexpr double kSwingKnee  = 0.20;  // knee: more positive -> shank folds up (gentler)
  constexpr double kSwingAnk   = 0.08;  // ankle_pitch: more negative -> foot stays level
  // ---- BALANCE-GATED stepping (2026-06-19; signed-danger gate 2026-06-20) ----
  // STAND STILL (bent-knee, g_amp=0) and ramp the gait in ONLY while balance is
  // genuinely being lost; ease back to a still stand once recovered. "Losing
  // balance" = the SIGNED capture-point danger (computed once below) crossing
  // catch_trig -- the SAME quantity the catch-step + hip/arm tier use. The old
  // gate keyed on ABSOLUTE base tilt (deadband 5.7deg) + ABSOLUTE |vx|, which the
  // real-robot bring-up settle (leans ~8.7deg) and recovery motion both exceed ->
  // it MARCHED while settling ("never stood still") and kept marching through the
  // recovery ("kept twisting legs"); the twin settles ~3-4deg so it never showed
  // it (live-only). Signed danger is recovery-aware: a settling/recovering lean is
  // NOT treated as imbalance. Toggle off with model numeric
  // stumble_balance_gated=0 to restore the old always-on march (A/B, no rebuild).
  constexpr double kArmSec = 2.0;     // arm the gate this long after engage (calm bring-up)
  double g_amp = 0.0, g_bump_l = 0.0, g_bump_r = 0.0;
  // Capture-point excursion (CoM heading vs equilibrium), hoisted to function
  // scope so BOTH the catch-step (below) AND the hip/arm angular-momentum
  // recovery tier (Centroidal angular momentum term, far below) read the same
  // signal -- the ankle->HIP->step hierarchy keyed off one quantity.
  double g_cap_ex = 0.0, g_cap_ey = 0.0;
  // TROT-window step scales (function scope so the Cartesian Gait/Step-Place
  // block far below sees them). Both default 1.0 (== quiet stand / push-recovery
  // BYTE-IDENTICAL) and only become >1 inside the trot starter window. Per the
  // MJPC-quadruped + Unitree-RL-walk references: a stable in-place step needs the
  // foot-CLEARANCE term to DOMINATE (quadruped Gait 2.0 = highest; Unitree
  // feet_swing_height = strongest), NOT a boosted joint reference the balance-
  // cautious planner ignores. trot_swing_scale raises the swing-foot height TARGET
  // (joint ref + Cartesian arc); trot_gait_wscale boosts the Gait foot-clearance
  // residual = an effective weight bump (25 -> ~25*scale^2) so foot-tracking wins
  // over the cheap lateral rock. Quiet stand keeps Gait 25 (residual ~0 when
  // planted anyway), so the validated 9/10 stand + push-recovery are untouched.
  double trot_swing_scale = 1.0, trot_gait_wscale = 1.0;
  if (is_stumble) {
    // gait CLOCK always runs; only the AMPLITUDE g_amp decides whether it steps.
    // Cadence is a live numeric (default kCadenceHz) so the operator can dial the
    // step rate -- gentler/fewer steps for the trot starter -- without a rebuild.
    int cad_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_cadence");
    double kCad = (cad_id >= 0)
        ? model->numeric_data[model->numeric_adr[cad_id]] : kCadenceHz;
    // stumble_swing_scale (default 1) multiplies the swing-leg lift DURING the
    // trot starter only (swing_mult below), so the operator can dial step HEIGHT
    // up to clear visibility live -- the balance-cautious planner damps the
    // reference, so the realized lift is a fraction of it. The catch-step keeps
    // mult=1 (validated push-recovery untouched).
    int ss_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_swing_scale");
    double kSwingScale = (ss_id >= 0)
        ? model->numeric_data[model->numeric_adr[ss_id]] : 1.0;
    // stumble_gait_boost (default 1): trot-window multiplier on the Gait foot-
    // clearance RESIDUAL -> effective weight 25*boost^2 (boost 2.6 -> ~170,
    // making foot-tracking the DOMINANT term per the references). Live-tunable.
    int gb_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_gait_boost");
    double kGaitBoost = (gb_id >= 0)
        ? model->numeric_data[model->numeric_adr[gb_id]] : 1.0;
    double ph_l = std::fmod(data->time * kCad,       1.0);  // L foot phase
    double ph_r = std::fmod(data->time * kCad + 0.5, 1.0);  // R antiphase
    g_bump_l = (ph_l < kDutyRatio) ? 0.0
        : std::sin(mjPI * (ph_l - kDutyRatio) / (1.0 - kDutyRatio));
    g_bump_r = (ph_r < kDutyRatio) ? 0.0
        : std::sin(mjPI * (ph_r - kDutyRatio) / (1.0 - kDutyRatio));
    double const *cvel = SensorByName(model, data, "waist_lower_subcomvel");
    double const *flp  = SensorByName(model, data, "foot_left_pos");
    double const *frp  = SensorByName(model, data, "foot_right_pos");
    // --- SIGNED CAPTURE-POINT DANGER (computed ONCE; drives the whole
    //     ankle->HIP->step hierarchy: the march amplitude g_amp, the catch-step
    //     foot choice, and the hip/arm angular-momentum recovery tier far below
    //     via g_cap_ex/ey). e = z*tilt_dir + tau*com_vel (meters from upright):
    //   * base TILT (torso_up) = a LAG-FREE CoM-displacement proxy (IMU-direct;
    //     the HW velocity estimate lags ~30 ms), measured RELATIVE to the steady
    //     ~3-4deg forward lean (lean_nominal_x) so the catch is SYMMETRIC about
    //     equilibrium (a back push immediately gives ex<0), and
    //   * com_vel * sqrt(z/g) = the capture-point velocity LEAD.
    //   danger = |SIGNED capture excursion| = sqrt(ex^2 + (z*ty)^2). SIGNED so a
    //   RECOVERING velocity SHRINKS danger (a settling lean already moving back is
    //   NOT "losing balance"); the LATERAL axis keeps tilt only (z*ty) and DROPS
    //   lateral velocity vy = the gait weight-shift ROCK, which would self-trigger.
    double const *tup = SensorByName(model, data, "torso_up");
    int pid = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pid < 0) pid = 1;
    const mjtNum *com = data->subtree_com + 3 * pid;
    double zc  = mju_max(0.5, com[2]);
    double tau = mju_sqrt(zc / 9.81);
    int lnx_id = mj_name2id(model, mjOBJ_NUMERIC, "lean_nominal_x");
    double kLeanX = (lnx_id >= 0)
        ? model->numeric_data[model->numeric_adr[lnx_id]] : 0.06;
    double tx = (tup ? tup[0] : 0.0) - kLeanX, ty = tup ? tup[1] : 0.0;
    double vx = cvel ? cvel[0] : 0.0, vy = cvel ? cvel[1] : 0.0;
    double ex = zc * tx + tau * vx;                      // signed fore-aft capture
    double ey = zc * ty + tau * vy;                      //   (full, for foot choice)
    g_cap_ex = ex; g_cap_ey = ey;         // share with the hip/arm recovery tier
    double ey_pos = zc * ty;                             // lateral: tilt only (rock-immune)
    double danger = mju_sqrt(ex * ex + ey_pos * ey_pos);
    int ct_id = mj_name2id(model, mjOBJ_NUMERIC, "catch_trig");
    int cf_id = mj_name2id(model, mjOBJ_NUMERIC, "catch_full");
    double kCatchTrig = (ct_id >= 0)
        ? model->numeric_data[model->numeric_adr[ct_id]] : 0.085;
    double kCatchFull = (cf_id >= 0)
        ? model->numeric_data[model->numeric_adr[cf_id]] : 0.16;
    // recov: 0 below catch_trig (STAND STILL), smoothstep to 1 at catch_full.
    double recov = mju_min(1.0, mju_max(0.0,
        (danger - kCatchTrig) / mju_max(1e-3, kCatchFull - kCatchTrig)));
    recov = recov * recov * (3.0 - 2.0 * recov);         // smoothstep
    double arm = mju_min(time_in_phase / kArmSec, 1.0);
    arm = arm * arm * (3.0 - 2.0 * arm);                 // calm bring-up, no spurious step
    // --- AMPLITUDE: balance-gated (default) or legacy continuous march ---
    int bg_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_balance_gated");
    bool balance_gated =
        (bg_id < 0) || (model->numeric_data[model->numeric_adr[bg_id]] > 0.5);
    if (balance_gated) {
      g_amp = arm * recov;   // STAND STILL until the signed capture point escapes
    } else {
      double amp_r = mju_min(time_in_phase / kAmpRampSec, 1.0);
      g_amp = amp_r * amp_r * (3.0 - 2.0 * amp_r);       // legacy continuous march
    }
    // --- TROT STARTER (opt-in, strategy 20 only): a deliberate in-place march
    // over the window [trot_delay, trot_delay+trot_sec] after engage, then HAND
    // OFF to the quiet balance-gated stand above ("trot 10 s -> stand & stumble").
    // Lets the operator SEE balance-during-stepping at startup without waiting
    // for a push. trot_delay skips the node's bring-up ramp; the window eases in
    // AND out over kTrotEase so start/stop are smooth (no snap). g_amp =
    // max(gated, env) so a genuine push DURING the trot still drives a decisive
    // catch-step. The march reuses the same clock/swing as the catch-step
    // (g_bump_l/r below), so the arm-quiet velocity split (leg_gate = 1 - g_amp)
    // damps the upper body throughout. trot_sec<=0 => OFF => byte-identical to
    // the pre-trot gait (and every other strategy stays untouched, name-gated).
    int td_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_trot_delay");
    int tn_id = mj_name2id(model, mjOBJ_NUMERIC, "stumble_trot_sec");
    double kTrotDelay = (td_id >= 0)
        ? model->numeric_data[model->numeric_adr[td_id]] : 0.0;
    double kTrotSec = (tn_id >= 0)
        ? model->numeric_data[model->numeric_adr[tn_id]] : 0.0;
    double t_trot = time_in_phase - kTrotDelay;
    if (kTrotSec > 1e-3 && t_trot >= 0.0 && t_trot < kTrotSec) {
      constexpr double kTrotEase = 1.5;                  // s, smooth ease in AND out
      double rin  = mju_min(t_trot / kTrotEase, 1.0);
      double rout = mju_min((kTrotSec - t_trot) / kTrotEase, 1.0);
      double env  = mju_max(0.0, mju_min(rin, rout));
      env = env * env * (3.0 - 2.0 * env);               // smoothstep envelope
      g_amp = mju_max(g_amp, env);                       // force the march, keep push-step
      trot_swing_scale = kSwingScale;   // taller swing TARGET (joint ref + Cartesian arc)
      trot_gait_wscale = kGaitBoost;    // dominant foot-clearance cost (the lift driver)
    }
    // --- DECISIVE OMNIDIRECTIONAL catch-step: pick the foot toward the capture
    // point and force it to swing NOW (N-step capturability, Pratt/Koolen --
    // relocate the base under the falling CoM before the weak ankle saturates).
    // In gated mode g_amp already reflects danger (arm*recov); in legacy mode
    // SNAP it (arm-gated, so bring-up stays calm). fwd->trailing foot fwd,
    // back->front foot back, left/right->falling-side foot out.
    if (cvel && flp && frp && danger > kCatchTrig) {
      g_amp = mju_max(g_amp, arm * recov);
      bool stepL;
      if (std::fabs(ey) > std::fabs(ex)) {
        stepL = (ey > 0.0);                  // falling LEFT -> left foot out left
      } else {
        stepL = (ex > 0.0) ? (flp[0] <= frp[0])    // forward -> trailing(rear) foot
                           : (flp[0] >= frp[0]);   // back -> front foot
      }
      if (stepL) g_bump_l = mju_max(g_bump_l, arm * recov);
      else       g_bump_r = mju_max(g_bump_r, arm * recov);
    }
    if (model->nq <= 64) {
      for (int i = 0; i < model->nq; i++)
        stumble_posture_target[i] = posture_target[i];
      double ll = g_amp * g_bump_l * trot_swing_scale, lr = g_amp * g_bump_r * trot_swing_scale;  // lift per leg
      // L leg joints: qpos 7+1 hip_pitch, 7+3 knee, 7+4 ankle_pitch
      stumble_posture_target[7 + 1] -= kSwingHip  * ll;
      stumble_posture_target[7 + 3] += kSwingKnee * ll;
      stumble_posture_target[7 + 4] -= kSwingAnk  * ll;
      // R leg joints: qpos 7+7 hip_pitch, 7+9 knee, 7+10 ankle_pitch
      stumble_posture_target[7 + 7]  -= kSwingHip  * lr;
      stumble_posture_target[7 + 9]  += kSwingKnee * lr;
      stumble_posture_target[7 + 10] -= kSwingAnk  * lr;
      posture_target = stumble_posture_target;
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

  // Auto-arm selection shared by counterbalance + forearm-brace phases (the
  // reach_to_target branch does its own identical pick). reach_hand numeric:
  // 0 = AUTO (mocap target y < torso y -> right hand reaches), 1 = force LEFT,
  // 2 = force RIGHT. The OTHER arm always braces/counterweights.
  int rh_id_sel = mj_name2id(model, mjOBJ_NUMERIC, "reach_hand");
  int rh_sel = (rh_id_sel >= 0)
      ? (int)std::lround(model->numeric_data[model->numeric_adr[rh_id_sel]])
      : 0;
  bool reach_right = (rh_sel == 2) ? true
                   : (rh_sel == 1) ? false
                   : (data->mocap_pos[1] < torso_pos[1]);
  double const *reaching_hand = reach_right ? right_hand_pos : left_hand_pos;
  double const *bracing_hand  = reach_right ? left_hand_pos  : right_hand_pos;

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

  double reward = 0;

  // Bracing position calculation. Reverted Y-clamp (was test 14) → back
  // to bracing_hand[1] (test 12 state). User confirmed test 14 introduced
  // chaotic early-phase behaviour. Y free means no restoring force on
  // lateral position; the eventual ~60s slip seen in test 12 is the
  // known trade-off for accepting this baseline.
  double const *table_pos = SensorByName(model, data, "table_surface_pos");
  // torso_pos declared above (near arm-selection) to keep one definition.

  // ----- Phase-dependent reach target ------------------------------------//
  // For most phases the reach residual targets the actual `object_pos`
  // (set by mocap to wherever the next task object is). But during the
  // posture-driven `arm_extend_standing` phase we DON'T want the body to
  // twist toward an off-centre object — the goal is "arm forward, body
  // squared up." So we override the target to a fixed point directly in
  // front of the left shoulder. Reach gradient pulls the LEFT hand
  // straight forward, no lateral component, no body-twist incentive.
  //
  // Position is computed in WORLD frame assuming body is upright (which
  // is enforced by the JSON's Pelvis Tilt=300 boost during this phase):
  //   x = torso_pos[0] + 0.55  → arm-length forward of the body
  //   y = torso_pos[1] + 0.20  → at the left shoulder y-offset
  //   z = torso_pos[2] − 0.05  → roughly shoulder height
  double phase1_target_storage[3];
  double const *reach_target = object_pos;
  if (residual_keyframe_.name == "arm_extend_standing") {
    phase1_target_storage[0] = torso_pos[0] + 0.55;
    phase1_target_storage[1] = torso_pos[1] + 0.20;
    phase1_target_storage[2] = torso_pos[2] - 0.05;
    reach_target = phase1_target_storage;
  }
  // counterbalance_standing (Strategy 16): FOOT-ANCHORED (lean-invariant) reach
  // target so a relaxed Pelvis Tilt lets the body PITCH FORWARD into a leaning
  // counterbalance WITHOUT runaway. The feet stay planted, so this point is
  // effectively world-fixed: as the torso bows toward it the reach error SHRINKS
  // (self-limiting) — unlike a torso-relative target, which would translate
  // forward with the lean and chase itself into a face-plant. Placed 0.70 m
  // forward of midfoot — well beyond the upright arm's horizontal reach — so the
  // body must lean forward AND fully EXTEND the left arm to get there. A nearer
  // target (0.55) let the elbow stay folded; pushing it out straightens the reach
  // so the hand extends further in front. The free right arm + hips swing back to
  // counterbalance. z = 0.75 is BELOW torso (~1.03) so reach_dir points
  // forward-DOWN — that is what lets `Torso Forward Tilt` (JSON weight, off in the
  // upright variant) pitch the torso FORWARD into the reach instead of leaning
  // back; a shoulder-height target gave reach_dir UP, so the only balance response
  // to the forward arm was a BACKWARD lean (measured −7.5°). Forward distance and
  // lean depth are the SAME knob: a further/lower target = deeper lean = bigger
  // counter-arm swing. Pipeline's `arm_extend_standing` override (above) untouched.
  else if (residual_keyframe_.name == "counterbalance_standing") {
    // Counterbalance (Strategy 16 pre-lean + pipeline stage 33): the reaching
    // arm pulls toward the LIVE mocap object (world-fixed, so the reach error
    // shrinks as the body bows in -> self-limiting, no runaway). NO sphere clamp
    // (unlike reach_to_target): an out-of-reach object is exactly what makes the
    // torso lean forward, with the free arm + hips swinging back to counterweight.
    // Lean depth is bounded by Pelvis Tilt / Torso Forward Tilt (JSON lean knobs).
    mju_copy3(phase1_target_storage, data->mocap_pos);
    reach_target = phase1_target_storage;
    // INVESTIGATED 2026-06-23: a deliberate static counterweight (free arm swung
    // back +/- knee bend) was trialed to add forward-push margin, but EVERY forced
    // posture override (arm-back >= 0.30 rad OR knee-bend 0.18) toppled it BACKWARD
    // during lean establishment -- the planner's EMERGENT counterbalance is already
    // optimal and a forced pose disrupts it. Kept the validated emergent behavior
    // (twin + GUI 3/3). The forward-push fragility of this FREE-STANDING no-brace
    // lean is inherent (planted feet, no step); the push-robust paths are BRACING
    // on the surface (strat 33/34) or STEPPING (strat 20), not strat 16.
  }
  // jab_extend (Strategy 19): the RIGHT arm punches straight forward. Posture
  // alone (one arm joint out of ~30) is too weak a reward for the warm-started
  // policy to commit to the big shoulder excursion — it leaves the arms hovering
  // near the guard pose (user report: "guard held for a minute, never jabs").
  // Give the punch a DEDICATED reach reward: repoint the reaching hand to the
  // RIGHT fist and target a fixed point one arm-length forward of the right
  // shoulder at shoulder height — a mirror of the proven arm_extend_standing
  // forward override with y flipped to the right side. The reach residual
  // (Reaching Hand Dist, enabled ONLY in the jab_extend JSON phase) then pulls
  // the fist out with a steep, targeted gradient the warm-start mean drifts
  // toward; the left arm holds guard via Posture. Gated on the keyframe name so
  // no other strategy's reach assignment is touched (auto arm-selection via the
  // outer reach_right covers the other branches).
  else if (residual_keyframe_.name == "jab_extend") {
    reaching_hand = right_hand_pos;
    // Target = the FK-measured location of the right fist at FULL straight
    // extension, expressed relative to torso_position. Measured (jab_fk.py,
    // build Lean_H12) for the jab_extend keyframe: fist is dx=+0.498, dy=-0.085,
    // dz=+0.366 from torso_position (torso sits at z~1.225, the extended fist at
    // z~1.59 — torso_position is well BELOW the shoulder, so a torso_z-0.05
    // "shoulder height" guess like arm_extend_standing uses puts the target
    // ~0.4 m too LOW and the planner punches DOWN with a bent elbow). x pushed a
    // touch past 0.498 to demand the elbow fully straighten rather than fold to
    // hit a nearer point. Torso stays upright (Pelvis Tilt=100) so this pulls
    // ARM extension, not a forward lean.
    // GENTLE target: the natural full-extension fist point (FK dx+0.498), NOT
    // pushed beyond. On the faithful deploy twin a more aggressive forward pull
    // (dx 0.62 + high reach weight) made the planner counterbalance backward and
    // OVERSHOOT into a backward topple at the punch. The jab sits on a marginal
    // stand, so the punch must stay a small perturbation: a modest reach weight
    // (35, set in the jab_extend JSON) toward this natural point gives a visible
    // forward jab without driving the CoM past the balance margin. Torso held
    // upright by Pelvis Tilt=100.
    phase1_target_storage[0] = torso_pos[0] + 0.58;   // just PAST full reach -> forces full straight extension
    phase1_target_storage[1] = torso_pos[1] - 0.09;   // natural extended-arm y
    phase1_target_storage[2] = torso_pos[2] + 0.37;   // extended-fist height
    reach_target = phase1_target_storage;
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
    double const *in_target = data->mocap_pos;
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
                     : (in_target[1] < torso_pos[1]);   // 0/auto
    reaching_hand = reach_right_reach ? right_hand_pos : left_hand_pos;
    // Shoulder anchor = torso_position + FK-measured offset (MJPC frame):
    // (+0.000, +-0.148, +0.219); rest |shoulder->hand| = 0.524.
    double shoulder[3] = {torso_pos[0],
                          torso_pos[1] + (reach_right_reach ? -0.148 : 0.148),
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
      // job (strat 16/33) — the reach/lean/step hierarchy is unchanged.
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
      mju_add3(phase1_target_storage, shoulder, v);
    } else {
      mju_copy3(phase1_target_storage, in_target);  // already in reach
    }
    reach_target = phase1_target_storage;
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
  double ideal_brace[3] = {
      torso_pos[0] + 0.4 * torso_to_table_x,  // Partway between torso and far edge
      // bracing arm = the OTHER arm (reach_right -> left arm braces, so +0.24).
      torso_pos[1] + (reach_right ? 0.24 : -0.24),
      // 2026-05-22: press TARGET 6 cm BELOW the surface (was -0.02). Under the
      // real-robot (doc) ROM the bracing forearm stalled ~7 cm ABOVE the table:
      // with the target only 2 cm under the surface the downward Brace-Pos pull
      // faded before contact, so no contact + no brace force formed and the lean
      // tipped. A deeper press target sustains the pull through to firm forearm
      // contact (the table collision arrests the hand at the surface and converts
      // the residual press into the brace force the Brace-Force cost rewards).
      table_pos[2] - 0.06
  };

  double penalty_hand = hand_dist_penalty * hand_dist;
  double brace_dist = mju_dist3(bracing_hand, ideal_brace);
  double reward_brace = brace_reward * mju_exp(-2.0 * brace_dist);
  double reward_success = (hand_dist < kHandDistThreshold && reach_contact_force > kContactForceThreshold) ? success : 0;
  
  // ========== PALM BRACING INTEGRATION (H12_Hands only) ========== //
  // Check if we have palm sensors (indicates H12_Hands model)
  bool has_palm_sensors = false;
  for (int i = 0; i < model->nsensor; i++) {
    if (std::string(model->names + model->name_sensoradr[i]) == "right_palm_pos") {
      has_palm_sensors = true;
      break;
    }
  }

  double palm_bracing_bonus = 0.0;
  if (has_palm_sensors && any_arm_contact) {
    double const *right_palm_pos = SensorByName(model, data, "right_palm_pos");
    double const *right_palm_normal = SensorByName(model, data, "right_palm_normal");
    double *right_palm_contact = SensorByName(model, data, "right_palm_contact");

    // Palm face-DOWN on the table is the human-like posture: the inside of
    // the palm contacts the table, taking weight through the wrist/forearm.
    // The right_palm_normal sensor is the +z axis of the palm_center site,
    // which points OUT of the palm surface. When the palm is face-down, that
    // outward normal points DOWN — i.e. opposite the table's upward normal.
    // Reward alignment with [0,0,-1] (palm-out vector pointing into the
    // ground) instead of [0,0,1] (which was rewarding back-of-hand-down,
    // the bug the user reported).
    double palm_down_target[3] = {0, 0, -1};
    double palm_alignment = mju_dot3(right_palm_normal, palm_down_target);

    double palm_height_error = mju_abs(right_palm_pos[2] - table_pos[2]);
    double contact_score = (right_palm_contact[0] > 1.0) ? 1.0 : 0.0;
    double flatness_score = mju_max(0.0, palm_alignment);
    double height_score = mju_exp(-10.0 * palm_height_error);

    // Weight bumped 0.5 → 1.5 so the bonus is strong enough to actually
    // drive the right_wrist_roll joint to twist palm-down. Below ~1.0 it
    // was getting drowned out by Posture and Joint Vel. regularisers.
    palm_bracing_bonus = 1.5 * flatness_score * contact_score * height_score;
  }
  // ========== END PALM BRACING INTEGRATION ========== //

  reward = -penalty_hand + reward_brace + reward_success + palm_bracing_bonus;

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
  double height_scale = arm_contact_or_lean ? 0.35 : 1.0;
  residual[counter++] = height_scale * (head_feet_error - height_goal);

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity ----- //
  mju_copy(residual + counter, data->qvel + 6, model->nu);
  if (is_stumble) {
    // Split the "Velocity" joint-velocity damping by joint group (the G1-
    // locomotion pattern: vel/pose-regularised ARMS, gait-driven LEGS). LEG/torso
    // damping (actuators 0..12) RELEASES as the gait amplitude ramps (1-g_amp) so
    // a catch-step is fast and unimpeded; ARM damping (13..26) is BOOSTED + always
    // on so the upper body stays quiet (no flail) even as the gait/balance perturb
    // it -- the user's "keep the arms at home, recover with legs/hips". When calm
    // (g_amp=0) the legs are fully damped (leg_gate=1) so the still-stand is
    // unchanged. is_stumble-gated => strat 0-19, 21 byte-identical.
    double leg_gate = 1.0 - g_amp;
    for (int i = 0; i < model->nu; i++)
      residual[counter + i] *= (i < 13) ? leg_gate : 1.6;
  }
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
  {
    int com_off_id = mj_name2id(model, mjOBJ_NUMERIC, "com_x_offset");
    if (com_off_id >= 0) capture_point[0] += model->numeric_data[model->numeric_adr[com_off_id]];
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
    // so validate Layer-A (crouch/counterbalance) on the twin with this at 0, and
    // dial this on the real robot.
    if (residual_keyframe_.name == "reach_to_target") {
      int rcb_id = mj_name2id(model, mjOBJ_NUMERIC, "reach_com_back");
      if (rcb_id >= 0)
        capture_point[0] += model->numeric_data[model->numeric_adr[rcb_id]];
    }
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
  // Nearest point of the {L_foot, R_foot, R_hand} triangle to the capture
  // point. Used by the two-foot braced phase below and, blended over the phase
  // ramp, by the leg-lift transition so balance doesn't snap from triangle to
  // single-foot segment the instant leg-lift begins (that snap stood the body
  // up out of the brace — the user-reported "reset").
  auto project_triangle = [&](const double L[3], const double Rf[3],
                              const double Rh[3], double out[2]) {
    double vx[3] = {L[0], Rf[0], Rh[0]};
    double vy[3] = {L[1], Rf[1], Rh[1]};
    double ccx = (vx[0] + vx[1] + vx[2]) / 3.0;
    double ccy = (vy[0] + vy[1] + vy[2]) / 3.0;
    for (int i = 0; i < 2; i++)
      for (int j = 0; j < 2 - i; j++) {
        double a1 = std::atan2(vy[j] - ccy, vx[j] - ccx);
        double a2 = std::atan2(vy[j+1] - ccy, vx[j+1] - ccx);
        if (a1 > a2) {
          double tt = vx[j]; vx[j] = vx[j+1]; vx[j+1] = tt;
          tt = vy[j]; vy[j] = vy[j+1]; vy[j+1] = tt;
        }
      }
    double px = capture_point[0], py = capture_point[1];
    bool inside = true;
    for (int i = 0; i < 3; i++) {
      int j = (i + 1) % 3;
      double cross = (vx[j]-vx[i])*(py-vy[i]) - (vy[j]-vy[i])*(px-vx[i]);
      if (cross < 0.0) { inside = false; break; }
    }
    if (inside) { out[0] = px; out[1] = py; return; }
    double best = 1.0e9; out[0] = px; out[1] = py;
    for (int i = 0; i < 3; i++) {
      int j = (i + 1) % 3;
      double ax = vx[i], ay = vy[i], abx = vx[j]-ax, aby = vy[j]-ay;
      double len2 = abx*abx + aby*aby;
      double tt = (len2 > 1e-9)
          ? mju_max(0.0, mju_min(1.0, ((px-ax)*abx + (py-ay)*aby)/len2)) : 0.0;
      double qx = ax + tt*abx, qy = ay + tt*aby;
      double d2 = (px-qx)*(px-qx) + (py-qy)*(py-qy);
      if (d2 < best) { best = d2; out[0] = qx; out[1] = qy; }
    }
  };
  if (is_leg_lift_stage_early) {
    if (any_arm_contact) {
      // L_foot + LOAD-LIMITED step toward R_hand.
      //
      // REGRESSION FIX (2026-05-19): the full {L_foot, R_hand} diagonal used
      // here previously over-promised lateral support during the single-foot
      // lift. A flat forearm on the table carries only ~brace_force_target N
      // (≈14% of body weight at 70 N), yet projecting onto the whole diagonal
      // told the planner the CoM was "supported" anywhere along it — so it
      // drifted the CoM off the planted left foot expecting hand support that
      // physically wasn't there, and the body tipped sideways (user: "tips to
      // the left"). The left foot is the ONLY contact that can arrest a
      // lateral fall; the right-side hand cannot. So we extend the support
      // point toward the hand only by load_frac = brace_force / body_weight
      // (capped 30%). At 70 N this sits ~5 cm off the foot — essentially the
      // iter-40 single-foot pin that held the lift stable, plus a small,
      // load-justified diagonal allowance.
      // 2026-05-20 (option 2): scale by the MEASURED brace force, not just the
      // target, and raise the cap 0.30 → 0.50. FK rollout after option 1 showed
      // the stance hip stopped twisting but the stance KNEE then bent 25° — the
      // balance demand relocated because Balance still believed the hand could
      // only carry 30% toward itself. Crediting the support point by the force
      // the arm is ACTUALLY transferring (≈ /body_weight 500 N) gives MPC the
      // WBC-correct incentive: "push harder on the brace and I'll let the
      // capture point sit toward your hand" — so the stance leg can stay
      // straight AND square instead of squatting/twisting to balance. The
      // per-phase target is kept as a floor so there's support credit before
      // contact force builds (avoids a twist at lift onset). Cap 0.50 keeps the
      // left foot the dominant support, so we don't re-open the over-promise
      // that caused the original sideways tip (cap was 0.30 for that reason).
      double load_frac =
          mju_min(0.50,
                  mju_max(brace_contact_force,
                          residual_keyframe_.brace_force_target) / 500.0);
      double ax = foot_left_pos[0], ay = foot_left_pos[1];
      double bx = ax + load_frac * (bracing_hand[0] - ax);
      double by = ay + load_frac * (bracing_hand[1] - ay);
      double abx = bx - ax, aby = by - ay;
      double len2 = abx*abx + aby*aby;
      double t = (len2 > 1e-9)
          ? mju_max(0.0, mju_min(1.0,
              ((capture_point[0] - ax) * abx +
               (capture_point[1] - ay) * aby) / len2))
          : 0.0;
      double pcp_seg[2] = {ax + t * abx, ay + t * aby};
      // 2026-05-20: blend the two-foot braced triangle -> single-foot segment
      // over the phase ramp (alpha). At leg-lift entry (alpha=0) the support
      // polygon is still the {L_foot,R_foot,R_hand} triangle the body was
      // braced in, so the CoM target does NOT jump back and the body does not
      // stand up; as the right foot actually lifts (alpha->1, in sync with the
      // ramped lift target) it eases to the load-limited single-foot segment.
      // This is what makes leg-lift CONTINUE from the brace (fixes the reset).
      double pcp_tri[2];
      project_triangle(foot_left_pos, foot_right_pos, bracing_hand, pcp_tri);
      pcp[0] = (1.0 - alpha) * pcp_tri[0] + alpha * pcp_seg[0];
      pcp[1] = (1.0 - alpha) * pcp_tri[1] + alpha * pcp_seg[1];
      pcp[2] = 1.0e-3;
    } else {
      mju_copy3(pcp, foot_left_pos);
      pcp[2] = 1.0e-3;
    }
  } else if (any_arm_contact) {
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
    double midfoot_x = 0.5 * (foot_left_pos[0] + foot_right_pos[0]);
    double midfoot_y = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    double hand_load_frac = mju_min(0.9, brace_contact_force / 140.0);
    double hand_vert_x = midfoot_x + hand_load_frac * (bracing_hand[0] - midfoot_x);
    double hand_vert_y = midfoot_y + hand_load_frac * (bracing_hand[1] - midfoot_y);
    double vx[3] = {foot_left_pos[0], foot_right_pos[0], hand_vert_x};
    double vy[3] = {foot_left_pos[1], foot_right_pos[1], hand_vert_y};
    // CCW sort by angle from centroid (3 vertices, bubble sort).
    double ccx = (vx[0] + vx[1] + vx[2]) / 3.0;
    double ccy = (vy[0] + vy[1] + vy[2]) / 3.0;
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 2 - i; j++) {
        double a1 = std::atan2(vy[j]   - ccy, vx[j]   - ccx);
        double a2 = std::atan2(vy[j+1] - ccy, vx[j+1] - ccx);
        if (a1 > a2) {
          double t = vx[j]; vx[j] = vx[j+1]; vx[j+1] = t;
          t = vy[j]; vy[j] = vy[j+1]; vy[j+1] = t;
        }
      }
    }
    double px = capture_point[0], py = capture_point[1];
    bool inside = true;
    for (int i = 0; i < 3; i++) {
      int j = (i + 1) % 3;
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
      for (int i = 0; i < 3; i++) {
        int j = (i + 1) % 3;
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
  // Two-foot lean phases (arm_plant / lean_forward) still load-gate so the
  // bracing arm can take real load without balance fighting it.
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
  double balance_scale =
      is_leg_lift_stage_early
          ? braced_balance_scale + alpha * (1.0 - braced_balance_scale)
          : braced_balance_scale;
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
  // NO table — arm_contact_or_lean is false — so that discount lets the CoM drift forward
  // unchecked into nothing => the persistent ~15° forward lean that topples. With no brace,
  // penalize forward as strictly as backward (symmetric => keep CoM centered, no escape
  // direction). Table-lean pipeline (arm_contact_or_lean == true) is unchanged.
  double fwd_scale = arm_contact_or_lean ? balance_scale : 1.0;
  double dir_scale_x = (cp_dx > 0.0) ? fwd_scale : 1.0;
  double dir_scale_y = 1.0;
  double eff_dx = cp_dx * dir_scale_x;
  double eff_dy = cp_dy * dir_scale_y;
  double balance_excursion = mju_sqrt(eff_dx * eff_dx + eff_dy * eff_dy);
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

  // Vector from torso to reach_target (object for most phases, fixed
  // forward point during arm_extend_standing). Drives Torso Forward Tilt.
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
  // for any tilt direction. During lean_forward (target 0.85 = 32° tilt),
  // backward tilt achieved the same target as forward tilt, and MPC chose
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
  // in lean_forward was forcing 32° pitch regardless of CoM state; with
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
  // Phase-aware lean cap:
  //   lean_with_arm_no_brace → cos(30°) = 0.866 (shallow lean, no brace)
  //   any other arm_contact_or_lean → cos(60°) = 0.5 (deep braced lean)
  // Reason: phase 2 is unbraced — if we let it use the full 60° lean
  // budget, it arrives at phase 3 already at the balance edge with no
  // margin to land the hand. Capping at 30° leaves room for phase 3's
  // brace landing to push deeper.
  double pelvis_tilt_threshold =
      is_lean_no_brace_phase ? 0.866 : 0.5;
  double pelvis_tilt_residual;
  if (is_leg_lift_stage_early) {
    pelvis_tilt_residual = 0.0;
  } else if (arm_contact_or_lean) {
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
      pelvis_tilt_residual = upright_gain * sin_tilt;
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
  double hip_square_scale = is_leg_lift_stage_early
                                ? 4.0
                                : (arm_contact_or_lean ? 2.0 : 1.0);
  // 2026-05-20: FK rollout (monitor/phase_snapshot.py) showed the slant is a
  // leg-lift-phase phenomenon almost entirely on the STANCE (left) leg:
  // L_hip_yaw swings to -21.8° and L_hip_roll to -11.5° in leg_lift_arm_plant,
  // while staying within ±4° through all earlier phases. That hip-yaw twist IS
  // the cross-legged look. The lifting (right) leg legitimately moves, so we
  // square the STANCE hip far harder (×12) during leg-lift but leave the
  // lifting hip at the base scale. User dir: stance leg vertical & square at
  // all times; let MPC find the brace+CoP balance instead of twisting the leg.
  double stance_square_scale =
      is_leg_lift_stage_early ? 12.0 : hip_square_scale;
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
  // Gated to free-standing (arm_contact_or_lean == false) so the table-lean
  // brace tasks 0-5 are byte-identical. Tunable: <numeric name="leg_extension_gain">
  // (1.0 = off / unchanged; sweep up if it still creeps, down if it over-stiffens).
  if (!arm_contact_or_lean) {
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
  // exceeding the target — for arm_plant(target=8N) the planner was penalised
  // ~100× harder for pushing 30N than for pushing 8N, even though more support
  // is exactly what the body needs. Switching to `max(0, desired - actual)`
  // means: pushing harder than the target is FREE, only under-supporting the
  // brace incurs cost. Combined with the bumped per-phase targets in the
  // strategy JSON (arm_plant 25 → lean_forward 70 → deep_reach 120) this
  // tells MPC "transfer this much of body weight through the arm", matching
  // one-sided contact-force tracking — only under-support costs.
  bool is_active_contact =
      (residual_keyframe_.contact_pairs[0].body1 != mjpc::humanoid::kNotSelectedInteract);
  double target_brace_force = residual_keyframe_.brace_force_target >= 0.0
                                   ? residual_keyframe_.brace_force_target
                                   : (is_active_contact ? 70.0 : 0.0);
  // ITER 28 (2026-05-18): smoothstep ramp the brace_force target across phase
  // boundaries — same machinery as phase_reach_scale etc. Without this, going
  // from stand_up (target=0) → arm_plant (target=60) creates a step change
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
  double bgg = mju_min(1.0, brace_reach_gap / 0.15);
  double brace_force_prox_gate = 1.0 - bgg * bgg * (3.0 - 2.0 * bgg);
  residual[counter++] = brace_force_prox_gate *
                        mju_max(0.0, desired_brace_force - brace_contact_force);

  // ------ object distance (reaching hand) ------ //
  // Phase-gated: zero during stand_up so the planner doesn't lunge.
  // Reach gradient is intentionally NOT balance-capped — the planner
  // should keep wanting to reach; Balance's edge_amplifier above is what
  // forces the trade-off (steep edge penalty makes the planner find a
  // counter-balanced posture rather than tipping).
  // Target = `reach_target` (object for most phases, fixed forward point
  // during arm_extend_standing — see top of Residual()).
  mju_sub3(&residual[counter], reaching_hand, reach_target);
  mju_scl3(&residual[counter], &residual[counter],
           phase_reach_scale * leaning);
  counter += 3;

  // ----- reaching hand distance to object ----- //
  mju_sub3(&residual[counter], reaching_hand, reach_target);
  mju_scl3(&residual[counter], &residual[counter], phase_reach_scale);
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
  // DROPPED — both feet stay grounded; the "right foot freed" branch below is
  // dormant. See is_leg_lift_stage_early and the lean.h header.)
  static constexpr double kRightFootHomeXY[2] = {0.2196, -0.163};
  static constexpr double kLeftFootHomeXY[2]  = {0.2196,  0.163};

  bool is_leg_lift_stage = is_leg_lift_stage_early;

  // Left foot is the primary ground anchor during all lean stages.
  // Scale 4x as soon as the arm contacts the table, 5x during leg lift.
  // This is needed because balance residual would otherwise slide the foot
  // to reposition the COM — the arm provides the forward support instead.
  double left_foot_scale = is_leg_lift_stage
                               ? 5.0
                               : (arm_contact_or_lean ? 4.0 : 1.0);

  // 2026-05-20: FK rollout showed the base of support COLLAPSES during
  // forearm_brace — the right foot creeps inward+forward (y -0.16→-0.03,
  // x 0.22→0.38), shrinking the stance from 33cm to 12cm. Anchor the right
  // foot as firmly as the left while braced (×4) so the wide stance holds.
  // 2026-05-26: leg-lift DROPPED → the right foot is NEVER freed; both feet
  // stay grounded (the `is_leg_lift_stage ? 0.0 : ...` below always takes the
  // anchored branch). WBC may still nudge foot placement to hold balance.
  double right_foot_scale = arm_contact_or_lean ? 4.0 : 1.0;
  residual[counter++] = is_leg_lift_stage ? 0.0 : right_foot_scale * (foot_right_pos[0] - kRightFootHomeXY[0]);
  residual[counter++] = is_leg_lift_stage ? 0.0 : right_foot_scale * (foot_right_pos[1] - kRightFootHomeXY[1]);
  residual[counter++] = left_foot_scale * (foot_left_pos[0] - kLeftFootHomeXY[0]);
  residual[counter++] = left_foot_scale * (foot_left_pos[1] - kLeftFootHomeXY[1]);

  // ----- hip clearance from table front face ----- //
  // table body is at world x=0.9, half-size 0.5 → front face at x=0.40.
  // penalise the pelvis entering within 0.08m of that face.
  double *pelvis_pos_3d = SensorByName(model, data, "pelvis_position");
  const double *table_surf = SensorByName(model, data, "table_surface_pos");
  double table_front_x = table_surf[0] - 0.7;
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
  if (is_leg_lift_stage || arm_contact_or_lean) {
    residual[counter++] = mju_max(0.0, 0.42 - left_knee_pos_3d[2])
                        + mju_max(0.0, foot_left_pos[2] - 0.02);
  } else {
    residual[counter++] = 0.0;
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
  if (is_leg_lift_stage) {
    // "Square the stance, THEN lift." (user 2026-05-19: the leg raise only
    // works if the standing leg is straight and the foot faces front; the
    // lifting leg alone goes back.) The lift target is gated by a readiness
    // factor that stays ~0 while the support stance is twisted/bent and ramps
    // to 1 only once it's squared, so MPC spends the early phase straightening
    // the support leg instead of lifting from a twisted base (which tipped it).
    //   - yaw_ready: from foot_left_forward[1] ≈ sin(foot_yaw). 1 when the
    //     standing foot points within ~3° of straight-ahead, → 0 by ~10°.
    //     World-frame, so it catches twist from root yaw, hip yaw, or ankle.
    //   - knee_ready: 1 when the standing knee is straight (knee height ≥0.44),
    //     → 0 when bent (≤0.38). Matches the 0.42 Left Leg Anchor target.
    double *foot_left_fwd = SensorByName(model, data, "foot_left_forward");
    double stance_yaw = mju_abs(foot_left_fwd[1]);
    double yaw_t =
        mju_min(1.0, mju_max(0.0, (stance_yaw - 0.05) / (0.18 - 0.05)));
    double yaw_ready = 1.0 - yaw_t * yaw_t * (3.0 - 2.0 * yaw_t);
    double knee_t = mju_min(
        1.0, mju_max(0.0, (left_knee_pos_3d[2] - 0.38) / (0.44 - 0.38)));
    double knee_ready = knee_t * knee_t * (3.0 - 2.0 * knee_t);
    double readiness = yaw_ready * knee_ready;
    double ramped_lift_target = 0.03 * alpha * readiness;
    residual[counter++] = mju_max(0.0, ramped_lift_target - foot_right_pos[2]);
  } else if (arm_contact_or_lean) {
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
  // Gated on arm_contact_or_lean (phases 2+). Phase 1 is excluded
  // because the home pose already has pelvis 3 cm behind the feet by
  // design — activating this in phase 1 would create constant forward
  // pull that fights the "stay upright" intent.
  if (arm_contact_or_lean) {
    double midfoot_x = 0.5 * (foot_right_pos[0] + foot_left_pos[0]);
    double pelvis_forward_target = midfoot_x + 0.05;
    residual[counter++] =
        mju_max(0.0, pelvis_forward_target - pelvis_pos_3d[0]);
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
  // Active only when the bracing arm is on the table (any_arm_contact);
  // arm_extend_standing has no contact and would point reach_target at the
  // body-relative phase1 storage anyway, which would be degenerate.
  double sp_residual = 0.0;
  if (any_arm_contact) {
    double midfoot_x = is_leg_lift_stage_early
                           ? foot_left_pos[0]
                           : 0.5 * (foot_left_pos[0] + foot_right_pos[0]);
    double midfoot_y = is_leg_lift_stage_early
                           ? foot_left_pos[1]
                           : 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
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
  // GATED to `!any_arm_contact` (2026-05-20): active only in the UNBRACED
  // phases (stand/extend = far from table → 0; lean_with_arm_no_brace = mode 2,
  // the case we care about). It is DISABLED once the arm/elbow are braced
  // (arm_plant / lean_forward / forearm_brace_lean / leg_lift / deep_reach):
  // in those phases the deep forearm-brace pose legitimately brings the body
  // low near the table, and an always-on penalty fought it — the forearm
  // brace wouldn't establish and the leg lifted unsupported. The arm bears
  // the table load there by design, so body-table proximity is not penalised.
  double body_table_force = 0.0;
  if (!any_arm_contact) {
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
  residual[counter++] = body_table_force;

  // ----- Knees straight during leg-lift --------------------------------- //
  // User (2026-05-20): the leg lift is only valuable if the STANDING leg is
  // completely straight (no knee buckle) and only the lifting leg rises a
  // little. The planner was buckling BOTH knees to lower the CoM for
  // stability. The Left Leg Anchor only checks knee HEIGHT, which misses a
  // buckle where the knee juts forward without dropping — so penalise the
  // knee JOINT ANGLE directly (qpos 0 = straight, + = flexed). Strict on the
  // standing (left) knee (×2, ~5° free), gentle on the lifting (right) knee
  // (~14° free, it extends back). Active ONLY in leg-lift so it doesn't fight
  // the two-foot braced squat in the earlier phases.
  double left_knee_angle  = data->qpos[7 + 3];
  double right_knee_angle = data->qpos[7 + 9];
  if (is_leg_lift_stage_early) {
    residual[counter++] = 2.0 * mju_max(0.0, left_knee_angle  - 0.08);
    residual[counter++] =       mju_max(0.0, right_knee_angle - 0.25);
  } else if (is_stumble) {
    // STUMBLE: repurpose this dormant slot into a BENT-KNEE FLOOR -- forbid
    // EITHER knee from diving toward the straight/locked stop (the strut), in
    // EVERY phase (calm OR stepping), so the one-leg passive prop is impossible
    // WHENEVER. One-sided max(0, floor - knee): a knee at the 0.35 march pose,
    // or folded UP for a swing, is FREE (>= floor); only a knee locking toward
    // the floor is penalised. UNLIKE the Symmetry term this is NOT g_amp-gated,
    // so even a sustained step cannot strut; and the swing leg rises ABOVE the
    // floor so the gait is untouched. Strength = the "Knees Straight" JSON
    // weight (0 = off, tunable live). Pairs with the (g_amp-gated) Symmetry term
    // above: Symmetry keeps L/R matched when calm, this floor blocks the lock
    // always. Non-stumble strategies fall through to the else (0,0) -> byte-id.
    constexpr double kKneeFloor = 0.15;  // rad; below this a knee is locking -> strut
    residual[counter++] = mju_max(0.0, kKneeFloor - left_knee_angle);
    residual[counter++] = mju_max(0.0, kKneeFloor - right_knee_angle);
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
  }

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
  // GATED to free-standing (!arm_contact_or_lean): leg-lift / lean / retrieve
  // legitimately stand on one leg, so symmetry is zeroed there. With the XML
  // default weight 0 the term is OFF for every strategy that does not opt in
  // via its JSON weight map ("Symmetry": w), so all other tasks stay
  // byte-identical (zero weight AND/OR zero residual = zero cost).
  if (!arm_contact_or_lean) {
    // STUMBLE (strat 20): RELEASE symmetry WHILE stepping. A catch-step is an
    // INTENTIONAL L/R asymmetry (one foot swings) so the symmetry term must not
    // fight it; when calm (g_amp=0) symmetry is FULL, making stumble's bent-knee
    // stand as strut-proof as the strat-6 stand (which runs Symmetry 200 and
    // never struts). g_amp is 0 for EVERY non-stumble strategy (set only inside
    // the is_stumble stepping block far above), so sym_gate == 1.0 there ->
    // stand/crouch/jab/reach stay byte-identical. This is why the user's strat-20
    // struts and strat-6 doesn't: stumble had Symmetry=0 (it conflicts with
    // stepping); gating restores it for the calm phase the march-gate now holds.
    double sym_gate = 1.0 - g_amp;
    residual[counter++] = sym_gate * (data->qpos[7 + 3] - data->qpos[7 + 9]);  // knee L-R
    residual[counter++] = sym_gate * (data->qpos[7 + 1] - data->qpos[7 + 7]);  // hipPitch L-R
    // anklePitch L-R: the SAGITTAL ankle. joint_forensics on the cold-start
    // backward fall (strat 19) showed the one-leg strut hides HERE — L/R
    // ankle_pitch diverged 13.9deg while knee diverged only 5.2deg. The
    // original term excluded "ankle" reasoning about the LATERAL ankle_roll
    // (asymmetry there is normal); ankle_PITCH is fore/aft and must stay
    // symmetric for a square stance, so penalising its (L-R) closes the
    // strut's last escape. Quadratic => tiny asymmetries stay ~free.
    residual[counter++] = sym_gate * (data->qpos[7 + 4] - data->qpos[7 + 10]); // anklePitch L-R
  } else {
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
    residual[counter++] = 0.0;
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
  if (!arm_contact_or_lean) {
    residual[counter++] = 10.0 * mju_max(0.0, posture_target[2] - data->qpos[2]);
  } else {
    residual[counter++] = 0.0;
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
  if (!arm_contact_or_lean) {
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pelvis_id < 0) pelvis_id = 1;  // floating-base root fallback (always exists)
    const mjtNum *angmom = data->subtree_angmom + 3 * pelvis_id;
    // HIP/ARM RECOVERY tier (Strategy 20): when the capture point excurses, target
    // COUNTER angular momentum (throw torso+arms OPPOSITE the fall) instead of zero
    // -> the flat-footed ankle->HIP->step INTERMEDIATE tier that hauls the CoM back
    // BEFORE a step is needed, WITHOUT rocking onto the toe/heel (the CoP/ankle
    // strategy we deliberately avoid). Zero target when calm == the original
    // regulate-to-zero (unchanged for every other strategy). Fore-aft capture
    // excursion -> pitch momentum (Ly); lateral -> roll (Lx); yaw (Lz) always ->0.
    // Gains are numerics (default 0 = OFF until tuned on the twin; SIGN found there).
    double Lx_tgt = 0.0, Ly_tgt = 0.0;
    if (is_stumble) {
      int rpg_id = mj_name2id(model, mjOBJ_NUMERIC, "recover_pitch_gain");
      int rrg_id = mj_name2id(model, mjOBJ_NUMERIC, "recover_roll_gain");
      double kRecPitch = (rpg_id >= 0)
          ? model->numeric_data[model->numeric_adr[rpg_id]] : 0.0;
      double kRecRoll = (rrg_id >= 0)
          ? model->numeric_data[model->numeric_adr[rrg_id]] : 0.0;
      Ly_tgt = kRecPitch * g_cap_ex;   // fore-aft fall -> pitch counter-momentum
      Lx_tgt = kRecRoll  * g_cap_ey;   // lateral fall  -> roll counter-momentum
    }
    residual[counter++] = 0.1 * (angmom[0] - Lx_tgt);
    residual[counter++] = 0.1 * (angmom[1] - Ly_tgt);
    residual[counter++] = 0.1 * angmom[2];
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
  if (!arm_contact_or_lean) {
    double midfoot_y = 0.5 * (foot_left_pos[1] + foot_right_pos[1]);
    // STUMBLE: rock the CoM toward the STANCE foot as the other foot swings --
    // you cannot raise a LOADED foot, so the weight must shift off it first.
    // (g_bump_r - g_bump_l) is -bump when the LEFT foot swings (shift CoM to the
    // -y RIGHT stance foot) and +bump when the RIGHT swings; 0 in double support
    // (stay centred). This gait-synced weight-shift reference is what converts
    // the folded-swing-leg Posture into a REAL step; the planner finds the hip-
    // roll / ankle to achieve it. Only for stumble; every other strategy keeps
    // the plain centred target (and weight 0 unless it opts in). --- //
    // kLatShift scaled by the trot-window swing scale: a bigger STEP needs a
    // bigger weight-SHIFT to fully unload the swing foot (the measured one-sided
    // step came from too small a rock -- the left foot never unloaded, so it could
    // not lift). At swing_scale 2.5 the rock target is ~0.18 m, enough to take the
    // load off EITHER foot in turn. ==base 0.07 outside the trot window (quiet
    // stand + push-recovery byte-identical). Capped so it can't drive the CoM off
    // the support polygon. --- //
    constexpr double kLatShift = 0.07;   // base CoM lateral rock amplitude [m]
    double lat_amp = mju_min(0.20, kLatShift * trot_swing_scale);
    double lat_tgt = midfoot_y +
        (is_stumble ? lat_amp * g_amp * (g_bump_r - g_bump_l) : 0.0);
    residual[counter++] = 10.0 * (subcom[1] - lat_tgt);
  } else {
    residual[counter++] = 0.0;
  }

  // ============ STUMBLE: Cartesian Gait + Step Place refinement ============ //
  // Strategy 20 "stumble" stepping. The gait clock + swing-leg JOINT reference
  // that BOOTSTRAP the step (via the Posture cost) were computed near the TOP of
  // this function; here we add the two Cartesian refinement terms that ground
  // the motion in task space, reusing g_amp / g_bump_l / g_bump_r. Both are the
  // LAST two <user> sensors (Gait dim 2, Step Place dim 4), XML default weight 0,
  // and write 0 for every non-stumble strategy -> strategies 0-19 stay byte-
  // identical (zero residual AND zero weight => zero cost), exactly the Symmetry/
  // Angular Momentum/Lateral Center opt-in pattern above.
  //
  // WHY STEPPING (not more ankle gain): the documented H1-2 limiter is ankle
  // under-authority -- the ankle saturates once the CoP reaches the support-
  // polygon edge, so a large capture-point excursion is UNRECOVERABLE by ankle
  // torque (the ~56%-marginal stand / one-leg-strut ceiling every other lean
  // strategy hits). Stepping RELOCATES the support polygon under the falling CoM
  // (N-step capturability, Koolen/Pratt), extending the recoverable region far
  // beyond the ankle limit -- the gradient-free win for a weak-ankle robot.
  if (is_stumble) {
    // Cartesian step height TARGET: kStepHeight (0.022) scaled by the trot-window
    // swing scale (default 1 outside the window). At swing_scale 2.5 the target is
    // ~0.055 m -- between the quadruped trot (0.03) and Unitree humanoid walk
    // (0.08) clearance, sized for the bigger H1-2 foot.
    double amp = kStepHeight * g_amp * trot_swing_scale;   // ramped Cartesian step height [m]
    // --- Gait foot-height (dim 2): reinforce the SWING foot rising its bump in
    //     task space; STANCE foot untracked (residual 0). Ground reference = the
    //     lower (planted) foot, so it auto-calibrates with no hard-coded foot z.
    //     trot_gait_wscale boosts this residual in the trot window -> the foot-
    //     clearance cost DOMINATES (the references' #1 lesson), so the planner
    //     actually lifts the foot instead of only rocking. ==1 outside the window
    //     -> quiet stand + push-recovery catch-step byte-identical. --- //
    double ref_z = mju_min(foot_left_pos[2], foot_right_pos[2]);
    // CONTACT-SCHEDULE signal (the references' bilateral-alternation glue, e.g.
    // Unitree's `contact` reward): a foot must be UNLOADED during its swing phase.
    // Foot-height alone gave ONE-SIDED steps -- the rock unloaded only the lighter
    // foot; the other stayed planted (loaded) so its height term just paid a fixed
    // penalty. Penalising the swing foot's CONTACT FORCE forces the planner to
    // shift the CoM far enough to take the load off EITHER foot in turn. Per-foot
    // normal force summed from the live contacts; normalised by ~half body weight
    // (~500 N) to sit at the same ~0.05 scale as the height target. Trot-window
    // only (sched=0 when trot_gait_wscale==1) -> quiet stand + push-recovery
    // byte-identical. Costs ~ncon iterations, stumble + trot only. --- //
    double ff_l = 0.0, ff_r = 0.0;
    if (trot_gait_wscale > 1.01) {  // only inside the trot window
      int fl_body = mj_name2id(model, mjOBJ_BODY, "left_ankle_roll_link");
      int fr_body = mj_name2id(model, mjOBJ_BODY, "right_ankle_roll_link");
      for (int c = 0; c < data->ncon; c++) {
        const mjContact& con = data->contact[c];
        int ba = model->geom_bodyid[con.geom1], bb = model->geom_bodyid[con.geom2];
        mjtNum f6[6];
        mj_contactForce(model, data, c, f6);
        double fn = mju_abs(f6[0]);  // normal force (contact-frame x)
        if (ba == fl_body || bb == fl_body) ff_l += fn;
        if (ba == fr_body || bb == fr_body) ff_r += fn;
      }
    }
    constexpr double kSched = 0.060;  // swing-load penalty scale (~= step-height scale)
    double sched_l = (trot_gait_wscale > 1.01) ? kSched * g_bump_l * mju_min(2.0, ff_l / 500.0) : 0.0;
    double sched_r = (trot_gait_wscale > 1.01) ? kSched * g_bump_r * mju_min(2.0, ff_r / 500.0) : 0.0;
    // SUBTRACT the contact penalty so it REINFORCES the height deficit (a planted
    // swing foot has a NEGATIVE height residual -amp*g_bump; a loaded one makes it
    // MORE negative -> larger cost -> forces the unload). Adding them cancels (the
    // bug that left it one-sided). A correctly-lifted unloaded foot: height~0 and
    // ff~0 -> residual~0 (no penalty).
    residual[counter++] = trot_gait_wscale *
        ((g_bump_l > 0.0) ? (foot_left_pos[2]  - ref_z - amp * g_bump_l) - sched_l : 0.0);
    residual[counter++] = trot_gait_wscale *
        ((g_bump_r > 0.0) ? (foot_right_pos[2] - ref_z - amp * g_bump_r) - sched_r : 0.0);

    // --- Step Place (dim 4): aim each SWING foot's xy at a capture-point /
    //     Raibert target: x_foot = com_xy + nominal_offset
    //                            + (com_vel - v_des)*sqrt(z/g) + v_des*T_stance.
    //     v_des=0, balanced -> nominal stance (march in place); a push -> target
    //     shifts into the push -> recovery STEP; v_des>0 -> feet step in the
    //     commanded direction -> walk (the nav hook). Stance foot untracked. --- //
    int pelvis_id = mj_name2id(model, mjOBJ_BODY, "pelvis");
    if (pelvis_id < 0) pelvis_id = 1;  // floating-base root fallback (always exists)
    const mjtNum *com_pos = data->subtree_com + 3 * pelvis_id;     // whole-body CoM xyz
    double const *com_vel = SensorByName(model, data, "waist_lower_subcomvel");
    double z_com     = mju_max(0.5, com_pos[2]);
    double tau_cap   = mju_sqrt(z_com / 9.81);     // sqrt(z/g): capture-point time const
    double t_stance  = kDutyRatio / kCadenceHz;    // stance duration [s]
    // CLAMP the capture offset to a physical step reach. Unclamped, an incipient
    // topple spikes com_vel -> the foot target explodes metres away -> the swing
    // leg FLINGS to chase it (seen as 0.5 m foot-height spikes that then caused
    // the fall). A real step can only reach ~+/-0.22 m, so clamp there: the term
    // still biases the step toward the capture point for recovery, but can never
    // command an unreachable lunge.
    constexpr double kStepReach = 0.30;   // max single-step xy reach [m] (0.22->0.30
                                          // 2026-06-18: bigger CATCH-STEP for push
                                          // recovery; forward shoves need a longer
                                          // reach to plant the foot ahead of the CoM)
    double off_x = (com_vel[0] - kDesVelX) * tau_cap + kDesVelX * t_stance;
    double off_y = (com_vel[1] - kDesVelY) * tau_cap + kDesVelY * t_stance;
    off_x = mju_max(-kStepReach, mju_min(kStepReach, off_x));
    off_y = mju_max(-kStepReach, mju_min(kStepReach, off_y));
    // nominal foot offsets from the CoM at the standing stance (measured ~0.064 m
    // fore, +/-0.26 m lateral). Biased to 0.09 fore so the feet land slightly
    // AHEAD of the CoM -- countering the documented H1-2 forward CoM lean (~3-4 cm
    // ahead) that topples the static stand. CoM-relative so the stance translates
    // when walking.
    // Fore-aft nominal stance offset: feet planted this far AHEAD of the CoM.
    // +0.13 gave forward-push margin back when forward was the weak axis -- but it
    // also CANCELS the backward catch-step's negative off_x (the rear foot only
    // reaches TO the CoM, never behind it -> can't catch a backward fall). Now
    // that the omnidirectional catch-step makes forward solid, REBALANCE this
    // toward centre so backward gets margin too. Model numeric `stance_off_x`
    // (default 0.13) -> tune live, no rebuild.
    int sox_id = mj_name2id(model, mjOBJ_NUMERIC, "stance_off_x");
    double kStanceOffX = (sox_id >= 0)
        ? model->numeric_data[model->numeric_adr[sox_id]] : 0.13;
    constexpr double kStanceOffY = 0.26;
    // NOTE (2026-06-18): a STAGGERED/braced stance (L foot fore, R foot aft, via a
    // staggered stumble_march keyframe + kStagger here) was TESTED to ~double the
    // fore-aft support polygon (0.26->0.46 m). It improved the static baseline (4/4)
    // but did NOT improve fwd/back PUSH survival and slightly hurt lateral -> REVERTED.
    // Root cause: fore-aft recovery is reaction-BANDWIDTH limited (the planner doesn't
    // exploit the bigger base / use the ankle headroom fast enough at spline 5), not
    // support-geometry limited. The remaining principled lever is the HIP/arm
    // angular-momentum strategy (ankle->HIP->step hierarchy) — unbuilt. The foot is
    // already reference-grade fore-aft (G1 0.17 m / H1 0.20 m) so do NOT lengthen it.
    constexpr double kStagL = 0.0, kStagR = 0.0;   // stagger off (see note)
    residual[counter++] = (g_bump_l > 0.0)
        ? (foot_left_pos[0]  - (com_pos[0] + kStanceOffX + kStagL + off_x)) : 0.0;
    residual[counter++] = (g_bump_l > 0.0)
        ? (foot_left_pos[1]  - (com_pos[1] + kStanceOffY + off_y)) : 0.0;
    residual[counter++] = (g_bump_r > 0.0)
        ? (foot_right_pos[0] - (com_pos[0] + kStanceOffX + kStagR + off_x)) : 0.0;
    residual[counter++] = (g_bump_r > 0.0)
        ? (foot_right_pos[1] - (com_pos[1] - kStanceOffY + off_y)) : 0.0;

    // --- Foot Slip (dim 2): penalize the STANCE foot SLIDING horizontally (gated
    //     on bump==0 = the planted foot). The reference MuJoCo Playground humanoid
    //     relies on this (G1 feet_slip cost): a sampling planner has no learned
    //     recovery reflex, so a stance foot that skids during the weight-shift
    //     builds the exact lateral momentum that topples a weak-ankle stepper --
    //     killing the slip at the source beats reacting to it via the capture
    //     point. (Source: Playground g1/joystick.py _cost_feet_slip.) --- //
    double const *vfl = SensorByName(model, data, "foot_left_velocity");
    double const *vfr = SensorByName(model, data, "foot_right_velocity");
    residual[counter++] =
        (g_bump_l > 0.0) ? 0.0 : mju_sqrt(vfl[0]*vfl[0] + vfl[1]*vfl[1]);
    residual[counter++] =
        (g_bump_r > 0.0) ? 0.0 : mju_sqrt(vfr[0]*vfr[0] + vfr[1]*vfr[1]);
  } else {
    residual[counter++] = 0.0;  // Gait L
    residual[counter++] = 0.0;  // Gait R
    residual[counter++] = 0.0;  // Step Place Lx
    residual[counter++] = 0.0;  // Step Place Ly
    residual[counter++] = 0.0;  // Step Place Rx
    residual[counter++] = 0.0;  // Step Place Ry
    residual[counter++] = 0.0;  // Foot Slip L
    residual[counter++] = 0.0;  // Foot Slip R
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
  // ---- DEBUG: print leg stability diagnostics every ~0.5 s ---- //
  static int debug_tick = 0;
  static const bool lean_dbg = (std::getenv("LEAN_DEBUG") != nullptr);
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
  if (reach_tgt_id >= 0) {
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
                   current_kf.success_sustain_time) {
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
      motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
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
void lean::ApplyRampedWeights(const mjModel *model, const mjData *data) {
  const std::size_t n = weight_names.size();
  if (prev_phase_weights_.size() != n || next_phase_weights_.size() != n) {
    return;
  }
  double dt = mju_max(0.0, data->time - residual_.keyframe_start_time_);
  double alpha_lin = mju_min(dt / ResidualFn::kPhaseRampSeconds, 1.0);
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
  if (reset_reach_tgt_id >= 0) {
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
std::map<std::string, double> lean::PlannerNumericOverrides(int strategy) const {
  const auto names = GetStrategyNames();
  if (strategy < 0 || strategy >= static_cast<int>(names.size())) return {};
  const std::string &name = names[strategy];
  // "Stumble" is the only stepping strategy: its gait clock oscillates the legs,
  // which the stand-tuned sampling_spline_points=3 cannot represent (the swing
  // foot never lifts). spline=5 is the twin-validated sweet spot -- spline=8
  // over-actuates and destabilises even a plain stand (2026-06-18). exploration
  // 0.05 == the XML default, pinned here so stumble is unaffected if that
  // default ever changes. Keyed by NAME so both the Lean_H12 and Lean_H12_Hands
  // stumble slots match. Adding a future strategy that needs a different planner
  // bandwidth is a one-line edit HERE (fork side) -- the deploy node stays
  // strategy-agnostic and never changes.
  if (name == "h12_simple_stumble" || name == "h12_hands_simple_stumble") {
    return {{"sampling_spline_points", 5.0}, {"sampling_exploration", 0.05}};
  }
  return {};
}

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