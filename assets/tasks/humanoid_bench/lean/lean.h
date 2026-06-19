#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"
#include "mujoco/mujoco.h"

namespace mjpc {

constexpr int kLeanStrategyParameterIndex = 1;

// Manual-phase override. -1 (default) = auto-advance through keyframes
// based on success_sustain_time + target_distance_tolerance. 0..N-1 =
// hold at that keyframe index regardless of progress. Lets the user
// scrub through the loaded strategy's phases without reloading (which
// would reset to keyframe 0 and snap the body back through stand_up).
constexpr int kLeanPhaseParameterIndex = 2;

constexpr char kLeanStrategyFilePath[] =
    SOURCE_DIR "/mjpc/tasks/humanoid_bench/lean/strategies/";

class lean : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const lean *task,
                        const mjpc::humanoid::ContactKeyframe& kf =
                            mjpc::humanoid::ContactKeyframe(),
                        mjtNum keyframe_start_time = 0.0,
                        mjtNum prev_reach_scale = 0.0,
                        mjtNum prev_brace_pos_scale = 0.0,
                        mjtNum prev_posture_scale = 1.0,
                        mjtNum prev_brace_force_target = 0.0,
                        int prev_posture_key_id = 0,
                        int num_phases = 1,
                        const bool* contact_pair_is_new = nullptr)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          prev_phase_reach_scale_(prev_reach_scale),
          prev_phase_brace_pos_scale_(prev_brace_pos_scale),
          prev_phase_posture_scale_(prev_posture_scale),
          prev_phase_brace_force_target_(prev_brace_force_target),
          prev_posture_key_id_(prev_posture_key_id),
          num_phases_(num_phases) {
      for (int i = 0; i < 5; ++i) {
        contact_pair_is_new_[i] =
            contact_pair_is_new ? contact_pair_is_new[i] : false;
      }
    }

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the reach + brace cost scales smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance. 1.5s gives the
    // robot time to absorb the new gradient instead of being shoved forward.
    // Tried bumping to 3.0 to slow the arm swing during 2→3 — backfired:
    // with MPC horizon 1.0s the lean_forward gradient stayed weak for ~2s
    // while Height (head wants to stay high, weight 35 effective in
    // arm_contact_or_lean) was full strength — body settled into a slight
    // backward bend as the cheap local optimum. If arm swing is still too
    // fast at 1.5s, a surgical fix (Brace Hand Velocity residual active
    // only during arm_plant) is preferable to slowing every cost ramp.
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

    // STAND-UP-only target-pose ramp duration (see the asymmetric target ramp
    // in lean.cc). Deliberately LONGER than kPhaseRampSeconds: at the live
    // ~60/s plan rate, straightening the legs from a crouch over only 1.5s
    // still launches the body backward on the second cycle (squat fell ~28s).
    // Spreading the leg extension over 3s lets the sampler keep the capture
    // point under the feet the whole way up. Only used when a phase transition
    // moves the target pose CLOSER to home (standing up); descents still snap.
    static constexpr mjtNum kAscentTargetRampSeconds = 3.0;

    // Crouch-DOWN target-pose ramp duration. Much SHORTER than the ascent ramp:
    // a pure snap folds the legs so fast the upper body overshoots into a
    // forward pitch (the recurring squat descent fall), but the slow ascent
    // ramp on a descent lets the robot catch the target and kills the
    // stabilising spring (forward pitch at ~2.6s). 0.6s threads the needle: the
    // target still leads the robot (spring preserved) yet the fold is spread
    // over ~0.6s instead of instantaneous, capping the overshoot. So short that
    // single-phase pose strategies (which settle for seconds) are unaffected.
    static constexpr mjtNum kDescentTargetRampSeconds = 0.6;

    enum LeanMode {
      kModeReach = 0,
      kModeRetrieve,
      kNumMode
    };

   protected:
    mjpc::humanoid::ContactKeyframe residual_keyframe_;

    // ----- Phase-transition state -----------------------------------------
    // `keyframe_start_time_`: wall time at which the current keyframe became
    // active (set in TransitionLocked). The residual uses `data->time -
    // keyframe_start_time_` to compute how far through the ramp we are.
    // `prev_phase_*_scale_`: the scales that were in effect just before the
    // last transition. Together they let Residual() lerp smoothly into the
    // new phase's scales, which is the WBC-style smooth handoff the robot
    // needs to avoid lurching when a contact cost switches on.
    mjtNum keyframe_start_time_ = 0.0;
    mjtNum prev_phase_reach_scale_ = 0.0;
    mjtNum prev_phase_brace_pos_scale_ = 0.0;
    // Posture scale starts at 1.0 (no boost) and ramps to 3.0 during stand_up.
    mjtNum prev_phase_posture_scale_ = 1.0;
    // ITER 28: previous phase's brace_force_target value, used to smoothstep
    // the brace force demand across phase boundaries so MPC doesn't see a
    // step change (which would plan an impulsive arm slam into the table).
    mjtNum prev_phase_brace_force_target_ = 0.0;

    // Previous phase's posture keyframe id (model <key> index), captured at
    // every transition (SnapshotEffectiveScales) so Residual() can ramp the
    // TARGET pose from it to the current keyframe over kPhaseRampSeconds —
    // parallels prev_phase_posture_scale_ but for the pose itself, not its
    // weight. 0 = home on cold start. Only matters when consecutive phases name
    // DIFFERENT keyframes (cyclic squat); pipeline phases all resolve to home.
    int prev_posture_key_id_ = 0;

    // Number of phases (keyframes) in the active strategy; set in TransitionLocked
    // from motion_strategy_.GetKeyframesCount(). The target-pose ramp in Residual()
    // is GATED on num_phases_ > 1 so single-phase strategies (stand/crouch/arms)
    // never enter the ramp branch -- byte-identical to before. This is the
    // per-strategy gate the 2026-06-08 revert note (lean.cc) said the ramp needed.
    int num_phases_ = 1;

    // Per-contact-pair "is new this phase" flags. true when a contact pair
    // went from inactive (body1=-1) in the previous keyframe to active in
    // the current one — i.e. a brand-new target that just appeared. Used
    // by ContactResidual to multiply each newly-appeared pair's residual
    // by smoothstep(t_in_phase / kPhaseRampSeconds) so the cost grows
    // from 0 to full strength over the same 1.5s window as the weights.
    // Without this, the planner sees the new contact target's gradient
    // instantly and slams the body toward it (the 2→3 hand-slam-into-
    // table failure mode). Pairs that were continuously active across
    // the transition keep factor 1.0 throughout.
    bool contact_pair_is_new_[5] = {false, false, false, false, false};

   private:
    friend class lean;

    static constexpr double kHandDistThreshold = 0.0;
    static constexpr double kContactStableTime = 0.0;
    static constexpr double kContactForceThreshold = 0.0;

    void ContactResidual(const mjModel *model, const mjData *data,
                         double *residual, int *counter) const;
  };

  lean() : residual_(this), current_strategy_(-1) {
    target_position_ = {1.5, 0.0, 0.83};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.4, 1.6);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.83};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

  // Populate phase-aware monitoring metrics (reach, CoP, ICP, brace force,
  // saturation, etc.) for the Research GUI / headless analyzer. Reads the
  // current keyframe + sensor stack; safe to call from the gRPC poll loop.
  // See task.h ComputeMetrics for contract.
  void ComputeMetrics(const mjModel *model, const mjData *data,
                      std::map<std::string, double> *metrics,
                      std::string *phase_name) const override;

  // Slider layout (Lean H12) — user's 6-phase decomposition:
  //   0  stand            — stand_up
  //   1  arm_extend       — stand → arm_extend_standing (arm out, body upright)
  //   2  lean_no_brace    — stand → extend → lean_with_arm_no_brace
  //   3  brace_hand_lean  — stand → extend → lean → arm_plant → lean_forward
  //   4  forearm_brace    — above + forearm_brace_lean (hand+elbow on table)
  //   5  full_pipeline    — identical to slot 4 now: ends in a HELD two-foot
  //                         braced lean (DEFAULT).
  //
  // DESIGN (2026-05-26): the leg-lift phase (leg_lift_arm_plant) is DROPPED
  // permanently. BOTH feet stay stable on the ground through EVERY phase of
  // the pipeline. The only lower-body motion allowed is WBC-driven foot
  // re-placement / hip twist IN SERVICE OF the brace (to hold balance while
  // reaching/leaning) — never lifting a leg off the floor. No strategy JSON
  // contains leg_lift_arm_plant anymore, so slot 5 == slot 4.
  //
  // Each slot is a literal truncation of the index-5 pipeline with the
  // last phase forced indefinite (sustain/time_limit = 9999).
  virtual std::vector<std::string> GetStrategyNames() const {
    return {// 0-5: the lean pipeline (unchanged; default Strategy=5)
            "h12_pipeline_stand",
            "h12_pipeline_arm_extend",
            "h12_pipeline_lean_no_brace",
            "h12_pipeline_brace_hand_lean",
            "h12_pipeline_forearm_brace",
            "h12_pipeline_full_pipeline",
            // 6+: standalone single-skill tasks for incremental sim2real
            // bring-up. Each is one indefinite phase. Pose tasks (crouch,
            // arms_*, lean_*, torso_twist) name a model <key> keyframe that
            // the Posture/Control costs track (see lean.cc pose-library
            // block). Slide the Strategy parameter to select; raise the
            // residual_Strategy slider max in Lean_H12.xml accordingly.
            "h12_simple_stand",          // 6  balance hold at home pose
            "h12_simple_reach_forward",  // 7  left arm forward, body squared
            "h12_simple_crouch",         // 8  symmetric squat, torso upright
            "h12_simple_arms_sideways",  // 9  bilateral lateral raise (T)
            "h12_simple_arms_forward",   // 10 bilateral front raise
            "h12_simple_arms_overhead",  // 11 both arms overhead
            "h12_simple_single_arm_raise",  // 12 right arm to the side
            "h12_simple_lean_left",      // 13 lateral weight-shift onto L
            "h12_simple_lean_right",     // 14 lateral weight-shift onto R
            "h12_simple_torso_twist",    // 15 waist yaw to the left
            "h12_simple_counterbalance", // 16 left-arm forward reach + emergent
                                         //    right-arm/torso counterweight
            "h12_simple_squat",          // 17 cyclic squat: crouch(5s)->stand(5s)->loop
                                         //    2-phase strategy; phase machine wraps
                                         //    via NextKeyframe() modulo (motion_strategy.cc)
            "h12_simple_squatter",       // 18 native squatter: 2-phase stand_up<->crouch
                                         //    auto-cycle reusing strat 6 (stand) + strat 8
                                         //    (crouch) weight blocks VERBATIM. The MJPC-side
                                         //    twin of the node's --strategy 18 sequencer
                                         //    (h12_control_node.cc), so the planner/monitor/
                                         //    analyzer can drive the squat natively.
            "h12_simple_jab",            // 19 standing boxing jab: 3-phase
                                         //    stand_up -> jab_guard -> jab_extend auto-cycle. A
                                         //    stand_up LEAD-IN (gentle 3s ramp from home, like the
                                         //    stand/squatter strategies) settles the legs/balance
                                         //    FIRST, THEN the arms raise to guard, THEN the RIGHT
                                         //    arm punches straight forward (shoulder_pitch -1.0,
                                         //    elbow straightens) and retracts. The lead-in is what
                                         //    gives the real robot a proper home->stand->guard
                                         //    transition (raising arms straight from home spun it).
                                         //    WBC base = strat 6 (stand) weights, Posture 60 for
                                         //    punch authority; asymmetric arms are free (Symmetry
                                         //    penalizes only legs). Slow, deliberate cadence.
            "h12_simple_stumble"};       // 20 stumble: gait-clock STEPPING. A
                                         //    stand_up LEAD-IN settles balance, THEN a continuous
                                         //    gait clock alternates the feet (antiphase, ~1.6 Hz,
                                         //    duty 0.65, 6 cm step) so the robot STEPS IN PLACE to
                                         //    stay up ("stumble") and WALKS when a desired CoM
                                         //    velocity is commanded (nav-package hook in lean.cc).
                                         //    Unlike every other lean strategy (both feet planted,
                                         //    ankle-only balance) it RELOCATES the support polygon
                                         //    under the CoM -> escapes the weak-ankle ceiling. New
                                         //    Gait + Step Place cost terms, name-gated; Symmetry/
                                         //    Lateral Center/Knees OFF (stepping breaks the static-
                                         //    stance symmetry on purpose).
  }

  // Live per-phase weight blending --------------------------------------- //
  // Per-phase keyframes in the strategy JSON carry a `weight: { name: val }`
  // map. On phase advance we snapshot the live cost weights, compute the new
  // phase's targets (JSON override OR XML default for missing keys), and ramp
  // weight[] from snapshot → target over kPhaseRampSeconds using the same
  // smoothstep curve the residual uses for reach/brace/posture scales.
  // This lets the user isolate behaviours from the strategy file alone:
  // setting "Brace Pos": 0 in a phase silences brace cost without recompiling.
  // Missing keys preserve XML defaults so existing strategies (empty `{}`)
  // keep their old behaviour.
  void ApplyRampedWeights(const mjModel *model, const mjData *data);

 private:
  void SnapshotXmlDefaultWeights(const mjModel *model);
  void PrepareNextPhaseWeights(const mjpc::humanoid::ContactKeyframe &kf);
  void SnapshotCurrentWeightsAsPrev();

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    // Copy the phase-transition timing state along with the keyframe so
    // freshly-spawned residuals (one per rollout thread) see the same ramp
    // progress as the canonical residual_.
    return std::make_unique<ResidualFn>(
        this, residual_.residual_keyframe_,
        residual_.keyframe_start_time_,
        residual_.prev_phase_reach_scale_,
        residual_.prev_phase_brace_pos_scale_,
        residual_.prev_phase_posture_scale_,
        residual_.prev_phase_brace_force_target_,
        residual_.prev_posture_key_id_,
        residual_.num_phases_,
        residual_.contact_pair_is_new_);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;

  // Weight-ramp state (parallel to ResidualFn::prev_phase_*_scale_):
  //   xml_default_weights_  -- per-residual default from sensor user data,
  //                            snapshot once in ResetLocked. Used as the
  //                            fallback when a phase's JSON weight map
  //                            doesn't include a particular residual name.
  //   prev_phase_weights_   -- weight[] snapshot at the start of the current
  //                            ramp. Captured mid-ramp so successive phase
  //                            advances blend smoothly through whatever the
  //                            rollouts were actually seeing.
  //   next_phase_weights_   -- target weight[] for the current phase.
  std::vector<double> xml_default_weights_;
  std::vector<double> prev_phase_weights_;
  std::vector<double> next_phase_weights_;
};

class Lean_H12 : public lean {
 public:
  std::string Name() const override { return "Lean H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12.xml");
  }
};

// Identical to Lean_H12 (same strategies/costs/weights, nu=27, nq=41) but the
// model carries the two fixed magpie grippers as ~0.506 kg mass-only bodies at
// the wrists. Load with --task "Lean H12 Magpie" when the grippers are mounted;
// fall back to "Lean H12" when they're removed. The grippers are NOT actuated
// here -- open/close is owned by the separate magpie_msgs controller.
class Lean_H12_Magpie : public lean {
 public:
  std::string Name() const override { return "Lean H12 Magpie"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Magpie.xml");
  }
};

class Lean_H12_Hands : public lean {
 public:
  std::string Name() const override { return "Lean H12 Hands"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Hands.xml");
  }

  // Mirrors Lean_H12::GetStrategyNames slot-for-slot. NOTE: the Hands model's
  // Posture cost (dim 27) does not reach the right arm (qpos[39:46] — the
  // 12-DOF dexterous hands shift it), so right-arm pose tasks are partial on
  // this variant until the Hands Posture dim is widened. Lower-body + left-arm
  // poses (stand/crouch/lean_*/arms_forward-left) track correctly.
  std::vector<std::string> GetStrategyNames() const override {
    return {"h12_hands_pipeline_stand",
            "h12_hands_pipeline_arm_extend",
            "h12_hands_pipeline_lean_no_brace",
            "h12_hands_pipeline_brace_hand_lean",
            "h12_hands_pipeline_forearm_brace",
            "h12_hands_pipeline_full_pipeline",
            "h12_hands_simple_stand",
            "h12_hands_simple_reach_forward",
            "h12_hands_simple_crouch",
            "h12_hands_simple_arms_sideways",
            "h12_hands_simple_arms_forward",
            "h12_hands_simple_arms_overhead",
            "h12_hands_simple_single_arm_raise",
            "h12_hands_simple_lean_left",
            "h12_hands_simple_lean_right",
            "h12_hands_simple_torso_twist",
            "h12_hands_simple_counterbalance",  // 16 (mirrors Lean_H12 slot 16)
            "h12_hands_simple_squat",   // 17 cyclic squat (mirrors Lean_H12 slot 17)
            "h12_hands_simple_squatter",  // 18 native squatter (mirrors Lean_H12 slot 18)
            "h12_hands_simple_jab",     // 19 standing boxing jab (mirrors Lean_H12 slot 19).
                                        //    Right-arm punch tracks only partially on the
                                        //    Hands model (Posture dim 27 misses the right arm).
            "h12_hands_simple_stumble"};  // 20 gait-clock stepping (mirrors Lean_H12 slot 20).
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
