#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_

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
                        mjtNum prev_brace_force_target = 0.0)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          prev_phase_reach_scale_(prev_reach_scale),
          prev_phase_brace_pos_scale_(prev_brace_pos_scale),
          prev_phase_posture_scale_(prev_posture_scale),
          prev_phase_brace_force_target_(prev_brace_force_target) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Phase-transition ramp duration: the reach + brace cost scales smoothly
    // interpolate from their previous-phase values to the new-phase values
    // over this many seconds after each keyframe advance. 1.5s gives the
    // robot time to absorb the new gradient instead of being shoved forward.
    static constexpr mjtNum kPhaseRampSeconds = 1.5;

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

  // Slider layout (Lean H12) — user's 6-phase decomposition:
  //   0  stand            — stand_up
  //   1  arm_extend       — stand → arm_extend_standing (arm out, body upright)
  //   2  lean_no_brace    — stand → extend → lean_with_arm_no_brace
  //   3  brace_hand_lean  — stand → extend → lean → arm_plant → lean_forward
  //   4  forearm_brace    — above + forearm_brace_lean (hand+elbow on table)
  //   5  full_pipeline    — above + leg_lift_arm_plant (DEFAULT)
  //
  // Each slot is a literal truncation of the index-5 pipeline with the
  // last phase forced indefinite (sustain/time_limit = 9999).
  virtual std::vector<std::string> GetStrategyNames() const {
    return {"h12_pipeline_stand",
            "h12_pipeline_arm_extend",
            "h12_pipeline_lean_no_brace",
            "h12_pipeline_brace_hand_lean",
            "h12_pipeline_forearm_brace",
            "h12_pipeline_full_pipeline"};
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
        residual_.prev_phase_brace_force_target_);
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

class Lean_H12_Hands : public lean {
 public:
  std::string Name() const override { return "Lean H12 Hands"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12_Hands.xml");
  }

  // Mirrors Lean_H12::GetStrategyNames slot-for-slot.
  std::vector<std::string> GetStrategyNames() const override {
    return {"h12_hands_pipeline_stand",
            "h12_hands_pipeline_arm_extend",
            "h12_hands_pipeline_lean_no_brace",
            "h12_hands_pipeline_brace_hand_lean",
            "h12_hands_pipeline_forearm_brace",
            "h12_hands_pipeline_full_pipeline"};
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
