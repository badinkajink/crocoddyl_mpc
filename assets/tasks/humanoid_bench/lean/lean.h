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
                        mjtNum prev_posture_scale = 1.0)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          keyframe_start_time_(keyframe_start_time),
          prev_phase_reach_scale_(prev_reach_scale),
          prev_phase_brace_pos_scale_(prev_brace_pos_scale),
          prev_phase_posture_scale_(prev_posture_scale) {}

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

   private:
    friend class lean;

    static constexpr double kHandDistThreshold = 0.0;
    static constexpr double kContactStableTime = 0.0;
    static constexpr double kContactForceThreshold = 0.0;

    void ContactResidual(const mjModel *model, const mjData *data,
                         double *residual, int *counter) const;
  };

  lean() : residual_(this), current_strategy_(-1) {
    target_position_ = {1.5, 0.0, 0.73};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.4, 1.6);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.73};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

  virtual std::vector<std::string> GetStrategyNames() const {
    return {"h12_table_lean_reach", "h12_table_lean_reach_extended"};
  }

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
        residual_.prev_phase_posture_scale_);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  mjpc::humanoid::MotionStrategy motion_strategy_;
  int current_strategy_;
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

  std::vector<std::string> GetStrategyNames() const override {
    return {"h12_hands_table_lean_reach", "h12_hands_table_lean_reach_extended"};
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
