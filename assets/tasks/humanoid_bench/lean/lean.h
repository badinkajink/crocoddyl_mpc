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
                            mjpc::humanoid::ContactKeyframe())
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    enum LeanMode {
      kModeReach = 0,
      kModeRetrieve,
      kNumMode
    };

   protected:
    mjpc::humanoid::ContactKeyframe residual_keyframe_;

   private:
    friend class lean;

    static constexpr double kHandDistThreshold = 0.0;
    static constexpr double kContactStableTime = 0.0;
    static constexpr double kContactForceThreshold = 0.0;

    void ContactResidual(const mjModel *model, const mjData *data,
                         double *residual, int *counter) const;
  };

  lean() : residual_(this), current_strategy_(-1) {
    target_position_ = {1.2, 0.0, 0.95};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.1, 1.3);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

  virtual std::vector<std::string> GetStrategyNames() const {
    return {"h12_table_lean_reach", "h12_table_lean_reach_extended"};
  }

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.residual_keyframe_);
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
