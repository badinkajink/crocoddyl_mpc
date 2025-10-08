#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_

#include <memory>
#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class lean : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const lean *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<lean *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Add mode enum
    enum LeanMode {
      kModeReach = 0,
      kModeRetrieve,
      kNumMode
    };

    private:
      lean *task_;
      friend class lean;

      // Add mode state variable
      LeanMode current_mode_ = kModeReach;
      double mode_start_time_ = 0;
      double last_transition_time_ = -1;

      // Thresholds for mode transition
      static constexpr double kHandDistThreshold = 0.08;  // meters
      static constexpr double kContactStableTime = 0.0;  // seconds to wait before retrieve
      static constexpr double kContactForceThreshold = 0.0;  // N
      double contact_start_time_ = -1;
  };

  lean() : residual_(this) {
    target_position_ = {1.2, 0.0, 0.95};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.1, 1.3);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const

      override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual()

      override {
    return &residual_;
  }

private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
  int object_left_weld_id_ = -1;
  int object_right_weld_id_ = -1;
};

class Lean_H12 : public lean {
 public:
  std::string Name() const override { return "Lean H12"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_H12.xml");
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_H_