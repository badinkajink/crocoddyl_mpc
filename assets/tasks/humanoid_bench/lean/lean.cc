#include "mjpc/tasks/humanoid_bench/lean/lean.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "mujoco/mujoco.h"

namespace mjpc {
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

  //------------- Reward for the lean task --------------//
  double const hand_dist_penalty = 1.0;
  double const brace_reward = 0.5;
  double const success = 1000;

  // ----- object position ----- //
  double const *object_pos = SensorByName(model, data, "object_pos");
  
  // ----- right hand position (reaching hand) ----- //
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  double hand_dist = mju_dist3(right_hand_pos, object_pos);
  double penalty_hand = hand_dist_penalty * hand_dist;

  // ----- left hand brace (on table) ----- //
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *table_pos = SensorByName(model, data, "table_surface_pos");
  
  // Ideal brace position: on table, forward of robot
  double ideal_brace[3] = {table_pos[0] - 0.2, left_hand_pos[1], table_pos[2] + 0.05};
  double brace_dist = mju_dist3(left_hand_pos, ideal_brace);
  double reward_brace = brace_reward * mju_exp(-2.0 * brace_dist);

  // Success when hand reaches object
  double reward_success = (hand_dist < 0.05) ? success : 0;

  // ----- reward ----- //
  double reward = -penalty_hand + reward_brace + reward_success;

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
  residual[counter++] = head_feet_error - height_goal;

  // ----- Balance: CoM-feet xy error ----- //

  // capture point
  double *com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], com_velocity, 2);
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
  mju_addScl(capture_point, subcom, subcomvel, 0.3, 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right_pos, foot_left_pos);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right_pos, foot_left_pos);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is leaning - modified to be less strict than standing
  double leaning =
      torso_height / mju_sqrt(torso_height * torso_height + 0.65 * 0.65) - 0.2;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], leaning, 2);

  counter += 2;

  // ----- torso forward tilt (NEW) ----- //
  // Encourage forward lean to reach object
  // ----- torso forward tilt (direction-based) ----- //
  double *torso_forward = SensorByName(model, data, "torso_forward");
  double *torso_pos = SensorByName(model, data, "torso_position");

  // Vector from torso to object (desired lean direction)
  double reach_dir[3];
  mju_sub3(reach_dir, object_pos, torso_pos);
  // double reach_dist = mju_normalize3(reach_dir);

  // Want torso forward axis to align with reach direction
  // dot product should be close to 1
  double alignment = mju_dot3(torso_forward, reach_dir);
  residual[counter++] = 1.0 - alignment;

  // ----- pelvis tilt (NEW) ----- //
  // Allow pelvis to tilt slightly forward for stability during lean
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  double target_pelvis_tilt = 0.85;  // Slight forward tilt allowed
  residual[counter++] = pelvis_up[2] - target_pelvis_tilt;

  // ----- posture ----- //
  // Reduced weight vs push task to allow more deviation for leaning
  mju_sub(&residual[counter], data->qpos + 7, model->key_qpos + 7, model->nu);
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
  mju_sub(&residual[counter], data->ctrl, model->key_qpos + 7,
          model->nu);  // because of pos control
  counter += model->nu;

  // ------ object distance (right hand) ------ //
  mju_sub3(&residual[counter], right_hand_pos, object_pos);
  mju_scl3(&residual[counter], &residual[counter], leaning);
  counter += 3;

  // ----- right hand distance to object ----- //
  mju_sub3(&residual[counter], right_hand_pos, object_pos);
  counter += 3;

  // ----- left hand brace on table (NEW) ----- //
  // Encourage left hand to brace on table surface for stability
  mju_sub3(&residual[counter], left_hand_pos, ideal_brace);
  counter += 3;

  std::cout << task_->target_position_[0] << std::flush;

  // sensor dim sanity check
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error(
        "mismatch between total user-sensor dimension %d "
        "and actual length of residual %d",
        user_sensor_dim, counter);
  }
}

// -------- Transition for humanoid_bench lean task -------- //
// ------------------------------------------------------------ //
void lean::TransitionLocked(mjModel *model, mjData *data) {
  double const *object_pos = SensorByName(model, data, "object_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  double hand_dist = mju_dist3(right_hand_pos, object_pos);
  
  if (hand_dist < 0.05) {  // consider task as solved
    // set random target position (farther away to require leaning)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.1, 1.3);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.95};
    printf("New target position: %f, %f, %f\n", target_position_[0],
           target_position_[1], target_position_[2]);
  }
  mju_copy3(data->mocap_pos, target_position_.data());
}

void lean::ResetLocked(const mjModel *model) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(1.1, 1.3);
  std::uniform_real_distribution<> dis_y(-0.3, 0.3);
  target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  printf("New target position: %f, %f, %f\n", target_position_[0],
         target_position_[1], target_position_[2]);
}
}  // namespace mjpc