#include "mjpc/tasks/humanoid_bench/lean/lean.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

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

  // ----- object position ----- //
  double const *object_pos = SensorByName(model, data, "object_pos");

  // ----- Determine which hand reaches and which braces ----- //
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");

  double left_obj_dist = mju_dist3(left_hand_pos, object_pos);
  double right_obj_dist = mju_dist3(right_hand_pos, object_pos);

  // Closer hand reaches, farther hand braces
  bool left_reaches = left_obj_dist < right_obj_dist;
  double const *reaching_hand = left_reaches ? left_hand_pos : right_hand_pos;
  double const *bracing_hand = left_reaches ? right_hand_pos : left_hand_pos;

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
  double brace_contact_force = left_reaches ? right_contact[0] : left_contact[0];
  double reach_contact_force = left_reaches ? left_contact[0] : right_contact[0];

  double reward = 0;

  // Bracing position calculation; Position brace closer to robot and slightly lower to encourage weight transfer
  double const *table_pos = SensorByName(model, data, "table_surface_pos");
  double *torso_pos = SensorByName(model, data, "torso_position");
  double torso_to_table_x = table_pos[0] - torso_pos[0];
  double ideal_brace[3] = {
      torso_pos[0] + 0.4 * torso_to_table_x,  // Partway between torso and far edge
      bracing_hand[1],  // Keep y close to current
      table_pos[2] - 0.02  // Lower - encourage pressing into table
  };

  double penalty_hand = hand_dist_penalty * hand_dist;
  double brace_dist = mju_dist3(bracing_hand, ideal_brace);
  double reward_brace = brace_reward * mju_exp(-2.0 * brace_dist);
  double reward_success = (hand_dist < kHandDistThreshold && reach_contact_force > kContactForceThreshold) ? success : 0;
  
  // // ========== PALM BRACING INTEGRATION (H12_Hands only) ========== //
  // // Check if we have palm sensors (indicates H12_Hands model)
  // bool has_palm_sensors = false;
  // for (int i = 0; i < model->nsensor; i++) {
  //   if (std::string(model->names + model->name_sensoradr[i]) == "left_palm_pos") {
  //     has_palm_sensors = true;
  //     break;
  //   }
  // }
  
  // double palm_bracing_bonus = 0.0;
  // if (has_palm_sensors) {
  //   // Get palm position and normal for bracing hand
  //   double const *left_palm_pos = SensorByName(model, data, "left_palm_pos");
  //   double const *right_palm_pos = SensorByName(model, data, "right_palm_pos");
  //   double const *bracing_palm = left_reaches ? right_palm_pos : left_palm_pos;
    
  //   double const *left_palm_normal = SensorByName(model, data, "left_palm_normal");
  //   double const *right_palm_normal = SensorByName(model, data, "right_palm_normal");
  //   double const *bracing_palm_normal = left_reaches ? right_palm_normal : left_palm_normal;
    
  //   double *left_palm_contact = SensorByName(model, data, "left_palm_contact");
  //   double *right_palm_contact = SensorByName(model, data, "right_palm_contact");
  //   double palm_contact_force = left_reaches ? right_palm_contact[0] : left_palm_contact[0];
    
  //   // Palm orientation: want palm flat on table (normal pointing up [0,0,1])
  //   double table_normal[3] = {0, 0, 1};
  //   double palm_alignment = mju_dot3(bracing_palm_normal, table_normal);
    
  //   // Palm height: should be at table surface
  //   double palm_height_error = mju_abs(bracing_palm[2] - table_pos[2]);
    
  //   // Bonus for good palm bracing (flat, in contact, at right height)
  //   double flatness_score = mju_max(0.0, palm_alignment);  // 0 to 1
  //   double contact_score = (palm_contact_force > 1.0) ? 1.0 : 0.0;
  //   double height_score = mju_exp(-10.0 * palm_height_error);
    
  //   palm_bracing_bonus = 0.3 * flatness_score * contact_score * height_score;
  // }
  // // ========== END PALM BRACING INTEGRATION ========== //
  
  // reward = -penalty_hand + reward_brace + reward_success + palm_bracing_bonus;
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

  // ----- torso forward tilt (direction-based) ----- //
  // Encourage forward lean to reach object
  double *torso_forward = SensorByName(model, data, "torso_forward");

  // Vector from torso to object (desired lean direction)
  double reach_dir[3];
  mju_sub3(reach_dir, object_pos, torso_pos);
  mju_normalize3(reach_dir);

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

  // ----- bracing hand position on table ----- //
  mju_sub3(&residual[counter], bracing_hand, ideal_brace);
  counter += 3;

  // Only want brace force when actively bracing (not during stand_up where all contacts are inactive)
  bool is_active_contact =
      (residual_keyframe_.contact_pairs[0].body1 != mjpc::humanoid::kNotSelectedInteract);
  double desired_brace_force = is_active_contact ? 15.0 : 0.0;
  residual[counter++] = desired_brace_force - brace_contact_force;

  // ------ object distance (reaching hand) ------ //
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  mju_scl3(&residual[counter], &residual[counter], leaning);
  counter += 3;

  // ----- reaching hand distance to object ----- //
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  counter += 3;

  // ----- foot stability: penalize horizontal foot velocity ----- //
  // Prevents the planner from using leg steps to reduce arm costs
  double *foot_right_vel_s = SensorByName(model, data, "foot_right_vel");
  double *foot_left_vel_s = SensorByName(model, data, "foot_left_vel");
  residual[counter++] = foot_right_vel_s[0];
  residual[counter++] = foot_right_vel_s[1];
  residual[counter++] = foot_left_vel_s[0];
  residual[counter++] = foot_left_vel_s[1];

  // ----- contact keyframe residual ----- //
  ContactResidual(model, data, residual, &counter);

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
  if (user_sensor_dim != counter) {
    mju_error(
        "mismatch between total user-sensor dimension %d "
        "and actual length of residual %d",
        user_sensor_dim, counter);
  }
}

void lean::ResidualFn::ContactResidual(const mjModel *model, const mjData *data,
                                       double *residual, int *counter) const {
  using mjpc::humanoid::kNotSelectedInteract;
  using mjpc::humanoid::kNumberOfContactPairsInteract;
  for (int i = 0; i < kNumberOfContactPairsInteract; i++) {
    const mjpc::humanoid::ContactPair& contact = residual_keyframe_.contact_pairs[i];
    if (contact.body1 != kNotSelectedInteract &&
        contact.body2 != kNotSelectedInteract &&
        contact.body1 < model->nbody &&
        contact.body2 < model->nbody) {
      double dist[3] = {0.};
      contact.GetDistance(dist, data);
      for (int j = 0; j < 3; j++) residual[(*counter)++] = mju_abs(dist[j]);
    } else {
      for (int j = 0; j < 3; j++) residual[(*counter)++] = 0;
    }
  }
}

// -------- Transition for humanoid_bench lean task -------- //
// ------------------------------------------------------------ //
void lean::TransitionLocked(mjModel *model, mjData *data) {
  // keep mocap target updated for the existing hand-reach residuals
  double const *object_pos = SensorByName(model, data, "object_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  double hand_dist = mju_dist3(right_hand_pos, object_pos);
  if (hand_dist < 0.05) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.1, 1.3);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.95};
    printf("New target position: %f, %f, %f\n", target_position_[0],
           target_position_[1], target_position_[2]);
  }
  mju_copy3(data->mocap_pos, target_position_.data());

  // strategy-based contact keyframe progression
  const auto kStrategyNames = GetStrategyNames();
  int requested_strategy =
      (int)std::round(parameters[kLeanStrategyParameterIndex]);
  requested_strategy = std::max(
      0, std::min(requested_strategy, (int)kStrategyNames.size() - 1));

  if (!motion_strategy_.HasKeyframes() ||
      requested_strategy != current_strategy_) {
    current_strategy_ = requested_strategy;
    motion_strategy_.ClearKeyframes();
    motion_strategy_.LoadStrategy(kStrategyNames[current_strategy_],
                                  kLeanStrategyFilePath);
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    return;
  }

  const mjpc::humanoid::ContactKeyframe& current_kf =
      motion_strategy_.GetCurrentKeyframe();
  const double total_distance = motion_strategy_.CalculateTotalKeyframeDistance(
      data, mjpc::humanoid::ContactKeyframeErrorType::kNorm);

  if (data->time - motion_strategy_.GetCurrentKeyframeStartTime() >
          current_kf.time_limit &&
      total_distance > current_kf.target_distance_tolerance) {
    motion_strategy_.Reset();
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
  } else if (total_distance <= current_kf.target_distance_tolerance &&
             data->time -
                     motion_strategy_.GetCurrentKeyframeSuccessStartTime() >
                 current_kf.success_sustain_time) {
    motion_strategy_.NextKeyframe();
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
  } else if (total_distance > current_kf.target_distance_tolerance) {
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
  }
}

void lean::ResetLocked(const mjModel *model) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(1.1, 1.3);
  std::uniform_real_distribution<> dis_y(-0.3, 0.3);
  target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  printf("New target position: %f, %f, %f\n", target_position_[0],
         target_position_[1], target_position_[2]);

  // DEBUG: Print joint order
  printf("\nJoint order for qpos:\n");
  for (int i = 0; i < model->njnt; i++) {
    const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
    int qpos_adr = model->jnt_qposadr[i];
    printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  }
}
}  // namespace mjpc