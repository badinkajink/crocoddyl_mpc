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
  double const retrieve_reward = 1000;

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

  if (current_mode_ == kModeReach) {
    double penalty_hand = hand_dist_penalty * hand_dist;
    double brace_dist = mju_dist3(bracing_hand, ideal_brace);
    double reward_brace = brace_reward * mju_exp(-2.0 * brace_dist);  // Success when hand reaches object
    double reward_success = (hand_dist < kHandDistThreshold && reach_contact_force > kContactForceThreshold) ? success : 0;
    reward = -penalty_hand + reward_brace + reward_success;
    
  } else if (current_mode_ == kModeRetrieve) {
    double *torso_pos = SensorByName(model, data, "torso_position");
    // Reward bringing object close to torso
    double obj_to_torso_dist = mju_dist3(object_pos, torso_pos);
    double reward_retrieve_dist = retrieve_reward * mju_exp(-2.0 * obj_to_torso_dist);
    // Reward standing upright
    double *torso_up = SensorByName(model, data, "torso_up");
    double reward_upright = 100.0 * (torso_up[2] - 0.9);  // Want torso_up[2] close to 1
    
    reward = reward_retrieve_dist + reward_upright;
  }

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

  // ----- bracing hand position on table ----- //
  mju_sub3(&residual[counter], bracing_hand, ideal_brace);
  counter += 3;

  // Want significant downward force (tune desired force via weight in XML)
  double desired_brace_force = 15.0;  // N
  residual[counter++] = desired_brace_force - brace_contact_force;

  // ------ object distance (reaching hand) ------ //
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  mju_scl3(&residual[counter], &residual[counter], leaning);
  counter += 3;

  // ----- reaching hand distance to object ----- //
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  counter += 3;

  task_->target_position_[0] = 0.0;  // DEBUG

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
  // Handle data reset
  if (data->time < residual_.last_transition_time_ || 
      residual_.last_transition_time_ == -1) {
    residual_.last_transition_time_ = data->time;
    residual_.current_mode_ = ResidualFn::kModeReach;
    residual_.contact_start_time_ = -1;
  }

  // Get hand and object info
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");
  int left_left_finger_act = mj_name2id(model, mjOBJ_ACTUATOR, "left_left_finger");
  int left_right_finger_act = mj_name2id(model, mjOBJ_ACTUATOR, "left_right_finger");
  int right_left_finger_act = mj_name2id(model, mjOBJ_ACTUATOR, "right_left_finger");
  int right_right_finger_act = mj_name2id(model, mjOBJ_ACTUATOR, "right_right_finger");
  double const *object_pos = SensorByName(model, data, "object_pos");

  double left_obj_dist = mju_dist3(left_hand_pos, object_pos);
  double right_obj_dist = mju_dist3(right_hand_pos, object_pos);
  bool left_reaches = left_obj_dist < right_obj_dist;

  // Get contact force on reaching hand
  double *left_contact = SensorByName(model, data, "left_hand_contact");
  double *right_contact = SensorByName(model, data, "right_hand_contact");
  double reach_contact_force = left_reaches ? left_contact[0] : right_contact[0];
  // double brace_contact_force = left_reaches ? right_contact[0] : left_contact[0];
  
  double hand_dist = left_reaches ? left_obj_dist : right_obj_dist;
  // Determine desired gripper positions based on mode
  double left_gripper_target = (residual_.current_mode_ == ResidualFn::kModeRetrieve && 
                                left_reaches) ? 0.0 : 0.035;
  double right_gripper_target = (residual_.current_mode_ == ResidualFn::kModeRetrieve && 
                                 !left_reaches) ? 0.0 : 0.035;
  
  // Lock gripper controls by setting both ctrl limits to desired value
  if (left_left_finger_act_ >= 0) {
    model->actuator_ctrlrange[2*left_left_finger_act_] = left_gripper_target;
    model->actuator_ctrlrange[2*left_left_finger_act_+1] = left_gripper_target;
    data->ctrl[left_left_finger_act_] = left_gripper_target;
  }
  if (left_right_finger_act_ >= 0) {
    model->actuator_ctrlrange[2*left_right_finger_act_] = left_gripper_target;
    model->actuator_ctrlrange[2*left_right_finger_act_+1] = left_gripper_target;
    data->ctrl[left_right_finger_act_] = left_gripper_target;
  }
  if (right_left_finger_act_ >= 0) {
    model->actuator_ctrlrange[2*right_left_finger_act_] = right_gripper_target;
    model->actuator_ctrlrange[2*right_left_finger_act_+1] = right_gripper_target;
    data->ctrl[right_left_finger_act_] = right_gripper_target;
  }
  if (right_right_finger_act_ >= 0) {
    model->actuator_ctrlrange[2*right_right_finger_act_] = right_gripper_target;
    model->actuator_ctrlrange[2*right_right_finger_act_+1] = right_gripper_target;
    data->ctrl[right_right_finger_act_] = right_gripper_target;
  }

  // Mode transitions
  if (residual_.current_mode_ == ResidualFn::kModeReach) {
    // print brace force
    // std::cout << (left_reaches ? "RIGHT" : "LEFT") << " brace force: " << brace_contact_force << std::endl;
    // Keep grippers OPEN during reach
    if (left_left_finger_act >= 0) data->ctrl[left_left_finger_act] = 0.03;  // Open
    if (left_right_finger_act >= 0) data->ctrl[left_right_finger_act] = 0.03;
    if (right_left_finger_act >= 0) data->ctrl[right_left_finger_act] = 0.03;
    if (right_right_finger_act >= 0) data->ctrl[right_right_finger_act] = 0.03;
    bool in_contact = (hand_dist < ResidualFn::kHandDistThreshold) && 
                     (reach_contact_force >= ResidualFn::kContactForceThreshold);
    if (in_contact) {
        residual_.current_mode_ = ResidualFn::kModeRetrieve;
        residual_.mode_start_time_ = data->time;
        printf("Switching to RETRIEVE mode at time %.2f\n", data->time);
    } else {
      // Lost contact, reset timer
      residual_.contact_start_time_ = -1;
    }
  } else if (residual_.current_mode_ == ResidualFn::kModeRetrieve) {
    // Check if object brought to torso
    double *torso_pos = SensorByName(model, data, "torso_position");
    double obj_to_torso_dist = mju_dist3(object_pos, torso_pos);
    if (left_reaches) {
      if (left_left_finger_act >= 0) data->ctrl[left_left_finger_act] = 0.0;  // Closed
      if (left_right_finger_act >= 0) data->ctrl[left_right_finger_act] = 0.0;
      // Keep right gripper open
      if (right_left_finger_act >= 0) data->ctrl[right_left_finger_act] = 0.03;
      if (right_right_finger_act >= 0) data->ctrl[right_right_finger_act] = 0.03;
    } else {
      if (right_left_finger_act >= 0) data->ctrl[right_left_finger_act] = 0.0;  // Closed
      if (right_right_finger_act >= 0) data->ctrl[right_right_finger_act] = 0.0;
      // Keep left gripper open
      if (left_left_finger_act >= 0) data->ctrl[left_left_finger_act] = 0.03;
      if (left_right_finger_act >= 0) data->ctrl[left_right_finger_act] = 0.03;
    }
    if (obj_to_torso_dist < 0.2) {
      printf("Object retrieved successfully at time %.2f\n", data->time);
      // Could switch to a new mode or reset here
    }
  }

  // Update mocap target
  mju_copy3(data->mocap_pos, target_position_.data());
  residual_.last_transition_time_ = data->time;
}

void lean::ResetLocked(const mjModel *model) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(1.1, 1.3);
  std::uniform_real_distribution<> dis_y(-0.3, 0.3);
  target_position_ = {dis_x(gen), dis_y(gen), 0.95};
  printf("New target position: %f, %f, %f\n", target_position_[0],
         target_position_[1], target_position_[2]);
  
  // Cache gripper actuator IDs
  left_left_finger_act_ = mj_name2id(model, mjOBJ_ACTUATOR, "left_left_finger");
  left_right_finger_act_ = mj_name2id(model, mjOBJ_ACTUATOR, "left_right_finger");
  right_left_finger_act_ = mj_name2id(model, mjOBJ_ACTUATOR, "right_left_finger");
  right_right_finger_act_ = mj_name2id(model, mjOBJ_ACTUATOR, "right_right_finger");
 
  // Reset mode state
  residual_.current_mode_ = ResidualFn::kModeReach;
  residual_.contact_start_time_ = -1;
  residual_.last_transition_time_ = -1;

  // DEBUG: Print joint order
  // printf("\nJoint order for qpos:\n");
  // for (int i = 0; i < model->njnt; i++) {
  //   const char* jnt_name = mj_id2name(model, mjOBJ_JOINT, i);
  //   int qpos_adr = model->jnt_qposadr[i];
  //   printf("  Joint %d: %s (qpos index %d)\n", i, jnt_name ? jnt_name : "unnamed", qpos_adr);
  // }
}
}  // namespace mjpc