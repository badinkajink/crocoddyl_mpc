#include "mjpc/tasks/humanoid_bench/lean/lean.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "mujoco/mujoco.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc {

namespace {
// Target (post-ramp) reach + brace scales for each named phase. Kept in one
// place so the residual and the transition logic can't drift out of sync.
inline void PhaseTargetScales(const std::string& name,
                              double& reach, double& brace_pos) {
  reach = 1.0;
  brace_pos = 1.0;
  if (name == "stand_up")        { reach = 0.0; brace_pos = 0.0; }
  else if (name == "arm_extend") { reach = 0.3; brace_pos = 1.0; }
  else if (name == "lean_plant") { reach = 0.7; brace_pos = 1.0; }
  // lean_reach / lean_reach_ext / leg_lift_arm_plant / deep_reach → 1.0/1.0.
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
  // is_leg_lift_stage: arm + elbow + ankle contact (3 contacts)
  const bool is_leg_lift_stage_early = (active_contact_count_early >= 3);

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
  PhaseTargetScales(residual_keyframe_.name, target_reach_scale,
                    target_brace_pos_scale);

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

  // ----- object position ----- //
  double const *object_pos = SensorByName(model, data, "object_pos");

  // ----- Determine which hand reaches and which braces ----- //
  double const *left_hand_pos = SensorByName(model, data, "left_hand_pos");
  double const *right_hand_pos = SensorByName(model, data, "right_hand_pos");

  // Right arm always braces on the table; left arm always reaches for the
  // object. body1=28 in the contact keyframes targets the right hand body, so
  // the reaching/bracing assignment must stay fixed — dynamic switching based
  // on object position causes both arms to be pulled toward the table
  // simultaneously, creating an irresolvable contradiction.
  constexpr bool left_reaches = true;
  double const *reaching_hand = left_hand_pos;
  double const *bracing_hand  = right_hand_pos;

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
  // During leg-lift/deep-lean the torso goes nearly horizontal so head drops
  // well below the 1.65m goal. Scale down to 0.35x so the planner isn't
  // fighting a huge height penalty while simultaneously trying to lean.
  double height_scale = is_leg_lift_stage_early ? 0.35 : 1.0;
  residual[counter++] = height_scale * (head_feet_error - height_goal);

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

  // is leaning - modified to be less strict than standing.
  // floored at 0.3 so balance never fully turns off when falling.
  double leaning =
      torso_height / mju_sqrt(torso_height * torso_height + 0.65 * 0.65) - 0.2;
  leaning = mju_max(leaning, 0.3);

  // When arm is on the table it provides forward support — relax balance so
  // the planner doesn't fight the lean by sliding the left foot forward.
  double balance_scale = any_arm_contact ? 0.35 : 1.0;
  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], leaning * balance_scale, 2);

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

  // ----- pelvis tilt ----- //
  // Upright stages: keep pelvis near-vertical (target 0.85).
  // Leg-lift/deep-lean stages: unconstrained — the arm brace takes over and
  // the torso needs to go near-horizontal.
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  residual[counter++] = is_leg_lift_stage_early ? 0.0 : (pelvis_up[2] - 0.85);

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
  residual[counter++] = data->qpos[7 + 0] - model->key_qpos[7 + 0];
  residual[counter++] = data->qpos[7 + 2] - model->key_qpos[7 + 2];
  residual[counter++] = data->qpos[7 + 6] - model->key_qpos[7 + 6];
  residual[counter++] = data->qpos[7 + 8] - model->key_qpos[7 + 8];

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
  mju_scl3(&residual[counter], &residual[counter], phase_brace_pos_scale);
  counter += 3;

  // Per-phase brace-force reference (Opt2Skill, arXiv 2409.20514).
  // If the strategy JSON for the current keyframe specifies brace_force_target
  // (>= 0), use it. Otherwise fall back to the legacy default: 15 N when any
  // contact pair is active, 0 N during contact-free phases (stand_up).
  bool is_active_contact =
      (residual_keyframe_.contact_pairs[0].body1 != mjpc::humanoid::kNotSelectedInteract);
  double desired_brace_force = residual_keyframe_.brace_force_target >= 0.0
                                   ? residual_keyframe_.brace_force_target
                                   : (is_active_contact ? 15.0 : 0.0);
  residual[counter++] = desired_brace_force - brace_contact_force;

  // ------ object distance (reaching hand) ------ //
  // Phase-gated: zero during stand_up so the planner doesn't lunge.
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  mju_scl3(&residual[counter], &residual[counter],
           phase_reach_scale * leaning);
  counter += 3;

  // ----- reaching hand distance to object ----- //
  mju_sub3(&residual[counter], reaching_hand, object_pos);
  mju_scl3(&residual[counter], &residual[counter], phase_reach_scale);
  counter += 3;

  // ----- foot stability: restoring force toward home XY position ----- //
  // Position-based (not velocity-based) so there's a continuous gradient even
  // when the foot is stationary but displaced. Home positions derived from the
  // H1_2 kinematic chain at the home keyframe (qpos x=0.19, all joints=0).
  // Right foot freed during leg-lift stages (right leg lifts, left foot is sole ground support).
  static constexpr double kRightFootHomeXY[2] = {0.19, -0.163};
  static constexpr double kLeftFootHomeXY[2]  = {0.19,  0.163};

  bool is_leg_lift_stage = is_leg_lift_stage_early;

  // Left foot is the primary ground anchor during all lean stages.
  // Scale 4x as soon as the arm contacts the table, 5x during leg lift.
  // This is needed because balance residual would otherwise slide the foot
  // to reposition the COM — the arm provides the forward support instead.
  double left_foot_scale = is_leg_lift_stage ? 5.0 : (any_arm_contact ? 4.0 : 1.0);

  residual[counter++] = is_leg_lift_stage ? 0.0 : (foot_right_pos[0] - kRightFootHomeXY[0]);
  residual[counter++] = is_leg_lift_stage ? 0.0 : (foot_right_pos[1] - kRightFootHomeXY[1]);
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
  // Right: knee x only (right leg lifts, so thigh constraint not needed there).
  double *left_knee_pos_3d  = SensorByName(model, data, "left_knee_pos");
  double *right_knee_pos_3d = SensorByName(model, data, "right_knee_pos");
  double left_thigh_mid_x   = 0.5 * (pelvis_pos_3d[0] + left_knee_pos_3d[0]);
  double left_thigh_penalty = mju_max(0.0, left_thigh_mid_x    - (table_front_x - 0.05));
  double right_knee_penalty = mju_max(0.0, right_knee_pos_3d[0] - (table_front_x - 0.06));
  residual[counter++] = left_thigh_penalty;
  residual[counter++] = right_knee_penalty;

  // ----- left leg anchor: prevent left knee from bending during leg lift ----- //
  // During leg-lift stages the left leg is the sole ground support and must stay
  // straight. Penalise the left knee dropping below standing height (~0.42m).
  // Zero in all other stages.
  residual[counter++] = is_leg_lift_stage ? mju_max(0.0, 0.42 - left_knee_pos_3d[2]) : 0.0;

  // ----- right foot lift: reward foot off ground during leg-lift stages ----- //
  // Penalises right foot below 0.25m when the right leg should be lifted.
  // Zeroed in all other stages so it doesn't interfere with normal standing.
  residual[counter++] = is_leg_lift_stage ? mju_max(0.0, 0.25 - foot_right_pos[2]) : 0.0;

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
  static constexpr double kJointVelLimit[27] = {
      // L_hip_yaw, L_hip_pitch, L_hip_roll, L_knee, L_ank_p, L_ank_r
      23.0, 23.0, 23.0, 14.0, 9.0, 9.0,
      // R_hip_yaw, R_hip_pitch, R_hip_roll, R_knee, R_ank_p, R_ank_r
      23.0, 23.0, 23.0, 14.0, 9.0, 9.0,
      // torso
      23.0,
      // L_sho_p, L_sho_r, L_sho_y, L_elbow, L_wr_r, L_wr_p, L_wr_y
      9.0, 9.0, 20.0, 20.0, 31.4, 31.4, 31.4,
      // R_sho_p, R_sho_r, R_sho_y, R_elbow, R_wr_r, R_wr_p, R_wr_y
      9.0, 9.0, 20.0, 20.0, 31.4, 31.4, 31.4,
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
  // ---- DEBUG: print leg stability diagnostics every ~0.5 s ---- //
  static int debug_tick = 0;
  if (++debug_tick % 33 == 0) {  // ~0.5 s at timestep=0.015
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
  }
  // ---- END DEBUG ---- //

  // keep mocap target updated for the existing hand-reach residuals
  double const *object_pos = SensorByName(model, data, "object_pos");
  double const *left_hand_pos_tr = SensorByName(model, data, "left_hand_pos");
  double hand_dist = mju_dist3(left_hand_pos_tr, object_pos);
  if (hand_dist < 0.05) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(1.4, 1.6);
    std::uniform_real_distribution<> dis_y(-0.3, 0.3);
    target_position_ = {dis_x(gen), dis_y(gen), 0.73};
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

  // Helper: snapshot the scales currently in effect (mid-ramp possible) so
  // the next phase ramps smoothly out of them. The lerp matches what
  // Residual() is doing, so prev_* always equals what the rollouts are
  // actually seeing at the moment of transition.
  auto SnapshotEffectiveScales = [&]() {
    double r_t, b_t;
    PhaseTargetScales(residual_.residual_keyframe_.name, r_t, b_t);
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
  };

  if (!motion_strategy_.HasKeyframes() ||
      requested_strategy != current_strategy_) {
    current_strategy_ = requested_strategy;
    motion_strategy_.ClearKeyframes();
    motion_strategy_.LoadStrategy(kStrategyNames[current_strategy_],
                                  kLeanStrategyFilePath);
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    // Cold start (or strategy switch): no previous phase, prev scales = 0
    // so the first ramp climbs cleanly out of (0, 0) into stand_up's
    // targets (which are also 0, 0 — i.e. no ramp during stand_up).
    residual_.keyframe_start_time_      = data->time;
    residual_.prev_phase_reach_scale_   = 0.0;
    residual_.prev_phase_brace_pos_scale_ = 0.0;
    return;
  }

  const mjpc::humanoid::ContactKeyframe& current_kf =
      motion_strategy_.GetCurrentKeyframe();
  const double total_distance = motion_strategy_.CalculateTotalKeyframeDistance(
      data, mjpc::humanoid::ContactKeyframeErrorType::kNorm);

  if (data->time - motion_strategy_.GetCurrentKeyframeStartTime() >
          current_kf.time_limit &&
      total_distance > current_kf.target_distance_tolerance) {
    // Time-limit reset (strategy restarts from keyframe 0). Save the scales
    // that were just in effect so the next ramp blends from them.
    SnapshotEffectiveScales();
    motion_strategy_.Reset();
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    residual_.keyframe_start_time_ = data->time;
  } else if (total_distance <= current_kf.target_distance_tolerance &&
             data->time -
                     motion_strategy_.GetCurrentKeyframeSuccessStartTime() >
                 current_kf.success_sustain_time) {
    // Normal phase advance — this is the path that fires after stand_up
    // succeeds. Snapshot first so the new ramp starts from the old scales.
    SnapshotEffectiveScales();
    motion_strategy_.NextKeyframe();
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
    residual_.keyframe_start_time_ = data->time;
  } else if (total_distance > current_kf.target_distance_tolerance) {
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
  }
}

void lean::ResetLocked(const mjModel *model) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis_x(1.4, 1.6);
  std::uniform_real_distribution<> dis_y(-0.3, 0.3);
  target_position_ = {dis_x(gen), dis_y(gen), 0.73};
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