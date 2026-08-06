#ifndef MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_SIMPLE_H_
#define MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_SIMPLE_H_

#include <memory>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {

// Lean Simple -- lean onto a table in a SPECIFIED CONTACT MODE.
//
// WHY THIS EXISTS (2026-08-06). `lean.cc` is 3.6 kloc, 35 cost terms, 8-keyframe
// strategies and a 14 s open-loop target ramp. docs/lean/2026-08-05_mjpc_chain.html
// measured what that buys: the robot braces 8/10 times, and the reaching hand ends
// up 41 mm FURTHER from the target than standing still would leave it, because the
// reach term is 15th of 35 by weight and the thing actually sequencing the motion
// is a hand-authored keyframe ramp, not the planner. Its §9 concluded that the
// replacement should be per-link signed-distance contact costs, no force feedback,
// task-space invariants instead of pose targets, and an objective that dominates
// its own phase. This task is that replacement, built on the shape of
// `lean_simple_gripper.cc` (the 385-line variant) rather than on `lean.cc`.
//
// WHAT IS DIFFERENT, concretely:
//   * 11 cost terms, no phases, no keyframe ramp, no strategy JSON. One static
//     weight vector runs the whole rollout; there is nothing to sequence.
//   * The CONTACT MODE IS AN INPUT. `Brace Elbow`, `Brace Forearm` and
//     `Brace Palm` are three interchangeable seat costs, one per candidate link.
//     Turning a weight on asks for that link on the slab; the same links with
//     weight 0 are held OFF the slab by `Table Keepout`, which reads those very
//     weights (see kTermBrace*) so the mode is specified in exactly one place.
//     The offline enumeration in lean_analysis/contact_select.py emits a subset
//     of {elbow, forearm, palm}; that subset maps onto this task as three numbers.
//   * Distance, never force. Contact force is an output of the constraint solver,
//     discontinuous across make/break, and this planner is a 10-sample CEM
//     differencing it. Signed distance is defined while the link is still in the
//     air and has an unambiguous gradient the whole way in.
//   * No pose targets beyond a weak posture regulariser. Standing is held up by
//     feet-planted + capture-point-in-support-polygon + a height LOWER BOUND, not
//     by Symmetry / Base Height / Left Leg Anchor / Brace Arm Plane.
//
// It shares the model, table and reach target with `Lean H12 Magpie`, so results
// are directly comparable with the S11/S12 pages.
class LeanSimple : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  // Residual term indices. These are the ORDER OF THE <user> SENSORS in the
  // task XML and the residual writes them in this order; Residual() also reads
  // weight_[kTermBrace*] to decide which links are being asked to seat, so the
  // two must not drift apart.
  enum Term {
    kTermBraceElbow = 0,   // dim 1  seat the upper arm on the slab
    kTermBraceForearm,     // dim 1  seat the forearm
    kTermBracePalm,        // dim 1  seat the gripper
    kTermKeepout,          // dim 3  elbow / forearm / palm, when NOT in the mode
    kTermTrunk,            // dim 1  torso + pelvis, never on the slab
    kTermReach,            // dim 3  reaching hand -> reach_target
    kTermBalance,          // dim 2  capture point inside the support polygon
    kTermFeetPlanted,      // dim 4  both feet down and flat
    kTermHeight,           // dim 1  head-above-feet LOWER bound
    kTermPosture,          // dim 27 weak regulariser toward the home key
    kTermJointVel,         // dim 27
    kTermControl,          // dim 27
    kNumTerm
  };

  // Task parameter indices (order of the `residual_*` numerics in the XML).
  static constexpr int kParamHeightMin = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const LeanSimple *task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

    // Seat saturation: a link counts as seated once its lowest brace-surface
    // point is within this of the slab's top plane, and the seat term is exactly
    // zero from there inward -- an attractor with a floor, so it stops fighting
    // the contact solver instead of pulling forever (the `Brace Pos` defect).
    //
    // 0 is not a guess. lean_analysis/seat_calib.py translates a braced pose
    // vertically in 5 mm steps and reports the surface height at which MuJoCo's
    // own narrowphase stops reporting link-vs-slab contact: elbow +6.2 mm,
    // forearm +4.7 mm, palm +4.1 mm. Contact therefore begins somewhere in
    // 0..5 mm of this measure for all three links, and saturating at 0 means the
    // term keeps a little tension on past first contact and never asks for
    // penetration.
    static constexpr double kSeatSaturation = 0.0;

    // How far a link that is NOT in the requested mode must stay off the slab.
    static constexpr double kKeepoutClearance = 0.05;

    // Slab edge that does not count as a seat: the seat terms measure clearance
    // to the face INSET by this much, so "on the table" cannot be satisfied by
    // hooking the lip. The keepout terms use the real face.
    static constexpr double kEdgeKeepout = 0.06;

    // Beyond this the seat term stops growing, so a link parked at the far side
    // of the room does not swamp the objective.
    static constexpr double kSeatCutoff = 0.60;

    // Ankle-roll-link height with the robot standing on the floor (measured at
    // the `home` key: 0.0507 m). `Feet Planted` charges any excess over this.
    static constexpr double kFootRestZ = 0.051;

    // A requested brace link counts toward the SUPPORT REGION once its measured
    // seat gap is within this. Deliberately looser than kSeatSaturation (which
    // is where the seat cost stops pulling) but tight enough that a link waving
    // 3 cm above the slab buys no permission to lean: the calibration sweep puts
    // real contact make/break inside 0..5 mm of the measure.
    static constexpr double kSupportSeatGate = 0.01;
  };

  LeanSimple() : residual_(this) {}

  void TransitionLocked(mjModel *model, mjData *data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

class LeanSimple_H12_Magpie : public LeanSimple {
 public:
  std::string Name() const override { return "Lean Simple H12 Magpie"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/lean/Lean_Simple_H12_Magpie.xml");
  }
};

}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_LEAN_LEAN_SIMPLE_H_
