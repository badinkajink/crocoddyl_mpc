#include "mjpc/tasks/humanoid_bench/lean/lean_simple.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

#include "mujoco/mujoco.h"

namespace mjpc {
namespace {

// --------------------------------------------------------------------------
// Table slab
// --------------------------------------------------------------------------
// The brace surface is the TOP FACE of `table_top_collision`, an axis-aligned
// box. Everything below measures against that plane and against the face's x/y
// extent; nothing is hardcoded, so moving the table in the XML moves the costs
// with it (lean_config.sh does exactly that).
struct Slab {
  bool ok = false;
  double top = 0.0;                 // world z of the top face
  double cx = 0.0, cy = 0.0;        // face centre
  double hx = 0.0, hy = 0.0;        // face half-extents
};

// Distance from a point to the slab's TOP FACE, with the face optionally inset.
//
// Height alone is not clearance: at the home key the robot's torso box bottoms
// out 57 mm BELOW the table's top plane while standing 240 mm clear of the near
// edge, so a pure `z - top` keepout would report the trunk as buried in a table
// it is nowhere near. This charges the horizontal excess too, which is the
// difference between "under the table" and "beside it".
//
// `inset` shrinks the usable face. The seat terms use 60 mm of it so a link
// cannot satisfy "on the table" by hooking the lip; the keepout uses 0 so the
// real slab is what links are held away from.
double SlabClearance(const Slab &s, const double xy[2], double z, double inset) {
  const double ex = std::max(0.0, std::abs(xy[0] - s.cx) - (s.hx - inset));
  const double ey = std::max(0.0, std::abs(xy[1] - s.cy) - (s.hy - inset));
  const double e = std::sqrt(ex * ex + ey * ey);
  const double gap = z - s.top;
  // Over the face: the gap IS the clearance, signed (negative = inside the
  // wood). Off the face: combine the lateral miss with any remaining height.
  return (e > 0.0) ? std::sqrt(e * e + std::max(0.0, gap) * std::max(0.0, gap))
                   : gap;
}

Slab TableSlab(const mjModel *model, const mjData *data) {
  Slab s;
  int g = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");
  if (g < 0) return s;
  s.ok = true;
  s.cx = data->geom_xpos[3 * g + 0];
  s.cy = data->geom_xpos[3 * g + 1];
  s.top = data->geom_xpos[3 * g + 2] + model->geom_size[3 * g + 2];
  s.hx = model->geom_size[3 * g + 0];
  s.hy = model->geom_size[3 * g + 1];
  return s;
}

// --------------------------------------------------------------------------
// Lowest brace-surface point of a link
// --------------------------------------------------------------------------
// Returns the world z of the lowest point of a shape, and its xy. Both are what
// the seat/keepout terms need: the z says how far the link is from the slab, the
// xy says whether it is over the slab at all.
//
// WHY NOT mj_geomDistance. It is exact for primitives and WRONG for this
// robot's mesh geoms, which is fatal because two of the three candidate brace
// links (`*_shoulder_yaw_link`, `*_elbow_link`) carry ONLY a mesh. Measured at
// the shipped `forearm_brace_reach` keyframe while lifting the whole robot
// (lean_analysis/seat_calib.py):
//
//     lift [mm]   upper-arm mesh   forearm mesh   forearm pad (capsule)
//         0          -333.4          -419.6            -35.0
//        50          -290.9          -372.8             14.7
//       400          +170.2          +183.1            364.7
//
// The capsule tracks the lift 1:1. The mesh columns are a third of a metre off
// at contact and are not even affine in the lift. A seat cost built on
// mj_geomDistance over "the link's collidable geoms" therefore reads noise for
// the elbow and the forearm -- which is what the 2026-08-05 per-site experiment
// in lean.cc did, and why nothing it reported about those two links means
// anything.
//
// The analytic version below is exact for the shapes it is given, orientation-
// correct against a horizontal plane, and needs no per-link calibration beyond
// the segment/radius already written into the model.

// Capsule specified in a body's frame (the two links that have no primitive).
double CapsuleLow(const mjModel *model, const mjData *data, int body,
                  const double p1[3], const double p2[3], double r,
                  double xy[2]) {
  const double *pos = data->xpos + 3 * body;
  const double *mat = data->xmat + 9 * body;
  double w1[3], w2[3];
  mju_mulMatVec3(w1, mat, p1);
  mju_addTo3(w1, pos);
  mju_mulMatVec3(w2, mat, p2);
  mju_addTo3(w2, pos);
  const double *low = (w1[2] < w2[2]) ? w1 : w2;
  xy[0] = low[0];
  xy[1] = low[1];
  return low[2] - r;
}

// Named primitive geom (box / capsule / sphere / cylinder).
double GeomLow(const mjModel *model, const mjData *data, int g, double xy[2]) {
  const double *pos = data->geom_xpos + 3 * g;
  const double *mat = data->geom_xmat + 9 * g;
  const double *sz = model->geom_size + 3 * g;
  switch (model->geom_type[g]) {
    case mjGEOM_SPHERE:
      xy[0] = pos[0];
      xy[1] = pos[1];
      return pos[2] - sz[0];
    case mjGEOM_CAPSULE:
    case mjGEOM_CYLINDER: {
      // half-length along the geom's local z, radius sz[0]
      double e1[3] = {0, 0, sz[1]}, e2[3] = {0, 0, -sz[1]}, w1[3], w2[3];
      mju_mulMatVec3(w1, mat, e1);
      mju_addTo3(w1, pos);
      mju_mulMatVec3(w2, mat, e2);
      mju_addTo3(w2, pos);
      const double *low = (w1[2] < w2[2]) ? w1 : w2;
      xy[0] = low[0];
      xy[1] = low[1];
      return low[2] - sz[0];
    }
    case mjGEOM_BOX: {
      double best = 1e6;
      for (int i = 0; i < 8; i++) {
        double c[3] = {(i & 1) ? sz[0] : -sz[0], (i & 2) ? sz[1] : -sz[1],
                       (i & 4) ? sz[2] : -sz[2]};
        double w[3];
        mju_mulMatVec3(w, mat, c);
        mju_addTo3(w, pos);
        if (w[2] < best) {
          best = w[2];
          xy[0] = w[0];
          xy[1] = w[1];
        }
      }
      return best;
    }
    default:
      // A mesh (or anything else) has no trustworthy cheap lower bound here --
      // see the note above. Report "far away" so the term is inert rather than
      // silently wrong.
      xy[0] = pos[0];
      xy[1] = pos[1];
      return pos[2];
  }
}

// Clearance of a whole BODY: the minimum over every collidable PRIMITIVE geom it
// carries. Primitives only, because a mesh has no trustworthy cheap lower bound
// here (see above) -- and `*_keepaway` is skipped because it is a deliberately
// oversized proxy that exists to keep the gripper from swiping the slab, not a
// surface anything rests on.
//
// Doing it per BODY rather than per named geom is what makes the gripper right:
// its jaws (`*_gripper_jaw_a/b`) hang 80 mm off the palm box and touch the table
// FIRST, so a keepout written against `*_gripper_collision` alone reports 40 mm
// of clearance while a jaw is already on the wood. Measured in the first
// rollout: `left_gripper_jaw_a` on the slab for 638 of 1000 sampled frames with
// the palm keepout reporting it clear.
//
// Returns false if the body has no usable primitive (the two mesh-only arm
// links), in which case the caller falls back to the declared capsule.
bool BodyClearance(const mjModel *model, const mjData *data, int body,
                   const Slab &slab, double inset, double *out,
                   double *xy_out = nullptr) {
  bool found = false;
  double best = 0.0, best_xy[2] = {0, 0};
  for (int g = 0; g < model->ngeom; g++) {
    if (model->geom_bodyid[g] != body) continue;
    if (!model->geom_contype[g] && !model->geom_conaffinity[g]) continue;
    if (model->geom_type[g] == mjGEOM_MESH) continue;
    const char *gn = mj_id2name(model, mjOBJ_GEOM, g);
    if (gn && std::strstr(gn, "keepaway")) continue;
    double p[2];
    const double z = GeomLow(model, data, g, p);
    const double c = SlabClearance(slab, p, z, inset);
    if (!found || c < best) {
      best = c;
      best_xy[0] = p[0];
      best_xy[1] = p[1];
      found = true;
    }
  }
  if (found) {
    *out = best;
    if (xy_out) {
      xy_out[0] = best_xy[0];
      xy_out[1] = best_xy[1];
    }
  }
  return found;
}

// True distance from a body's collision shapes to the slab, via mj_geomDistance.
//
// The top-face clearance above answers "how far above the table is this link",
// which is the right question for SEATING and the wrong one for KEEPING OUT: a
// shape can be at zero clearance from the slab without being above it. The
// far-target rollout found exactly that loophole -- it rested the `torso` box on
// the slab's NEAR EDGE (contact reported at x = 0.500, the edge, dist = 0.0)
// while the top-face measure read +50.4 mm, because the box's lowest corner is
// below the table top and 50 mm behind it. mj_geomDistance sees the edge; it is
// also exact for primitives, which is what every trunk and gripper shape is.
//
// Mesh geoms are still excluded (see the note above -- the function is not
// usable on them), so this covers the trunk and the gripper but not the two
// mesh-only arm links, whose keepout stays on the top-face measure.
// contype/conaffinity are NOT filtered here: this is a geometric query, and the
// `*_pad` proxies are the right shape for the arm even though they only collide
// through explicit pairs.
bool BodyGeomDistance(const mjModel *model, const mjData *data, int body,
                      int slab_geom, double distmax, double *out) {
  if (slab_geom < 0) return false;
  bool found = false;
  double best = distmax;
  for (int g = 0; g < model->ngeom; g++) {
    if (model->geom_bodyid[g] != body) continue;
    if (model->geom_type[g] == mjGEOM_MESH) continue;
    const char *gn = mj_id2name(model, mjOBJ_GEOM, g);
    if (gn && std::strstr(gn, "keepaway")) continue;
    const double c = mj_geomDistance(model, data, g, slab_geom, distmax, nullptr);
    if (!found || c < best) {
      best = c;
      found = true;
    }
  }
  if (found) *out = best;
  return found;
}

// The three candidate brace links. `*_shoulder_yaw_link` and `*_elbow_link`
// ship ONLY a mesh, so each carries a capsule that stands in for its underside:
// the forearm's is `*_forearm_pad`'s own fromto and radius (35 mm, the model's
// declared brace proxy); the upper arm ships no pad, so its radius is the mesh
// bounding-box half-width (42-49 mm -> 45). seat_calib.py confirms both read
// within 6 mm of zero at the exact height MuJoCo's narrowphase makes and breaks
// the contact. The gripper needs no fallback -- it is all primitives.
struct LinkSpec {
  const char *body;      // body carrying the brace surface
  bool capsule;          // true: no usable primitive, use p1/p2/r
  double p1[3], p2[3];
  double r;
};

const LinkSpec kBraceLinks[3] = {
    {"%s_shoulder_yaw_link", true,
     {0.002, -0.007, -0.030}, {0.002, -0.007, -0.182}, 0.045},
    {"%s_elbow_link", true,
     {0.020, -0.010, -0.015}, {0.110, -0.030, -0.015}, 0.035},
    {"%s_magpie_gripper", false, {0, 0, 0}, {0, 0, 0}, 0.0},
};

// Bodies that must never take load from the table. This is the "do not rest
// your chest on it" term: S12 §6.1 measured a seed putting 398 N -- 60% of body
// weight -- through the trunk under the old stand cost, and the first Lean
// Simple rollout put the `torso` box on the slab for 571 of 1000 sampled frames
// and the `hip` capsule for 260 with a keepout that only looked at two named
// geoms. Whole bodies, so the pelvis sphere and the head/helmet are covered too.
const char *kTrunkBodies[2] = {"torso_link", "pelvis"};

// Body name with the brace arm's side substituted in.
void SideName(char *out, int n, const char *pattern, const char *side) {
  std::snprintf(out, n, pattern, side);
}

}  // namespace

// --------------------------------------------------------------------------
// Residual
// --------------------------------------------------------------------------
//   0 Brace Elbow    1   seat the upper arm  (weight != 0 => in the mode)
//   1 Brace Forearm  1   seat the forearm
//   2 Brace Palm     1   seat the gripper
//   3 Table Keepout  3   elbow / forearm / palm, held OFF when not in the mode
//   4 Trunk Clear    1   torso + pelvis, never on the slab
//   5 Reach          3   reaching hand -> reach_target
//   6 Balance        2   capture point outside the support region
//   7 Feet Planted   4   both feet at rest height and flat
//   8 Height         1   head-above-feet lower bound
//   9 Posture       27   weak regulariser toward the home key
//  10 Joint Vel.    27
//  11 Control       27
void LeanSimple::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                      double *residual) const {
  int counter = 0;
  const int nu = model->nu;

  // Which arm braces. 0 = left (the handedness of Allen's target, whose y is
  // negative => the RIGHT hand reaches). Anything else = right.
  const int brace_arm =
      static_cast<int>(GetNumberOrDefault(0.0, model, "brace_arm"));
  const char *brace = (brace_arm == 0) ? "left" : "right";
  const char *reach = (brace_arm == 0) ? "right" : "left";

  const Slab slab = TableSlab(model, data);
  const int slab_geom = mj_name2id(model, mjOBJ_GEOM, "table_top_collision");

  // ---- clearance of each candidate link from the slab -------------------- //
  // seat_gap[i]: distance to the USABLE face (inset 60 mm) -- what the seat
  //              terms drive to zero.
  // keep_gap[i]: distance to the real face -- what the keepout terms hold open.
  // Both are computed from the lowest point of the link's brace surface, which
  // is the point that would touch.
  // seat_xy[i]: world xy of that lowest point, used by the balance term to put a
  //             seated brace into the support region.
  double seat_gap[4], keep_gap[4], seat_xy[3][2];
  for (int i = 0; i < 4; i++) {
    seat_gap[i] = keep_gap[i] = kSeatCutoff;
  }
  for (int i = 0; i < 3; i++) {
    seat_xy[i][0] = seat_xy[i][1] = 0.0;
  }
  if (slab.ok) {
    for (int i = 0; i < 3; i++) {
      char name[64];
      SideName(name, sizeof(name), kBraceLinks[i].body, brace);
      int b = mj_name2id(model, mjOBJ_BODY, name);
      if (b < 0) continue;
      if (kBraceLinks[i].capsule) {
        double p[2] = {0, 0};
        const double z =
            CapsuleLow(model, data, b, kBraceLinks[i].p1, kBraceLinks[i].p2,
                       kBraceLinks[i].r, p);
        seat_gap[i] = SlabClearance(slab, p, z, kEdgeKeepout);
        keep_gap[i] = SlabClearance(slab, p, z, 0.0);
        seat_xy[i][0] = p[0];
        seat_xy[i][1] = p[1];
      } else {
        double s = 0.0, k = 0.0;
        if (BodyClearance(model, data, b, slab, kEdgeKeepout, &s, seat_xy[i]))
          seat_gap[i] = s;
        if (BodyClearance(model, data, b, slab, 0.0, &k, nullptr)) keep_gap[i] = k;
        // ... and the true distance, which also sees an edge contact.
        double gd = 0.0;
        if (BodyGeomDistance(model, data, b, slab_geom, kSeatCutoff, &gd)) {
          keep_gap[i] = std::min(keep_gap[i], gd);
        }
      }
    }
    // trunk: closest primitive on the torso or the pelvis. Never a candidate
    // contact, so it has no seat term -- only a keepout.
    for (int k = 0; k < 2; k++) {
      int b = mj_name2id(model, mjOBJ_BODY, kTrunkBodies[k]);
      if (b < 0) continue;
      double c = 0.0;
      if (BodyGeomDistance(model, data, b, slab_geom, kSeatCutoff, &c)) {
        keep_gap[3] = std::min(keep_gap[3], c);
      } else if (BodyClearance(model, data, b, slab, 0.0, &c)) {
        keep_gap[3] = std::min(keep_gap[3], c);
      }
    }
  }

  // ---- 0..2: seat costs, one per candidate link -------------------------- //
  // One-sided and SATURATING: pull until the surface reaches the slab, then
  // exactly zero. Not force-based -- see the class comment.
  //
  // `seat_depth` moves the saturation point, in metres, INTO the slab. 0 is the
  // shipped default and stops exactly at the calibrated surface. A small
  // negative value is not the old "aim 115 mm inside the wood" mistake: the
  // calibration only localises first contact to within 0..5 mm of this measure,
  // so a -0.005 target is the calibration's own uncertainty, and the contact
  // solver stops the link at the real surface regardless. It exists because the
  // last few millimetres are otherwise below the planner's resolution -- at
  // weight 300 a 3 mm gap is 0.9 cost units against a 10-sample CEM, which is
  // noise, and the measured result is a forearm that parks 3-5 mm off the slab.
  const double seat_sat =
      kSeatSaturation + GetNumberOrDefault(0.0, model, "seat_depth");
  for (int i = 0; i < 3; i++) {
    residual[counter++] =
        std::min(kSeatCutoff, std::max(0.0, seat_gap[i] - seat_sat));
  }

  // ---- 3: keepout for the candidate links NOT in the requested mode ------- //
  // The mode is specified once, as the three seat weights; this term reads them
  // rather than taking a second, separately-editable list of links. A link the
  // user asked to seat is exempt from its own keepout.
  for (int i = 0; i < 3; i++) {
    residual[counter++] = (weight_[kTermBraceElbow + i] != 0.0)
                              ? 0.0
                              : std::max(0.0, kKeepoutClearance - keep_gap[i]);
  }

  // ---- 4: trunk clearance ------------------------------------------------- //
  // Separate from the link keepout because it is a different KIND of statement.
  // A non-mode link on the slab is a mode violation; the trunk on the slab is
  // the robot supporting itself on its chest, which S12 §6.1 measured at 398 N
  // (60% of body weight) and which is not something to deploy. Its own term so
  // it can carry a near-hard weight without also forcing the gripper of the
  // bracing arm to be held clear at the same strength -- the gripper hangs off
  // the very forearm being seated, and at the shipped braced keyframe it clears
  // the slab by 9 mm, so a large weight there buys a wrist contortion rather
  // than a cleaner brace.
  //
  // Measured need: at Table Keepout = 200 (one shared term), the far-target
  // rollout finished with the `torso` box on the slab in 309 of 309 sampled
  // frames and the `hip` capsule in 260.
  residual[counter++] = std::max(0.0, kKeepoutClearance - keep_gap[3]);

  // ---- 5: reach ---------------------------------------------------------- //
  // The objective. Weighted to dominate: S12 §9 measured the reach term sitting
  // 15th of 35 by weight in the phase whose NAME is `brace_reach`, and measured
  // the reach it produced (-41 mm against standing still).
  double target[3] = {0.9047, -0.2348, 1.0982};
  {
    int n = mj_name2id(model, mjOBJ_NUMERIC, "reach_target");
    if (n >= 0 && model->numeric_size[n] >= 3) {
      mju_copy3(target, model->numeric_data + model->numeric_adr[n]);
    }
  }
  char hand_site[64];
  SideName(hand_site, sizeof(hand_site), "%s_hand", reach);
  int hs = mj_name2id(model, mjOBJ_SITE, hand_site);
  if (hs >= 0) {
    mju_sub3(residual + counter, data->site_xpos + 3 * hs, target);
  } else {
    mju_zero3(residual + counter);
  }
  counter += 3;

  // ---- 6: balance -------------------------------------------------------- //
  // Capture point outside the support REGION, in the frame the feet define.
  //
  // Two departures from humanoid_bench's balance, and the second is the whole
  // reason a brace is worth anything:
  //
  //   * A RECTANGLE, not the ankle-to-ankle SEGMENT. The segment penalises any
  //     forward excursion -- i.e. it penalises leaning, which is the task.
  //   * The region EXTENDS FORWARD TO A SEATED BRACE. A contact that is bearing
  //     load is part of the support region; that is what bracing is for. With a
  //     feet-only region the cost forbids exactly the excursion the brace makes
  //     safe, and the two terms fight: measured at the far target under a
  //     feet-only region, the robot seats its elbow and then refuses to extend
  //     (+0.064 m of reach out of 0.556 m needed).
  //
  // The extension is gated on MEASURED seating (gap <= kSupportSeatGate), not on
  // having been requested: asking for a contact does not widen anything until
  // the link is actually down. `brace_support=0` restores the feet-only region
  // so the two can be A/B'd from one binary.
  //
  // Not modelled: whether the seated contact can actually carry the load its
  // friction cone would have to. That is the offline QP's job (contact_select.py
  // certifies it per mode); here the region is geometric.
  {
    const double *com = SensorByName(model, data, "robot_com");
    const double *comvel = SensorByName(model, data, "robot_com_vel");
    const double *fl = SensorByName(model, data, "foot_left_pos");
    const double *fr = SensorByName(model, data, "foot_right_pos");
    const double *flf = SensorByName(model, data, "foot_left_forward");
    const double *frf = SensorByName(model, data, "foot_right_forward");
    if (com && comvel && fl && fr && flf && frf) {
      double cp[2] = {com[0] + 0.3 * comvel[0], com[1] + 0.3 * comvel[1]};
      double c[2] = {0.5 * (fl[0] + fr[0]), 0.5 * (fl[1] + fr[1])};
      double fwd[2] = {0.5 * (flf[0] + frf[0]), 0.5 * (flf[1] + frf[1])};
      double n = mju_sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1]);
      if (n < 1e-6) {
        fwd[0] = 1.0;
        fwd[1] = 0.0;
      } else {
        fwd[0] /= n;
        fwd[1] /= n;
      }
      const double lat[2] = {-fwd[1], fwd[0]};
      const double d[2] = {cp[0] - c[0], cp[1] - c[1]};
      const double stance[2] = {fl[0] - fr[0], fl[1] - fr[1]};
      // half-length: ankle centre to toe/heel. half-width: half the stance plus
      // the sole's own half-width.
      const double half_len = 0.13;
      const double half_wid =
          0.5 * std::abs(stance[0] * lat[0] + stance[1] * lat[1]) + 0.04;

      // forward edge of the support region: the toes, or a seated brace if one
      // is further out.
      double fwd_edge = half_len;
      if (GetNumberOrDefault(1.0, model, "brace_support") != 0.0) {
        for (int i = 0; i < 3; i++) {
          if (weight_[kTermBraceElbow + i] == 0.0) continue;
          if (seat_gap[i] > kSupportSeatGate) continue;
          fwd_edge = std::max(fwd_edge, (seat_xy[i][0] - c[0]) * fwd[0] +
                                            (seat_xy[i][1] - c[1]) * fwd[1]);
        }
      }
      const double along = d[0] * fwd[0] + d[1] * fwd[1];
      residual[counter++] = (along >= 0.0) ? std::max(0.0, along - fwd_edge)
                                           : std::max(0.0, -along - half_len);
      residual[counter++] = std::max(
          0.0, std::abs(d[0] * lat[0] + d[1] * lat[1]) - half_wid);
    } else {
      residual[counter++] = 0.0;
      residual[counter++] = 0.0;
    }
  }

  // ---- 7: feet planted --------------------------------------------------- //
  // Height is one-sided (a foot may not rise; it is already on the floor) and
  // the z-axis terms keep the soles flat, which is what stops the robot from
  // rolling onto its toes as it leans.
  {
    const double *fl = SensorByName(model, data, "foot_left_pos");
    const double *fr = SensorByName(model, data, "foot_right_pos");
    const double *ul = SensorByName(model, data, "foot_left_up");
    const double *ur = SensorByName(model, data, "foot_right_up");
    residual[counter++] = fl ? std::max(0.0, fl[2] - kFootRestZ) : 0.0;
    residual[counter++] = fr ? std::max(0.0, fr[2] - kFootRestZ) : 0.0;
    residual[counter++] = ul ? 1.0 - ul[2] : 0.0;
    residual[counter++] = ur ? 1.0 - ur[2] : 0.0;
  }

  // ---- 8: height, as a LOWER BOUND --------------------------------------- //
  // A target height fights the lean (leaning lowers the head); a floor does not.
  // Standing head-above-feet is 1.677 m at the home key.
  {
    const double *head = SensorByName(model, data, "head_position");
    const double *fl = SensorByName(model, data, "foot_left_pos");
    const double *fr = SensorByName(model, data, "foot_right_pos");
    double hmin = parameters_.size() > kParamHeightMin
                      ? parameters_[kParamHeightMin]
                      : 1.20;
    double v = 0.0;
    if (head && fl && fr) {
      v = std::max(0.0, hmin - (head[2] - 0.5 * (fl[2] + fr[2])));
    }
    residual[counter++] = v;
  }

  // ---- 9: posture -------------------------------------------------------- //
  // Weak, and the ONLY pose term in the stack. Keyframe 0 is `home`.
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, nu);
  counter += nu;

  // ---- 10: joint velocity ------------------------------------------------- //
  mju_copy(residual + counter, data->qvel + 6, nu);
  counter += nu;

  // ---- 11: control ------------------------------------------------------- //
  // Position servos: ctrl IS a joint angle, so this regularises toward home in
  // the same units as Posture.
  mju_sub(residual + counter, data->ctrl, model->key_qpos + 7, nu);
  counter += nu;

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

// --------------------------------------------------------------------------
// Transition
// --------------------------------------------------------------------------
// There is nothing to sequence: one static weight vector runs the whole
// rollout. All this does is park the mocap marker on the reach target so the
// GUI and the rendered videos show where the hand is being asked to go.
void LeanSimple::TransitionLocked(mjModel *model, mjData *data) {
  if (model->nmocap < 1) return;
  int n = mj_name2id(model, mjOBJ_NUMERIC, "reach_target");
  if (n >= 0 && model->numeric_size[n] >= 3) {
    mju_copy3(data->mocap_pos, model->numeric_data + model->numeric_adr[n]);
  }
}

}  // namespace mjpc
