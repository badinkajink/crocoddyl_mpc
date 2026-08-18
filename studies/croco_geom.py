#!/usr/bin/env python3
"""Table keep-out for the crocoddyl OCP, with the geometry MuJoCo actually has.

WHY THIS EXISTS.  The S12 plan drove the bracing gripper 68 mm through the
tabletop, and the reason it could not be tuned away is structural: the OCP's only
table term was a HALF-SPACE barrier, `z_site >= table_z + clearance`, applied to
the arm sites.  A half-space cannot express "below the top surface but outside
the table footprint", and the certified pose needs exactly that -- at q* the
gripper's lowest corner sits ~7 mm BELOW the tabletop plane, hanging over the
table's near edge where there is no table.  Tightening the half-space enough to
exclude the gripper forbids the pose the whole study certified.

WHAT REPLACES IT.  The table top is a BOX, so the honest constraint is the box's
signed distance function.  This module builds

  * the tabletop box (centre + half extents) read off `table_top_collision` in
    the MJCF, so the keep-out and MuJoCo's narrowphase are the same slab;
  * a keep-out point set: one point per collision geom of the bodies that are NOT
    bracing contacts, each carrying the radius of the sphere that bounds that
    geom, so "this geom is clear of the table" becomes "this point is at least r
    from the box";
  * `ActivationModelBoxKeepOut`, a crocoddyl activation evaluating
    0.5 * max(0, r_min - sdf_box(p - c))^2, which is zero everywhere outside the
    inflated box -- including under the overhang -- and grows quadratically
    inside it.

crocoddyl's own `ResidualModelPairCollision` would be the library answer, but it
is compiled only when crocoddyl is built against hpp-fcl/coal and the
conda-forge build in use is not (checked: the symbol is absent).  An activation
on top of the C++ `ResidualModelFrameTranslation` needs no new Jacobians -- the
residual already supplies d(p)/dq.

EVERY POINT IS CHECKED AT q* BEFORE IT IS USED (`audit`).  A keep-out that the
certified pose violates is a keep-out that forbids the answer, which is the trap
the half-space fell into.

THREE IMPLEMENTATIONS, ONE FUNCTION (`CROCO_KEEPOUT` picks).  The activation
started as a Python subclass of `crocoddyl.ActivationModelAbstract`, which costs
an interpreter round-trip per point per node per solver sweep -- 86 x 290 =
24 940 of them for the S13 plan, and it dominated both the offline solve and the
MPC step time.  `croco_ext/` holds a C++ transcription.  That was S13, and it was
only half the problem:

  python   one Python activation per point, one cost term per point.  S12/S13's
           starting point.
  cpp      the C++ activation, still one cost term per point.  S13's speed-up:
           50 s -> 14 s offline, 119 -> 76 ms per MPC step.
  fused    (default) ALL points in one C++ `CostModelBoxKeepOut` cost term.

The third exists because the second stopped short.  What a keep-out point costs
is not mostly its own arithmetic: `CostModelSum::calcDiff` accumulates a private
66x66 Lxx, 66x27 Lxu and 27x27 Luu per cost TERM regardless of what the term
computes, which croco_speed.py measures at 1.61 us for a term that computes
nothing at all against 2.20 us for a keep-out point.  86 points is therefore
~130 us per node of pure bookkeeping and a 5.4 MB per-node working set, and the
inactive points -- nearly all of them, nearly always -- pay it in full.  The
fused term accumulates once and skips inactive points at the cost of one SDF
evaluation.  `croco_ext/test_keepout.py` checks all three give the same cost,
Lx and Lxx on the real planned states.
"""

import os
import sys

import numpy as np
import mujoco

import contact_select as cs
import croco_bridge as cb

crocoddyl = cb.import_crocoddyl()

# --------------------------------------------------------------------------- #
# The C++ activation, if it has been built.  Imported AFTER crocoddyl (it calls
# bp::import("crocoddyl") itself, but the RTLD_GLOBAL dance in croco_bridge has
# to have happened first) and never fatally: a missing .so falls back to the
# Python class with a one-line note, so a fresh checkout still runs.
_EXT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "croco_ext")
_MODE = os.environ.get("CROCO_KEEPOUT", "fused")
_CPP = None
if _MODE != "python":
    try:
        if _EXT_DIR not in sys.path:
            sys.path.insert(0, _EXT_DIR)
        import croco_keepout as _CPP          # noqa: N813
    except ImportError:
        _CPP = None

_FUSED = _MODE == "fused" and hasattr(_CPP, "CostModelBoxKeepOut")
IMPL = "fused" if _FUSED else ("c++" if _CPP is not None else "python")

# Bodies whose geoms are candidates for keep-out: the bracing arm from the wrist
# out, the reaching arm's hand, and the head.  Legs and torso are nowhere near
# the slab in this maneuver and every point costs a Python callback per node.
_BRACE = cs.BRACE_ARM
_REACH = "right" if _BRACE == "left" else "left"
KEEPOUT_BODIES = [
    f"{_BRACE}_shoulder_yaw_link", f"{_BRACE}_elbow_link",
    f"{_BRACE}_wrist_roll_link", f"{_BRACE}_wrist_pitch_link",
    f"{_BRACE}_wrist_yaw_link", f"{_BRACE}_magpie_gripper",
    f"{_REACH}_wrist_yaw_link", f"{_REACH}_magpie_gripper",
]

# MJCF body -> the site name whose contact model already owns it.  A body that is
# a prescribed contact must NOT also be kept out of the table: the contact IS the
# table.
SITE_BODY = {s: cs.SITES[s][0] for s in cs.SITES}


def table_box(m, d):
    """(centre, half-extents) of `table_top_collision`, in world.

    Asserts the box is axis-aligned, because the SDF below is written in world
    axes.  It is (table body quat = identity) and this is what makes the cheap
    formula legitimate rather than approximately true.
    """
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    R = d.geom_xmat[g].reshape(3, 3)
    if np.max(np.abs(R - np.eye(3))) > 1e-9:
        raise RuntimeError("table_top_collision is not axis-aligned in world; "
                           "the box SDF in this module assumes it is")
    return d.geom_xpos[g].copy(), m.geom_size[g].copy()


def _geom_samples(m, g):
    """[(body-frame point, radius)] whose union covers one collision geom.

    A single bounding sphere per geom is unusable here and the audit is what says
    so: the bracing wrist's mesh bbox has ||half|| = 100 mm, so its bounding
    sphere overlaps the slab at the certified pose by 58 mm.  A keep-out built
    from bounding spheres forbids q* outright -- the same defect as the
    half-space, arrived at from the other side.

    So each geom is covered by the extreme points of its own shape instead:
      box / mesh   the 8 corners of the (bbox) half-extents, radius 0.  Exact for
                   a box; for a mesh it is the bbox, which is what MuJoCo's
                   `geom_size` reports and what the study has always used.
      capsule      the two segment endpoints, radius = the capsule radius (exact).
      sphere       the centre, radius r (exact).
      cylinder     the two cap centres, radius r (exact enough: it ignores the
                   flat cap rim, which is 5 mm of a part that never approaches).
    """
    t = m.geom_type[g]
    s = np.asarray(m.geom_size[g], float)
    pos = np.asarray(m.geom_pos[g], float)
    quat = np.asarray(m.geom_quat[g], float)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [(pos.copy(), float(s[0]))]
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        ax = R[:, 2] * float(s[1])
        return [(pos + ax, float(s[0])), (pos - ax, float(s[0]))]
    out = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                out.append((pos + R @ (np.array([sx, sy, sz]) * s[:3]), 0.0))
    return out


def keepout_points(m, subset, bodies=None, geoms=None, skip_contacts=False):
    """[(body, body-frame offset, radius, geom name)] covering the kept-out geoms.

    `skip_contacts` drops the bodies that carry a contact in `subset`.  It is OFF
    by default and that is a correction, not a preference.  Excluding contact
    bodies sounds right -- their contact model places them on the table, so a
    keep-out would fight it -- but the contact only exists during the BRACED
    phase, and during the approach the same body is completely free.  Measured:
    with the gripper excluded because `palm` is a contact, the approach drove it
    44 mm through the slab in the first 0.1 s and the contact it was heading for
    made no difference at all.  Because every threshold is taken at q* (see
    `keepout_spec`), a contact body's own keep-out reads "do not go deeper than
    the certified pose does", which is satisfied by the braced phase by
    construction and is exactly the constraint the approach was missing.

    `geoms`, if given, restricts the set to those geom names -- a lever for
    keeping the point count (and so the number of Python activation callbacks per
    node) down to the geoms that a diagnostic run showed reach the slab.
    """
    bodies = KEEPOUT_BODIES if bodies is None else bodies
    contact_bodies = {SITE_BODY[s] for s in subset if s in SITE_BODY}
    out = []
    for name in bodies:
        if skip_contacts and name in contact_bodies:
            continue
        b = cs.bid(m, name)
        if b < 0:
            continue
        for g in range(m.ngeom):
            if m.geom_bodyid[g] != b:
                continue
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue
            gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
            if gname.endswith("keepaway"):
                continue                    # a planning-only proxy, not collision
            if geoms is not None and gname not in geoms:
                continue
            for off, r in _geom_samples(m, g):
                out.append((name, off, r, gname))
    return out


def sdf_box(p, half):
    """Signed distance from `p` (already box-centred) to an axis-aligned box."""
    a = np.abs(p) - half
    amax = np.maximum(a, 0.0)
    n = np.linalg.norm(amax)
    return float(n + min(float(np.max(a)), 0.0))


def sdf_box_grad(p, half):
    a = np.abs(p) - half
    sg = np.sign(p)
    sg[sg == 0] = 1.0
    if np.max(a) > 0.0:
        amax = np.maximum(a, 0.0)
        n = np.linalg.norm(amax)
        return sg * amax / max(n, 1e-12)
    g = np.zeros(3)
    i = int(np.argmax(a))
    g[i] = sg[i]
    return g


class ActivationModelBoxKeepOut(crocoddyl.ActivationModelAbstract):
    """0.5 * max(0, r_min - sdf_box(r))^2 on a 3-vector residual.

    Pair it with `ResidualModelFrameTranslation(state, fid, table_centre, nu)` so
    the residual handed in is already the point relative to the box centre.

    Zero (and zero-gradient) everywhere the point is clear of the inflated box,
    so it costs nothing on the certified pose and does not bias the solution the
    way a always-on half-space penalty does.  The Hessian is the Gauss-Newton
    outer product of the SDF gradient, which is exact for the active branch away
    from the box's edges and corners.
    """

    def __init__(self, half, r_min):
        crocoddyl.ActivationModelAbstract.__init__(self, 3)
        self.half = np.asarray(half, float)
        self.r_min = float(r_min)

    def calc(self, data, r):
        v = self.r_min - sdf_box(np.asarray(r), self.half)
        data.a_value = float(0.5 * max(v, 0.0) ** 2)

    def calcDiff(self, data, r):
        r = np.asarray(r)
        v = self.r_min - sdf_box(r, self.half)
        if v <= 0.0:
            data.Ar = np.zeros(3)
            data.Arr = np.zeros((3, 3))
            return
        g = sdf_box_grad(r, self.half)
        data.Ar = -v * g
        # NB crocoddyl stores Arr as a DIAGONAL matrix and its Python setter
        # keeps only the diagonal, so the off-diagonal half of this outer product
        # never reaches the solver.  Written as the outer product anyway because
        # that is the Gauss-Newton term this is meant to be; the C++ version
        # writes diag(g^2) directly and the two agree exactly (croco_ext tests).
        data.Arr = np.outer(g, g)


def activation(half, r_min):
    """The keep-out activation, C++ if available and Python otherwise.

    Same function either way -- see `croco_ext/test_keepout.py`, which checks
    a_value, Ar and diag(Arr) agree to 0.0 over 2 000 random residuals plus every
    face, edge and corner of the box.
    """
    if _CPP is not None:
        h = np.asarray(half, float)
        return _CPP.ActivationModelBoxKeepOut(float(h[0]), float(h[1]),
                                              float(h[2]), float(r_min))
    return ActivationModelBoxKeepOut(half, r_min)


def fused_cost(state, nu, half, center, spec):
    """All keep-out points as ONE crocoddyl cost, or None if not available.

    `spec` is `keepout_spec`'s output AFTER croco_plan has resolved each point to
    a Pinocchio frame (the `fid` key).  Returns a `CostModelBoxKeepOut`, which the
    caller adds to its CostModelSum under a single name and a single weight --
    the same weight the per-point stack applied to every point, so the two are
    the same objective.
    """
    if not _FUSED or not spec:
        return None
    h = np.asarray(half, float)
    c = np.asarray(center, float)
    return _CPP.CostModelBoxKeepOut(
        state, int(nu), float(h[0]), float(h[1]), float(h[2]),
        float(c[0]), float(c[1]), float(c[2]),
        [int(p["fid"]) for p in spec], [float(p["thresh"]) for p in spec])


def keepout_spec(m, d, q_star, subset, bodies=None, geoms=None, prune=0.35):
    """Keep-out points with a per-point threshold the certified pose satisfies.

    threshold = min(radius, sdf at q*), i.e. "do not get closer to the slab than
    the certified pose already is".  Two of the bracing wrist's mesh geoms have
    bounding boxes that dip 16 mm and 3 mm into the tabletop at q* while their
    real meshes do not touch it at all (MuJoCo's narrowphase reports no contact
    there), so a keep-out written on the bbox with threshold = radius would
    forbid the pose the study certified.  Taking q*'s own clearance as the datum
    keeps the constraint honest about the approximation: it forbids going DEEPER
    into the slab than the certified pose does, which is the property that
    actually matters, and it costs exactly zero at q*.

    `prune` drops points further than that (metres) from the slab at q* -- they
    are structurally clear of the table in this maneuver and each surviving point
    costs one Python activation callback per node per solver iteration.
    """
    c, half = table_box(m, d)
    d.qpos[:len(q_star)] = q_star
    mujoco.mj_forward(m, d)
    spec = []
    for body, off, r, gname in keepout_points(m, subset, bodies, geoms):
        s = sdf_box(cs.point_world(m, d, body, off) - c, half)
        if s - r > prune:
            continue
        spec.append(dict(body=body, offset=np.asarray(off, float), geom=gname,
                         radius=float(r), sdf_qstar=float(s),
                         thresh=float(min(r, s))))
    return spec, c, half


def audit(m, d, q_star, subset, points=None, verbose=True):
    """Clearance of every keep-out point at the certified pose.

    A point with clearance < 0 is a constraint the OCP cannot satisfy without
    leaving the pose the static QP certified, which is precisely the failure mode
    the half-space barrier had.  Returns the list with a `slack` field so the
    caller can shrink or drop the offenders instead of discovering them as a
    stalled solve.
    """
    c, half = table_box(m, d)
    pts = keepout_points(m, subset) if points is None else points
    d.qpos[:len(q_star)] = q_star
    mujoco.mj_forward(m, d)
    rows = []
    for body, off, r, gname in pts:
        p = cs.point_world(m, d, body, off)
        s = sdf_box(p - c, half)
        rows.append(dict(body=body, geom=gname, offset=off.tolist(), radius=r,
                         sdf=s, slack=s - r))
    if verbose:
        per = {}
        for x in rows:
            k = x["geom"]
            if k not in per or x["slack"] < per[k]["slack"]:
                per[k] = x
        print(f"keep-out audit at q*: {len(rows)} points on {len(per)} geoms, "
              f"subset {'+'.join(subset) or 'legs_only'}")
        for x in sorted(per.values(), key=lambda z: z["slack"]):
            flag = "  <-- VIOLATES q*" if x["slack"] < 0 else ""
            print(f"  {x['geom']:30s} worst point: sdf={x['sdf']*1000:7.1f} mm  "
                  f"r={x['radius']*1000:5.1f}  slack={x['slack']*1000:+7.1f} mm{flag}")
    return rows


if __name__ == "__main__":
    import sys
    subset = sys.argv[1].split("+") if len(sys.argv) > 1 else ["elbow", "forearm"]
    qfile = sys.argv[2] if len(sys.argv) > 2 else \
        f"runs/2026-08-05_session12/croco/q_{'+'.join(subset)}.txt"
    m, d = cs.load(ik_margin=0.0)
    c, half = table_box(m, d)
    print(f"table box centre {np.round(c, 4)}  half {np.round(half, 4)}  "
          f"top z {c[2] + half[2]:.4f}")
    audit(m, d, np.loadtxt(qfile), subset)
