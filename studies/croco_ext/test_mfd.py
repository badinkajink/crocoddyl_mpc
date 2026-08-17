#!/usr/bin/env python3
"""Check the matrix-free Delassus operator against an explicit one.

The operator is the reason this extension exists, and it is exactly the kind of
thing that can be wrong in a way that still returns plausible numbers -- a wrong
reference frame, a stale decomposition, a damping applied twice.  So the test
does not check that it runs; it checks that G agrees with J M^-1 J^T + mu I
computed independently, at several configurations, for two contact sets, and
that solve() inverts apply().

usage: croco_ext/test_mfd.py          (exit 0 = pass)
"""

import ctypes
import os
import sys

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import croco_bridge as cb              # noqa: E402  first, sets RTLD_GLOBAL
import pinocchio as pin                # noqa: E402
import croco_mfd                       # noqa: E402

DAMPING = 1e-4
TOL = 1e-12


def explicit_delassus(rmodel, rdata, q, cms, cds, damping):
    """G = J M^-1 J^T + mu I, assembled from scratch with dense linear algebra.

    Deliberately NOT pinocchio's sparse route: an independent check should not
    share an implementation with the thing it is checking.
    """
    pin.computeJointJacobians(rmodel, rdata, q)
    for c, d in zip(cms, cds):
        c.calc(rmodel, rdata, d)
    J = np.vstack([np.array(c.jacobian(rmodel, rdata, d))
                   for c, d in zip(cms, cds)])
    M = np.array(pin.crba(rmodel, rdata, q))
    M = np.triu(M) + np.triu(M, 1).T
    G = J @ np.linalg.solve(M, J.T)
    return G + damping * np.eye(G.shape[0]), J


def constraint_set(rmodel, sites, names_6d, names_3d):
    cms = pin.StdVec_ConstraintModel()
    for s in names_6d:
        fr = rmodel.frames[sites[s]]
        cms.append(pin.ConstraintModel(pin.FrameAnchorConstraintModel(
            rmodel, fr.parentJoint, fr.placement)))
    for s in names_3d:
        fr = rmodel.frames[sites[s]]
        cms.append(pin.ConstraintModel(pin.PointContactConstraintModel(
            rmodel, fr.parentJoint, fr.placement)))
    return cms


def main():
    rmodel = cb.build_pin()
    sites = cb.mj_site_frames(rmodel)
    rng = np.random.default_rng(0)
    fails = []

    cases = [("feet only", ("sole_left", "sole_right"), ()),
             ("feet + brace", ("sole_left", "sole_right"),
              ("elbow", "forearm", "palm"))]

    for label, n6, n3 in cases:
        cms = constraint_set(rmodel, sites, n6, n3)
        nc = 6 * len(n6) + 3 * len(n3)
        for trial in range(4):
            # NOT pin.randomConfiguration: the free-flyer's position limits are
            # +/-inf on this model, so it samples NaN for the base translation
            # and every downstream number is NaN.  Integrating a bounded random
            # tangent off `neutral` stays on the manifold and keeps the
            # configuration in a range the robot could actually be in.
            q0 = pin.neutral(rmodel)
            q = q0 if trial == 0 else pin.integrate(
                rmodel, q0, 0.4 * rng.standard_normal(rmodel.nv))
            q = np.ascontiguousarray(q)

            rd_ref = rmodel.createData()
            cds = pin.StdVec_ConstraintData()
            for c in cms:
                cds.append(c.createData())
            G_ref, _ = explicit_delassus(rmodel, rd_ref, q, cms, cds, DAMPING)

            rd = rmodel.createData()
            pin.computeJointJacobians(rmodel, rd, q)
            op = croco_mfd.DelassusMF(rmodel, rd, cms, DAMPING)
            op.compute()
            assert op.size == nc, (op.size, nc)
            G = np.array(op.matrix())

            err = np.linalg.norm(G - G_ref) / np.linalg.norm(G_ref)
            # applyOnTheRight vs solveInPlace: G^-1 (G x) should be x.
            x = rng.standard_normal(nc)
            rt = np.linalg.norm(np.array(op.solve(np.array(op.apply(x)))) - x)
            rt /= np.linalg.norm(x)
            sym = np.linalg.norm(G - G.T) / np.linalg.norm(G)

            ok = err < TOL and rt < 1e-8 and sym < 1e-12
            print(f"{label:14s} nc={nc:2d} q{trial}  |G-Gref|/|Gref|={err:.2e}  "
                  f"round-trip={rt:.2e}  asym={sym:.2e}  "
                  f"{'ok' if ok else 'FAIL'}")
            if not ok:
                fails.append((label, trial, err, rt, sym))

    # The damping has to reach the operator, or every solve is against the wrong
    # system while every apply still looks right.
    cms = constraint_set(rmodel, sites, ("sole_left", "sole_right"), ())
    rd = rmodel.createData()
    q = np.ascontiguousarray(pin.neutral(rmodel))
    pin.computeJointJacobians(rmodel, rd, q)
    op_a = croco_mfd.DelassusMF(rmodel, rd, cms, 1e-4)
    op_a.compute()
    Ga = np.array(op_a.matrix())
    rd2 = rmodel.createData()
    pin.computeJointJacobians(rmodel, rd2, q)
    op_b = croco_mfd.DelassusMF(rmodel, rd2, cms, 1e-1)
    op_b.compute()
    Gb = np.array(op_b.matrix())
    d = np.diag(Gb - Ga)
    ok = np.allclose(d, 1e-1 - 1e-4, atol=1e-9) and \
        np.allclose(Gb - Ga - np.diag(d), 0, atol=1e-9)
    print(f"{'damping':14s} G(1e-1) - G(1e-4) = {d.mean():.4f} I   "
          f"{'ok' if ok else 'FAIL'}")
    if not ok:
        fails.append(("damping", 0, float(d.mean()), 0, 0))

    if fails:
        print(f"\n{len(fails)} FAILURES")
        return 1
    print("\nall pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
