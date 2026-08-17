#!/usr/bin/env python3
"""Build regions.json: the static-equilibrium CoM regions plus LABELLED contacts.

The S10 version stored `pts` as an unlabelled point cloud, so every contact drew
as the same grey dot and the plot could show that the region grew without showing
WHICH contact grew it.  This records, per contact: the label (foot / elbow /
forearm / palm / hip / torso), the world point MuJoCo's narrowphase actually
resolved, the contact normal, and the normal force the min-effort QP puts through
it -- so a dot can be coloured by identity and sized by load.

usage: make_regions.py [x y z] [set,set,...]
       TAU_BASIS=... SITE_SET=v1|v2 MU=... STANCE_DY=...
"""
import json
import os
import sys

import numpy as np

import contact_select as cs
import stability as st

DEFAULT_SETS = [(), ("palm",), ("elbow", "forearm"),
                ("elbow", "forearm", "palm"), ("elbow", "forearm", "hip")]


def one(target, subset):
    m, d = cs.load()
    ik = cs.solve_ik(m, d, np.array(target), subset)
    qp = cs.equilibrium_qp(m, d, subset)
    contacts, P, A, g, mass, com, Rs = st.blocks(m, d, subset)

    # labels follow resolved_contacts' own order: 8 foot corners, then the
    # selected sites in `subset` order.  Keeping them in lockstep here rather
    # than re-deriving from body ids means a change to that ordering shows up as
    # a wrong label, not as a silently mislabelled plot.
    labels = ["foot"] * (len(P) - len(subset)) + list(subset)
    pts = []
    for i, (lab, p, R) in enumerate(zip(labels, P, Rs)):
        pts.append(dict(label=lab, p=[float(p[0]), float(p[1]), float(p[2])],
                        n=[float(v) for v in R[0]],
                        fn=float(qp["lam"][3 * i]),
                        ft=float(np.linalg.norm(qp["lam"][3 * i + 1:3 * i + 3]))))

    pa, c0, ma = st.equilibrium_region(m, d, subset, actuated=True)
    pc, _, mc = st.equilibrium_region(m, d, subset, actuated=False)
    slips = st.slip_margins(m, d, subset, qp["lam"])
    return dict(actuated=pa.tolist(), contact=pc.tolist(),
                com=[float(c0[0]), float(c0[1])],
                margin_a=float(ma), margin_c=float(mc),
                pts=pts,
                feasible=bool(qp["feasible"]), peak=float(qp["max_ratio"]),
                reach=float(ik["reach"]),
                placed=bool(ik.get("all_placed", True)),
                slip={k: float(v["utilisation"]) for k, v in slips.items()},
                table_x=[float(v) for v in cs.table_x_range(m, d)],
                table_y=[float(v) for v in cs.table_y_range(m, d)])


def main():
    tgt = ([float(a) for a in sys.argv[1:4]] if len(sys.argv) >= 4
           else [1.10, -0.2348, 1.0982])
    sets = ([tuple(s.split("+")) if s != "legs" else () for s in sys.argv[4].split(",")]
            if len(sys.argv) > 4 else DEFAULT_SETS)

    out = {"_meta": dict(target=tgt, basis=cs.TAU_BASIS, mu=cs.MU,
                         site_set=cs.SITE_SET, stance_dy=cs.STANCE_DY,
                         stance_dx=cs.STANCE_DX,
                         model=os.path.basename(cs.MODEL))}
    for s in sets:
        name = "+".join(s) or "legs"
        r = one(tgt, s)
        out[name] = r
        print("%-24s margin %.4f (contact %.4f)  feas=%s placed=%s  peak %.3f"
              % (name, r["margin_a"], r["margin_c"], r["feasible"],
                 r["placed"], r["peak"]), flush=True)

    dst = os.environ.get("OUT", "regions.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
