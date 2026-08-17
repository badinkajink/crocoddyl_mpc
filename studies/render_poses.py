#!/usr/bin/env python3
"""Render the IK lean pose for each contact subset, side + front view.

Sanity check on contact_select.py: the QP numbers mean nothing if the IK is
producing a pose that is folded through the table or self-intersecting.
"""
import itertools
import sys

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import contact_select as cs

W, H = 640, 480


def shot(m, d, cam):
    r = mujoco.Renderer(m, H, W)
    r.update_scene(d, camera=cam)
    return r.render()


def make_cam(m, d, azim, elev=-12, dist=2.6):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = [0.6, 0.0, 0.95]
    c.distance = dist
    c.azimuth = azim
    c.elevation = elev
    return c


def main(target):
    subsets = [s for k in range(4)
               for s in itertools.combinations(("elbow", "forearm", "palm"), k)]
    fig, axes = plt.subplots(4, 4, figsize=(17, 13),
                             gridspec_kw=dict(wspace=0.02, hspace=0.12))
    for i, subset in enumerate(subsets):
        m, d = cs.load()
        P = cs.solve_ik(m, d, np.asarray(target), subset)
        ik, pen = P['reach'], P['penetration']
        r = cs.equilibrium_qp(m, d, subset)
        row, col = divmod(i, 2)
        for k, azim in enumerate((90, 150)):        # side, three-quarter
            ax = axes[row][col * 2 + k]
            ax.imshow(shot(m, d, make_cam(m, d, azim)))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        name = " + ".join(subset) if subset else "legs only"
        ok = P["ok"] and (r["base_residual"] < 1.0) and (r["max_ratio"] <= 1.0)
        axes[row][col * 2].set_title(
            f"{name}   |   reach err {ik*1000:.0f} mm   base res {r['base_residual']:.1f} N   "
            f"max |τ|/limit {r['max_ratio']:.2f}   pen {pen*1000:.0f} mm   "
            f"{'FEASIBLE' if ok else 'infeasible'}",
            loc="left", fontsize=10.5, color=("#1baf7a" if ok else "#e34948"),
            pad=6)
        print(f"{name:28s} ik={ik:.4f} pen={pen*1000:5.1f}mm base_res={r['base_residual']:7.2f} "
              f"ratio={r['max_ratio']:.2f} brace={r['brace_force']:6.1f}")
    fig.suptitle(f"IK lean poses per contact subset — reach target {target}",
                 fontsize=13, y=0.985)
    out = f"poses_{target[0]:.2f}_{target[1]:.2f}_{target[2]:.2f}.png"
    fig.savefig(out, dpi=105, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    t = [float(x) for x in sys.argv[1:4]] if len(sys.argv) >= 4 else [1.20, 0.15, 0.90]
    main(t)
