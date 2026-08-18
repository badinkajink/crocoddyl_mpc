#!/usr/bin/env python3
"""Emit the reach-vs-brace SVG for the S12 docpage.

The one picture that settles what the chain rollout did: distance from the
REACHING hand to the target, and from the BRACING forearm to the same target,
against time, with the phase boundaries marked and the t=0 standing distance
drawn as a baseline.  Written as inline SVG -- no library, no network, and it
inherits the page's CSS variables so it themes with everything else.

usage: reach_figure.py TRAJ.csv > fragment.svg
"""
import csv
import sys

import mujoco
import numpy as np

sys.path.insert(0, "/home/humanoid/Programs/mjpc_icra2026/studies")
import contact_select as cs

W, H = 760, 250
L, R, TP, BT = 46, 14, 12, 30
YMAX = 0.85
PHASES = [(3, "lean"), (19, "reach"), (27, "release"),
          (41, "r1"), (46, "r2"), (51, "r3"), (56, "stand")]


def main(path, stride=250):
    m, d = cs.load(ik_margin=0)
    nid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    tgt = np.array([float(v) for v in
                    m.numeric_data[m.numeric_adr[nid]:m.numeric_adr[nid] + 3]])
    lines = [l for l in open(path) if not l.startswith("#")]
    rd = csv.reader(lines)
    hdr = next(rd)
    col = {n: i for i, n in enumerate(hdr)}
    rows = np.array([[float(v) for v in r] for r in rd])
    qi = [col["qpos%d" % i] for i in range(m.nq)]

    T, A, B = [], [], []
    for k in range(0, len(rows), stride):
        d.qpos[:] = rows[k][qi]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        T.append(float(rows[k][col["time"]]))
        A.append(float(np.linalg.norm(
            tgt - cs.point_world(m, d, "right_wrist_yaw_link", cs.REACH_OFF))))
        B.append(float(np.linalg.norm(
            tgt - cs.point_world(m, d, *cs.SITES["forearm"]))))

    t0, t1 = T[0], T[-1]
    sx = lambda t: L + (t - t0) / (t1 - t0) * (W - L - R)
    sy = lambda v: TP + (1 - min(v, YMAX) / YMAX) * (H - TP - BT)
    poly = lambda Y: " ".join("%.1f,%.1f" % (sx(t), sy(v)) for t, v in zip(T, Y))

    o = []
    o.append('<svg viewBox="0 0 %d %d" role="img" aria-label="Distance from the '
             'reaching hand and from the bracing forearm to the reach target, '
             'over the 66 s chain rollout.">' % (W, H))
    for v in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = sy(v)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="currentColor" '
                 'stroke-opacity=".14"/>' % (L, y, W - R, y))
        o.append('<text x="%d" y="%.1f" font-size="11" fill="currentColor" '
                 'fill-opacity=".55" text-anchor="end">%.1f</text>'
                 % (L - 6, y + 4, v))
    for t, lbl in PHASES:
        x = sx(t)
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%.1f" stroke="currentColor" '
                 'stroke-opacity=".22" stroke-dasharray="3 3"/>' % (x, TP, x, H - BT))
        o.append('<text x="%.1f" y="%d" font-size="10" fill="currentColor" '
                 'fill-opacity=".5">%s</text>' % (x + 3, TP + 10, lbl))
    for t in range(0, int(t1) + 1, 10):
        o.append('<text x="%.1f" y="%d" font-size="11" fill="currentColor" '
                 'fill-opacity=".55" text-anchor="middle">%d</text>'
                 % (sx(t), H - 12, t))
    o.append('<text x="%.1f" y="%d" font-size="11" fill="currentColor" '
             'fill-opacity=".55" text-anchor="middle">time [s]</text>'
             % ((L + W - R) / 2, H - 1))
    # the baseline: where the reaching hand already was, standing, at t = 0
    y0 = sy(A[0])
    o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="var(--warn)" '
             'stroke-width="1.5" stroke-dasharray="6 4"/>' % (L, y0, W - R, y0))
    o.append('<text x="%d" y="%.1f" font-size="11" fill="var(--warn)">'
             'standing at t = 0 (%.3f m)</text>' % (L + 6, y0 - 6, A[0]))
    o.append('<polyline fill="none" stroke="var(--s2)" stroke-width="2" '
             'points="%s"/>' % poly(B))
    o.append('<polyline fill="none" stroke="var(--s1)" stroke-width="2" '
             'points="%s"/>' % poly(A))
    o.append("</svg>")
    return "\n".join(o)


if __name__ == "__main__":
    print(main(sys.argv[1]))
