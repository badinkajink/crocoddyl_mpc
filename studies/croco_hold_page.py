#!/usr/bin/env python3
"""Figures for the braced-hold docpage, straight off twin telemetry dumps.

WHY SVG BY HAND. Every other figure in this study is matplotlib-through-a-PNG,
which is fine on a page nobody reads in dark mode. These are line plots of five
traces against time -- the one case where hand-written SVG is both shorter and
strictly better, because the strokes can use the page's own CSS variables and
follow the reader's theme instead of baking one background into a raster.

Reads `*_telemetry_*.json` written by croco_twin (the same records the live
panel plots) and emits standalone <svg> elements to stdout or a file.
"""
import argparse
import glob
import json
import os

import numpy as np

W, H = 720, 300
PAD_L, PAD_R, PAD_T, PAD_B = 52, 12, 14, 34
STROKES = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)",
           "var(--warn)"]


def load(path):
    d = json.load(open(path))
    rows = d["rows"]
    t = np.array([r["t"] for r in rows], float)
    return d, rows, t


def series(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], float)


def svg(traces, ylab, title, t_max=None, y0=None, baseline=None):
    """traces: [(label, t, y, stroke)] -> one standalone <svg> string."""
    tm = t_max or max(tr[1].max() for tr in traces)
    ys = np.concatenate([tr[2][np.isfinite(tr[2])] for tr in traces])
    lo, hi = (float(np.min(ys)) if y0 is None else y0), float(np.max(ys))
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def X(v):
        return PAD_L + (W - PAD_L - PAD_R) * (v / tm)

    def Y(v):
        return PAD_T + (H - PAD_T - PAD_B) * (1.0 - (v - lo) / (hi - lo))

    out = ['<svg viewBox="0 0 %d %d" role="img" aria-label="%s" '
           'style="width:100%%;height:auto;font:11px ui-monospace,monospace">'
           % (W, H, title)]
    # grid + y labels
    for i in range(5):
        v = lo + (hi - lo) * i / 4.0
        y = Y(v)
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                   'stroke="var(--line)" stroke-width="1"/>'
                   % (PAD_L, y, W - PAD_R, y))
        out.append('<text x="%d" y="%.1f" text-anchor="end" fill="var(--ink3)">'
                   '%g</text>' % (PAD_L - 6, y + 3.5, round(v, 1)))
    for s in range(0, int(tm) + 1, max(1, int(tm // 8))):
        out.append('<text x="%.1f" y="%d" text-anchor="middle" '
                   'fill="var(--ink3)">%d</text>' % (X(s), H - PAD_B + 16, s))
    if baseline is not None:
        out.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" '
                   'stroke="var(--ink3)" stroke-width="1" '
                   'stroke-dasharray="4 3"/>'
                   % (PAD_L, Y(baseline), W - PAD_R, Y(baseline)))
    for i, (label, t, y, stroke) in enumerate(traces):
        ok = np.isfinite(y)
        # decimate: 2000 points per trace is 40 kB of path data nobody can see
        idx = np.linspace(0, ok.sum() - 1, min(500, ok.sum())).astype(int)
        pts = " ".join("%.1f,%.1f" % (X(a), Y(b))
                       for a, b in zip(t[ok][idx], y[ok][idx]))
        out.append('<polyline fill="none" stroke="%s" stroke-width="1.8" '
                   'stroke-linejoin="round" points="%s"/>' % (stroke, pts))
        out.append('<text x="%.1f" y="%.1f" fill="%s">%s</text>'
                   % (X(t[ok][idx][-1]) - 4, Y(y[ok][idx][-1]) - 6, stroke,
                      label) if False else "")
    # legend
    lx = PAD_L + 6
    for i, (label, _, _, stroke) in enumerate(traces):
        out.append('<rect x="%d" y="%d" width="10" height="3" fill="%s"/>'
                   % (lx, PAD_T + 4, stroke))
        out.append('<text x="%d" y="%d" fill="var(--ink2)">%s</text>'
                   % (lx + 14, PAD_T + 9, label))
        lx += 16 + 7.2 * len(label) + 12
    out.append('<text x="6" y="%d" fill="var(--ink3)" '
               'transform="rotate(-90 6 %d)" text-anchor="middle">%s</text>'
               % (H // 2, H // 2, ylab))
    out.append('<text x="%d" y="%d" text-anchor="middle" fill="var(--ink3)">'
               'time [s]</text>' % ((W + PAD_L) // 2, H - 4))
    out.append("</svg>")
    return "".join(out)


def find(run_dir, stem):
    g = sorted(glob.glob(os.path.join(run_dir, "%s_telemetry_*.json" % stem)))
    if not g:
        raise SystemExit("no telemetry for %s in %s" % (stem, run_dir))
    return g[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir holding *_telemetry_*.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    parts = []

    # -- Kp ladder: pelvis sink --------------------------------------------
    kps = [("kp000", "Kp = 0"), ("kp010", "Kp = 10"), ("kp030", "Kp = 30"),
           ("kp050", "Kp = 50"), ("kp100", "Kp = 100")]
    tr = []
    for i, (stem, lab) in enumerate(kps):
        _, rows, t = load(find(args.dir, stem))
        y = series(rows, "pelvis_drop_mm")
        tr.append((lab, t, y - y[np.argmax(t >= 4.0)], STROKES[i]))
    parts.append(("kp_sink", svg(tr, "pelvis sink [mm]",
                                 "pelvis sink against time, per contact Kp",
                                 baseline=0.0)))

    # -- Kp ladder: brace drift --------------------------------------------
    tr = []
    for i, (stem, lab) in enumerate(kps):
        _, rows, t = load(find(args.dir, stem))
        y = series(rows, "drift_elbow_mm")
        tr.append((lab, t, y, STROKES[i]))
    parts.append(("kp_drift", svg(tr, "elbow drift [mm]",
                                  "brace drift against time, per contact Kp")))

    # -- modes at Kp=50 ----------------------------------------------------
    modes = [("mode_elbow", "elbow"), ("kp050", "elbow+forearm"),
             ("mode_elbow_palm", "elbow+palm"),
             ("mode_elbow_wrist", "elbow+wrist"),
             ("mode_forearm_wrist", "forearm+wrist")]
    tr = []
    for i, (stem, lab) in enumerate(modes):
        _, rows, t = load(find(args.dir, stem))
        y = series(rows, "pelvis_drop_mm")
        tr.append((lab, t, y - y[np.argmax(t >= 4.0)], STROKES[i]))
    parts.append(("mode_sink", svg(tr, "pelvis sink [mm]",
                                   "pelvis sink against time, per contact mode",
                                   baseline=0.0)))

    text = "\n\n".join('<!-- %s -->\n%s' % (n, s) for n, s in parts)
    if args.out:
        open(args.out, "w").write(text)
        print("wrote %s (%d figures)" % (args.out, len(parts)))
    else:
        print(text)


if __name__ == "__main__":
    main()
