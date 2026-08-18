#!/usr/bin/env python3
"""Aggregate a Lean Simple matrix run into the numbers and figures a page needs.

Reads the per-cell CSVs (not just matrix.json, so it can plot series), and writes

    report_data.json     per-cell aggregates, seeds pooled
    table_results.html   the results table as an HTML fragment
    fig_reach.svg        reach-to-target vs time, braced vs unbraced, far target
    fig_gaps.svg         the three seat gaps vs time for one braced rollout
    fig_modes.svg        contact duty per link and reach gain, one row per cell

The SVGs use `currentColor` and the same `--s*` palette variables as the other
docs/lean pages, so they invert with the page theme.

usage: simple_metrics.py --run DIR [--out DIR]
"""
import argparse
import glob
import json
import os

import numpy as np

import simple_lean as S

PAL = {"elbow": "var(--s1)", "forearm": "var(--s2)", "palm": "var(--s3)",
       "trunk": "var(--s5)", "none": "var(--s4)"}


class Fig:
    def __init__(self, w=760, h=250, pad=(46, 14, 26, 12)):
        self.w, self.h = w, h
        self.l, self.r, self.b, self.t = pad
        self.parts = []

    def px(self, x, x0, x1):
        return self.l + (self.w - self.l - self.r) * (x - x0) / max(1e-9, x1 - x0)

    def py(self, y, y0, y1):
        return (self.h - self.b) - (self.h - self.b - self.t) * \
            (y - y0) / max(1e-9, y1 - y0)

    def grid(self, y0, y1, ticks, fmt="%.2f"):
        for v in ticks:
            y = self.py(v, y0, y1)
            self.parts.append(
                '<line x1="%.0f" y1="%.1f" x2="%.0f" y2="%.1f" '
                'stroke="currentColor" stroke-opacity=".14"/>'
                % (self.l, y, self.w - self.r, y))
            self.parts.append(
                '<text x="%.0f" y="%.1f" font-size="11" fill="currentColor" '
                'fill-opacity=".55" text-anchor="end">%s</text>'
                % (self.l - 6, y + 4, fmt % v))

    def line(self, xs, ys, x0, x1, y0, y1, color, width=1.6, opacity=1.0,
             dash=None):
        pts = " ".join("%.1f,%.1f" % (self.px(x, x0, x1), self.py(y, y0, y1))
                       for x, y in zip(xs, ys))
        self.parts.append(
            '<polyline fill="none" stroke="%s" stroke-width="%.1f" '
            'stroke-opacity="%.2f"%s points="%s"/>'
            % (color, width, opacity,
               ' stroke-dasharray="%s"' % dash if dash else "", pts))

    def hline(self, v, x0, x1, y0, y1, color, label=None, dash="5 4"):
        y = self.py(v, y0, y1)
        self.parts.append(
            '<line x1="%.0f" y1="%.1f" x2="%.0f" y2="%.1f" stroke="%s" '
            'stroke-width="1.3" stroke-dasharray="%s" stroke-opacity=".8"/>'
            % (self.l, y, self.w - self.r, y, color, dash))
        if label:
            self.parts.append(
                '<text x="%.0f" y="%.1f" font-size="10.5" fill="%s" '
                'fill-opacity=".85">%s</text>'
                % (self.l + 6, y - 5, color, label))

    def text(self, x, y, s, size=10.5, anchor="start", opacity=.75):
        self.parts.append(
            '<text x="%.1f" y="%.1f" font-size="%.1f" fill="currentColor" '
            'fill-opacity="%.2f" text-anchor="%s">%s</text>'
            % (x, y, size, opacity, anchor, s))

    def xaxis(self, x0, x1, ticks, label=""):
        for v in ticks:
            self.text(self.px(v, x0, x1), self.h - 8, "%g" % v, anchor="middle",
                      opacity=.55)
        if label:
            self.text(self.w - self.r, self.h - 8, label, anchor="end",
                      opacity=.45)

    def svg(self, title):
        return ('<svg viewBox="0 0 %d %d" role="img" aria-label="%s">\n%s\n</svg>'
                % (self.w, self.h, title, "\n".join(self.parts)))


def cell_of(name):
    return name.rsplit("_s", 1)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or a.run

    per_run, series = {}, {}
    for p in sorted(glob.glob(os.path.join(a.run, "*.csv"))):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            o, ser, con = S.score(p)
        except Exception as e:                              # noqa: BLE001
            print("  skip %s: %s" % (name, e))
            continue
        if isinstance(o, dict) and "error" in o:
            print("  skip %s: %s" % (name, o["error"]))
            continue
        per_run[name], series[name] = o, (ser, con)
        print("%-26s got=%-18s gain=%+.3f fell=%-5s trunk=%+.3f churn=%4.0f "
              "tau=%.2f" % (name, "+".join(o["achieved_mode"]) or "none",
                            o["reach_gain"], o["fell"], o["trunk_gap_min"],
                            o["churn"] or -1, o["peak_tau_ratio"]))

    cells = {}
    for name, o in per_run.items():
        cells.setdefault(cell_of(name), []).append((name, o))
    agg = {}
    for c, runs in cells.items():
        g = np.array([r["reach_gain"] for _, r in runs])
        agg[c] = dict(
            n=len(runs),
            achieved_per_seed=["+".join(r["achieved_mode"]) or "none"
                               for _, r in runs],
            duty={k: float(np.mean([r["duty"][k] for _, r in runs]))
                  for k in ("elbow", "forearm", "palm", "trunk")},
            gain_mean=float(g.mean()), gain_min=float(g.min()),
            gain_max=float(g.max()),
            reach_base=float(np.mean([r["reach_base"] for _, r in runs])),
            reach_settled=float(np.mean([r["reach_settled"] for _, r in runs])),
            falls=int(sum(r["fell"] for _, r in runs)),
            trunk_min=float(np.min([r["trunk_gap_min"] for _, r in runs])),
            churn=float(np.mean([r["churn"] or 0 for _, r in runs])),
            peak_tau=float(np.max([r["peak_tau_ratio"] for _, r in runs])),
            sat_frac=float(np.mean([r["saturated_frac"] for _, r in runs])),
            pitch=float(np.mean([r["torso_pitch_settled"] for _, r in runs])),
            pelvis_x=float(np.mean([r["pelvis_x_settled"] for _, r in runs])),
            hand_x=float(np.mean([r["hand_x_settled"] for _, r in runs])),
            t_end=float(np.min([r["t_end"] for _, r in runs])),
        )
    with open(os.path.join(out, "report_data.json"), "w") as f:
        json.dump(dict(cells=agg, runs=per_run), f, indent=1)

    # ---- fig 1: reach vs time at the far target, braced vs unbraced -------- #
    f = Fig()
    y0, y1 = 0.0, 0.7
    x0, x1 = 0.0, max((series[n][0]["t"][-1] for n in series), default=18.0)
    f.grid(y0, y1, [0.0, 0.2, 0.4, 0.6])
    base = None
    for name in sorted(series):
        c = cell_of(name)
        if c not in ("far_brace", "far_none"):
            continue
        ser = series[name][0]
        f.line(ser["t"], ser["reach"], x0, x1, y0, y1,
               PAL["elbow"] if c == "far_brace" else PAL["none"], 1.6, .9)
        base = per_run[name]["reach_base"]
    if base:
        f.hline(base, x0, x1, y0, y1, "currentColor",
                "standing at t = 0  (%.3f m)" % base)
    f.xaxis(x0, x1, [0, 3, 6, 9, 12, 15, 18], "time [s]")
    f.text(f.l + 6, 26, "distance from the reaching hand to the target [m]",
           size=11, opacity=.6)
    open(os.path.join(out, "fig_reach.svg"), "w").write(
        f.svg("Distance from the reaching hand to the far target over time, "
              "braced versus unbraced, two seeds each."))

    # ---- fig 2: seat gaps for one braced rollout --------------------------- #
    pick = next((n for n in sorted(series) if cell_of(n) == "far_brace"), None)
    if pick:
        ser = series[pick][0]
        f = Fig()
        y0, y1 = -0.05, 0.45
        x0, x1 = 0.0, ser["t"][-1]
        f.parts.append(
            '<rect x="%.0f" y="%.1f" width="%.0f" height="%.1f" '
            'fill="currentColor" fill-opacity=".07"/>'
            % (f.l, f.py(0.005, y0, y1), f.w - f.l - f.r,
               f.py(-0.05, y0, y1) - f.py(0.005, y0, y1)))
        f.grid(y0, y1, [0.0, 0.1, 0.2, 0.3, 0.4])
        for k in ("elbow", "forearm", "palm"):
            f.line(ser["t"], ser["gap_%s" % k], x0, x1, y0, y1, PAL[k], 1.7)
        f.line(ser["t"], ser["reach"], x0, x1, y0, y1, "currentColor", 1.2, .5,
               dash="4 3")
        f.xaxis(x0, x1, [0, 3, 6, 9, 12, 15, 18], "time [s]")
        f.text(f.l + 6, 26,
               "seat gap [m], shaded band = in contact; dashed = reach error",
               size=11, opacity=.6)
        open(os.path.join(out, "fig_gaps.svg"), "w").write(
            f.svg("Seat gap of the elbow, forearm and palm over one braced "
                  "rollout, with the reach error."))

    # ---- fig 3: requested vs achieved, per cell ---------------------------- #
    order = [c for c in ["near_none", "near_brace", "far_none", "far_brace",
                         "far_forearm", "far_palm", "far_all",
                         "far_seatdepth", "far_brace_nosupport",
                         "far_brace_samples", "far_brace_horizon",
                         "vfar_none", "vfar_brace"] if c in agg]
    rowh = 24
    f = Fig(w=760, h=44 + rowh * len(order), pad=(196, 14, 20, 22))
    for j, k in enumerate(("elbow", "forearm", "palm", "trunk")):
        f.text(f.l + j * 106, 18, k, size=10.5, opacity=.55)
    f.text(f.w - 14, 18, "reach gain", size=10.5, opacity=.55, anchor="end")
    for i, c in enumerate(order):
        y = 36 + i * rowh
        f.text(4, y + 4, c, size=11, opacity=.85)
        for j, k in enumerate(("elbow", "forearm", "palm", "trunk")):
            duty = agg[c]["duty"][k]
            f.parts.append(
                '<rect x="%.0f" y="%.1f" width="%.1f" height="11" rx="2" '
                'fill="%s" fill-opacity="%.2f"/>'
                % (f.l + j * 106, y - 5, 88 * max(duty, 0.014), PAL[k],
                   .92 if duty > .5 else .32))
        f.text(f.w - 14, y + 4, "%+.3f m" % agg[c]["gain_mean"], size=11,
               opacity=.85, anchor="end")
    open(os.path.join(out, "fig_modes.svg"), "w").write(
        f.svg("Contact duty per link over the settled window, and mean reach "
              "gain, for every experiment cell."))

    # ---- results table ----------------------------------------------------- #
    rows_html = []
    for c in order:
        d = agg[c]
        rows_html.append(
            "<tr><td>%s</td><td>%s</td><td>%.2f</td><td>%.2f</td><td>%.2f</td>"
            "<td%s>%.2f</td><td>%+.3f</td><td>%d</td><td>%.0f</td></tr>"
            % (c, " / ".join(d["achieved_per_seed"]),
               d["duty"]["elbow"], d["duty"]["forearm"], d["duty"]["palm"],
               ' class="bad"' if d["duty"]["trunk"] > 0.05 else "",
               d["duty"]["trunk"], d["gain_mean"], d["falls"], d["churn"]))
    open(os.path.join(out, "table_results.html"), "w").write(
        '<div class="scroll"><table>\n<tr><th>cell</th>'
        '<th>achieved (per seed)</th><th>elbow</th><th>forearm</th>'
        '<th>palm</th><th>trunk</th><th>reach gain</th><th>falls</th>'
        '<th>churn</th></tr>\n%s\n</table></div>\n' % "\n".join(rows_html))

    print("wrote report_data.json, table_results.html, fig_reach.svg, "
          "fig_gaps.svg, fig_modes.svg into", out)


if __name__ == "__main__":
    main()
