#!/usr/bin/env python3
"""Assemble the S15 MPC-speed docpage from runs/2026-08-13_session15/.

Same contract as croco_delassus_page.py: every number and every figure comes out
of the run directory, so the prose cannot drift from the measurement.  The page
reads several JSONs rather than one, because the "before" rows have to be
measured in a process where CROCO_KEEPOUT/CROCO_PASSIVE selected the old
implementations -- those are import-time choices, so the before/after pair is two
runs by construction (see croco_speed.LADDER).

usage: croco_speed_page.py --dir runs/2026-08-13_session15 \
                           --body ../docs/lean/_body_speed_2026-08-13.html \
                           --out 2026-08-13_mpc_speed.html
"""

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs", "lean")
sys.path.insert(0, DOCS)

from simple_page import style_from                      # noqa: E402

# `th` in the shared stylesheet is text-transform:uppercase, which turns "µs"
# into "MS" -- three orders of magnitude wrong.  Units in headers opt out.
US = '<span style="text-transform:none">µs</span>'
MS = '<span style="text-transform:none">ms</span>'


def load(run_dir):
    """Every JSON in the run directory, keyed by basename."""
    d = {}
    for name in os.listdir(run_dir):
        if name.endswith(".json"):
            d[name[:-5]] = json.load(open(os.path.join(run_dir, name)))
    return d


# ------------------------------------------------------------------ fig 1 --
def fig_ladder(d):
    """The step time down the ladder, against the control period.

    A horizontal bar per rung on a LINEAR axis, because the thing the reader has
    to see is a threshold crossing -- where the bar stops being longer than the
    20 ms period -- and a log axis makes a threshold look like a gentle slope.
    The first rung is 204 ms and would compress everything else to nothing, so
    it is drawn clipped with its value written in, which is honest about the
    scale break rather than quietly rescaling.
    """
    rows = d["ladder"]["ladder"]
    period = d["meta"]["meta"]["control_period_ms"]
    W, H = 760, 44 + 30 * len(rows)
    L, R, T = 232, 74, 26
    span = W - L - R
    # Full scale at 2x the period, so the crossing sits mid-plot and the rungs
    # that matter are legible; anything longer is clipped and labelled.
    hi = 2.4 * period
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="MPC step time '
           f'down the optimisation ladder">']
    x_period = L + span * period / hi
    out.append(f'<rect x="{x_period:.1f}" y="{T - 8}" width="{W - R - x_period:.1f}" '
               f'height="{H - T - 12}" fill="var(--bad)" fill-opacity="0.055"/>')
    out.append(f'<line x1="{x_period:.1f}" y1="{T - 8}" x2="{x_period:.1f}" '
               f'y2="{H - 20}" stroke="var(--bad)" stroke-width="1.3" '
               f'stroke-dasharray="4 3"/>')
    out.append(f'<text x="{x_period + 5:.1f}" y="{T - 12}" font-size="10.5" '
               f'fill="var(--bad)">20 ms control period</text>')
    for i, r in enumerate(rows):
        y = T + 8 + 30 * i
        w = span * min(r["solve_ms"], hi) / hi
        clipped = r["solve_ms"] > hi
        col = "var(--bad)" if r["solve_ms"] > period else "var(--good)"
        out.append(f'<text x="{L - 8}" y="{y + 12}" font-size="11" '
                   f'text-anchor="end" fill="var(--ink2)">{r["rung"]}</text>')
        out.append(f'<rect x="{L}" y="{y}" width="{w:.1f}" height="17" '
                   f'fill="{col}" fill-opacity="0.8"/>')
        if clipped:
            out.append(f'<path d="M{L + w - 10:.1f} {y} l8 8.5 l-8 8.5" '
                       f'fill="none" stroke="var(--bg)" stroke-width="2"/>')
        out.append(f'<text x="{L + w + 6:.1f}" y="{y + 12}" font-size="11" '
                   f'fill="var(--ink2)">{r["solve_ms"]:.1f} ms '
                   f'<tspan fill="var(--ink3)">({r["hz"]:.0f} Hz)</tspan></text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------ fig 2 --
def fig_scaling(d):
    """Node cost against the NUMBER of cost terms, for two kinds of term.

    The figure of the page: two straight lines with almost the same slope, one
    for a 3-row keep-out point and one for a 1-row control residual that shares
    none of its arithmetic.  If the cost of a term were the cost of what it
    computes, those slopes would differ by the ratio of their work; they differ
    by 0.6 µs, and the common ~1.6 µs is CostModelSum's dense accumulation.
    """
    sc = d["scaling"]["scaling"]
    W, H = 760, 300
    L, R, T, B = 56, 200, 20, 40
    xs = [p["n"] for p in sc["keepout"]["points"]]
    ys = [p["us"] for s in sc.values() for p in s["points"]]
    x1, y1 = max(xs), max(ys) * 1.08

    def X(v):
        return L + v / x1 * (W - L - R)

    def Y(v):
        return H - B - v / y1 * (H - B - T)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="node cost '
           f'against number of cost terms">']
    for gy in range(0, int(y1) + 1, 50):
        out.append(f'<line x1="{L}" y1="{Y(gy):.1f}" x2="{W - R}" '
                   f'y2="{Y(gy):.1f}" stroke="var(--rule)" stroke-width="0.7"/>')
        out.append(f'<text x="{L - 7}" y="{Y(gy) + 3.5:.1f}" font-size="10" '
                   f'text-anchor="end" fill="var(--ink3)">{gy}</text>')
    for gx in xs:
        out.append(f'<text x="{X(gx):.1f}" y="{H - B + 15}" font-size="10" '
                   f'text-anchor="middle" fill="var(--ink3)">{gx}</text>')
    out.append(f'<text x="{L - 7}" y="{T + 2}" font-size="10" '
               f'text-anchor="end" fill="var(--ink3)">µs</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" font-size="10.5" '
               f'text-anchor="middle" fill="var(--ink3)">'
               f'cost terms in one CostModelSum</text>')
    series = [("keepout", "3-row box keep-out point", "var(--s2)"),
              ("ctrl", "1-row control residual", "var(--s4)")]
    for key, label, col in series:
        pts = sc[key]["points"]
        path = " ".join(f'{"M" if i == 0 else "L"}{X(p["n"]):.1f},{Y(p["us"]):.1f}'
                        for i, p in enumerate(pts))
        out.append(f'<path d="{path}" fill="none" stroke="{col}" '
                   f'stroke-width="2"/>')
        for p in pts:
            out.append(f'<circle cx="{X(p["n"]):.1f}" cy="{Y(p["us"]):.1f}" '
                       f'r="3" fill="{col}"/>')
        last = pts[-1]
        out.append(f'<text x="{X(last["n"]) + 9:.1f}" y="{Y(last["us"]) + 4:.1f}" '
                   f'font-size="10.5" fill="{col}">{label}</text>')
        out.append(f'<text x="{X(last["n"]) + 9:.1f}" y="{Y(last["us"]) + 18:.1f}" '
                   f'font-size="10.5" fill="var(--ink3)">'
                   f'{sc[key]["per_term_us"]:.2f} µs per term</text>')
    y0 = sc["ctrl"]["intercept_us"]
    out.append(f'<line x1="{L}" y1="{Y(y0):.1f}" x2="{W - R}" y2="{Y(y0):.1f}" '
               f'stroke="var(--ink3)" stroke-width="1" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{W - R - 4:.0f}" y="{Y(y0) - 6:.1f}" font-size="10" '
               f'text-anchor="end" fill="var(--ink3)">'
               f'contact dynamics alone, {y0:.0f} µs</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------ fig 3 --
def fig_budget(d):
    """One braced node before and after, as two stacked bars to the same scale.

    Same scale is the whole point: the "after" bar has to be visibly shorter,
    not merely differently proportioned, and a per-bar normalisation would hide
    exactly the thing that changed.
    """
    def segs(terms):
        r = {x["node"]: x for x in terms["terms"]}["braced"]
        g = {x["group"]: x for x in r["groups"]}
        ko = g.get("keepout", {"us": 0.0, "n_terms": 0})
        cone = g.get("cones", {"us": 0.0})
        rest = r["full_us"] - r["bare_us"] - ko["us"] - cone["us"]
        return r, [(f'keep-out ({ko["n_terms"]} '
                    f'{"terms" if ko["n_terms"] > 1 else "term"})',
                    ko["us"], "var(--s2)"),
                   ("cones", cone["us"], "var(--s1)"),
                   ("other cost terms", max(rest, 0.0), "var(--s5)"),
                   ("contact dynamics", r["bare_us"], "var(--s4)")]

    before, sb = segs(d["terms_s13"])
    after, sa = segs(d["terms_s15"])
    total = before["full_us"]
    W, H, L, R = 760, 150, 96, 96
    span = W - L - R
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="one braced node '
           f'before and after, microseconds">']
    for row, (label, r, segments) in enumerate(
            (("S13", before, sb), ("S15", after, sa))):
        y = 20 + 46 * row
        x = L
        out.append(f'<text x="{L - 10}" y="{y + 20}" font-size="11.5" '
                   f'text-anchor="end" fill="var(--ink2)" '
                   f'font-weight="600">{label}</text>')
        out.append(f'<text x="{L - 10}" y="{y + 33}" font-size="9.5" '
                   f'text-anchor="end" fill="var(--ink3)">'
                   f'{r["n_terms"]} terms</text>')
        for name, us, col in segments:
            w = span * us / total
            out.append(f'<rect x="{x:.1f}" y="{y}" width="{max(w, 0.6):.1f}" '
                       f'height="30" fill="{col}" fill-opacity="0.85"/>')
            x += w
        out.append(f'<text x="{x + 8:.1f}" y="{y + 20}" font-size="11.5" '
                   f'fill="var(--ink2)">{r["full_us"]:.0f} µs</text>')
    for i, (name, us, col) in enumerate(sb):
        lx = L + (i % 2) * (span / 2)
        ly = 132 + (i // 2) * 15
        out.append(f'<rect x="{lx}" y="{ly - 8}" width="9" height="9" '
                   f'fill="{col}" fill-opacity="0.85"/>')
        out.append(f'<text x="{lx + 13}" y="{ly}" font-size="10" '
                   f'fill="var(--ink3)">{name.split(" (")[0]}</text>')
    out.append("</svg>")
    return "\n".join(out)


# ----------------------------------------------------------------- tables --
def table_ladder(d):
    """The ladder, with the repeat run's step time beside the first.

    Two runs and not one because the machine drifts: a full re-run of the ladder
    an hour later came back 5-10% slower ACROSS EVERY RUNG, ratios intact.  That
    is thermal/scheduling variance and not a regression, but a single column of
    milliseconds would invite reading a 12.2 as if it were repeatable to 0.1, and
    the p95 claim at the bottom rung has only ~15% of margin against the control
    period.  So both runs are shown.
    """
    period = d["meta"]["meta"]["control_period_ms"]
    rerun = {r["rung"]: r for r in d.get("ladder_rerun", {}).get("ladder", [])}
    out = ['<div class="scroll"><table><thead><tr><th>rung</th>'
           '<th>keep-out</th><th>actuation</th><th>H</th><th>iters</th>'
           f'<th>step {MS}</th><th>repeat {MS}</th><th>p95 {MS}</th><th>Hz</th>'
           '<th>reach err mm</th><th>survives</th></tr></thead><tbody>']
    for r in d["ladder"]["ladder"]:
        cls = "bad" if r["solve_ms"] > period else "good"
        out.append(f'<tr><td>{r["rung"]}</td>'
                   f'<td>{r["CROCO_KEEPOUT"]}</td>'
                   f'<td>{r["CROCO_PASSIVE"]}</td>'
                   f'<td>{r["horizon"]}</td><td>{r["iters"]}</td>'
                   f'<td class="{cls}">{r["solve_ms"]:.1f}</td>'
                   f'<td>{rerun[r["rung"]]["solve_ms"]:.1f}</td>'
                   f'<td>{r["p95_ms"]:.1f} / {rerun[r["rung"]]["p95_ms"]:.1f}</td>'
                   f'<td class="{cls}">{r["hz"]:.1f}</td>'
                   f'<td>{r["reach_mm"]:.1f}</td>'
                   f'<td>{"no — FELL" if r["fell"] else "yes"}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_terms(d, key, caption):
    out = [f'<div class="scroll"><table><thead><tr><th>node</th>'
           f'<th>terms</th><th>group</th><th>terms in group</th>'
           f'<th>{US}</th><th>{US} per term</th><th>share of node</th>'
           f'</tr></thead><tbody>']
    for r in d[key]["terms"]:
        for i, g in enumerate(r["groups"]):
            first = i == 0
            out.append(
                f'<tr>'
                f'<td>{r["node"] if first else ""}</td>'
                f'<td>{r["n_terms"] if first else ""}</td>'
                f'<td>{g["group"]}</td><td>{g["n_terms"]}</td>'
                f'<td>{g["us"]:.1f}</td><td>{g["per_term_us"]:.2f}</td>'
                f'<td>{100 * g["us"] / r["full_us"]:.1f}%</td></tr>')
        out.append(f'<tr><td></td><td></td><td><i>contact dynamics</i></td>'
                   f'<td>—</td><td>{r["bare_us"]:.1f}</td><td>—</td>'
                   f'<td>{100 * r["bare_us"] / r["full_us"]:.1f}%</td></tr>')
    out.append(f'</tbody></table></div><p class="meta">{caption}</p>')
    return "".join(out)


def table_offline(d):
    """The offline staged solve under each keep-out implementation."""
    out = ['<div class="scroll"><table><thead><tr><th>keep-out</th>'
           '<th>solve s</th><th>iters</th><th>cost</th>'
           '<th>reach err mm</th><th>elbow / forearm / palm landing mm</th>'
           '<th>|q − q*| rad</th></tr></thead><tbody>']
    for r in d["offline"]["offline"]:
        sites = " / ".join(f'{v:.2f}' for v in r["site_mm"].values())
        out.append(f'<tr><td>{r["keepout"]}</td>'
                   f'<td class="{"good" if r["keepout"] == "fused" else ""}">'
                   f'{r["seconds"]:.1f}</td>'
                   f'<td>{r["iters"]}</td><td>{r["cost"]:.4g}</td>'
                   f'<td>{r["reach_mm"]:.2f}</td><td>{sites}</td>'
                   f'<td>{r["q_err_rad"]:.3f}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_sweep(d):
    rows = sorted(d["sweep"]["sweep"], key=lambda r: r["solve_ms"])
    out = ['<div class="scroll"><table><thead><tr><th>H</th><th>iters</th>'
           f'<th>cones</th><th>step {MS}</th><th>Hz</th>'
           '<th>reach err mm</th><th>table pen. mm</th>'
           '<th>min CoM margin mm</th><th>survives</th></tr></thead><tbody>']
    for r in rows:
        out.append(f'<tr><td>{r["horizon"]}</td><td>{r["iters"]}</td>'
                   f'<td>{"on" if r["cones"] else "off"}</td>'
                   f'<td>{r["solve_ms"]:.1f}</td><td>{r["hz"]:.1f}</td>'
                   f'<td>{r["reach_mm"]:.1f}</td><td>{r["pen_mm"]:.1f}</td>'
                   f'<td>{r["margin_mm"]:+.1f}</td>'
                   f'<td class="{"bad" if r["fell"] else "good"}">'
                   f'{"FELL" if r["fell"] else "upright"}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_alphas(d):
    rows = sorted(d["sweep_alphas"]["sweep"],
                  key=lambda r: (-r["horizon"], -(r["alphas"] or 10)))
    out = ['<div class="scroll"><table><thead><tr><th>H</th>'
           f'<th>ladder rungs</th><th>step {MS}</th><th>p95 {MS}</th>'
           '<th>reach err mm</th><th>survives</th></tr></thead><tbody>']
    for r in rows:
        out.append(f'<tr><td>{r["horizon"]}</td><td>{r["alphas"] or 10}</td>'
                   f'<td>{r["solve_ms"]:.1f}</td><td>{r["p95_ms"]:.1f}</td>'
                   f'<td>{r["reach_mm"]:.1f}</td>'
                   f'<td class="{"bad" if r["fell"] else "good"}">'
                   f'{"FELL" if r["fell"] else "upright"}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_noise(d):
    """Fall rate per configuration over the noise seeds."""
    cells = {}
    for name, blob in d.items():
        if not name.startswith("sweep_noise"):
            continue
        for r in blob["sweep"]:
            k = (r["horizon"], r["iters"])
            cells.setdefault(k, []).append(r)
    out = ['<div class="scroll"><table><thead><tr><th>H</th><th>iters</th>'
           '<th>seeds</th><th>fell</th><th>reach err mm, survivors</th>'
           '</tr></thead><tbody>']
    for k in sorted(cells, reverse=True):
        rs = cells[k]
        up = [r["reach_mm"] for r in rs if not r["fell"]]
        n_fell = sum(r["fell"] for r in rs)
        out.append(f'<tr><td>{k[0]}</td><td>{k[1]}</td><td>{len(rs)}</td>'
                   f'<td class="bad">{n_fell} / {len(rs)}</td>'
                   f'<td>{", ".join(f"{v:.1f}" for v in sorted(up)) or "—"}'
                   f'</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        HERE, "runs", "2026-08-13_session15"))
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title",
                    default="Making the crocoddyl MPC real-time: the cost stack "
                            "was the wrong axis — S15")
    ap.add_argument("--style-from", default="2026-08-06_delassus.html")
    a = ap.parse_args()

    d = load(a.dir)
    body = open(a.body).read()
    lad = {r["rung"].split(":")[0]: r for r in d["ladder"]["ladder"]}
    meta = d["meta"]["meta"]
    meta13 = d["meta_s13"]["meta"]
    sc = d["scaling"]["scaling"]
    t13 = {r["node"]: r for r in d["terms_s13"]["terms"]}
    t15 = {r["node"]: r for r in d["terms_s15"]["terms"]}
    st13, st15 = d["step_s13"]["step"], d["step_s15"]["step"]
    pc, sv = d["pieces"]["pieces"], d["solver"]["solver"]

    parts = {
        "FIG_LADDER": fig_ladder(d),
        "FIG_SCALING": fig_scaling(d),
        "FIG_BUDGET": fig_budget(d),
        "TABLE_LADDER": table_ladder(d),
        "TABLE_TERMS_S13": table_terms(
            d, "terms_s13",
            "The S13 cost stack, per group, on the real action models. "
            "A group is switched off with changeCostStatus and the node "
            "re-timed, so its number includes its share of the CostModelSum "
            "accumulation."),
        "TABLE_TERMS_S15": table_terms(
            d, "terms_s15",
            "The same measurement after the keep-out is one term. Nothing about "
            "the objective changed; 85 cost terms did."),
        "TABLE_OFFLINE": table_offline(d),
        "TABLE_SWEEP": table_sweep(d),
        "TABLE_ALPHAS": table_alphas(d),
        "TABLE_NOISE": table_noise(d),
    }
    parts.update({
        "OFF_PY": f'{[r for r in d["offline"]["offline"] if r["keepout"] == "python"][0]["seconds"]:.0f}',
        "OFF_CPP": f'{[r for r in d["offline"]["offline"] if r["keepout"] == "cpp"][0]["seconds"]:.1f}',
        "OFF_FUSED": f'{[r for r in d["offline"]["offline"] if r["keepout"] == "fused"][0]["seconds"]:.1f}',
        "STEP_S12": f'{lad["S12"]["solve_ms"]:.0f}',
        "STEP_S13": f'{lad["S13"]["solve_ms"]:.1f}',
        "HZ_S13": f'{lad["S13"]["hz"]:.0f}',
        "STEP_FUSED": f'{lad["S15a"]["solve_ms"]:.1f}',
        "STEP_FINAL": f'{lad["S15d"]["solve_ms"]:.1f}',
        "P95_FINAL_RERUN": f'{d["ladder_rerun"]["ladder"][-1]["p95_ms"]:.1f}',
        "MODE": json.load(open(os.path.join(a.dir, "croco", "plan_s13.json")))["mode"],
        "P95_FINAL": f'{lad["S15d"]["p95_ms"]:.1f}',
        "HZ_FINAL": f'{lad["S15d"]["hz"]:.0f}',
        "HZ_H50": f'{lad["S15c"]["hz"]:.0f}',
        "STEP_2ITER": f'{lad["S15b"]["solve_ms"]:.1f}',
        "STEP_H50": f'{lad["S15c"]["solve_ms"]:.1f}',
        "P95_H50": f'{lad["S15c"]["p95_ms"]:.1f}',
        "P95_H50_OVER": f'{lad["S15c"]["p95_ms"] - meta["control_period_ms"]:.1f}',
        "PEN_FINAL": f'{abs([r for r in d["sweep"]["sweep"] if r["horizon"] == 35 and r["iters"] == 1 and r["cones"]][0]["pen_mm"]):.1f}',
        "PEN_H50": f'{abs([r for r in d["sweep"]["sweep"] if r["horizon"] == 50 and r["iters"] == 1 and r["cones"]][0]["pen_mm"]):.1f}',
        "REACH_FINAL": f'{lad["S15d"]["reach_mm"]:.1f}',
        "SPEEDUP_S13": f'{lad["S13"]["solve_ms"] / lad["S15d"]["solve_ms"]:.1f}',
        "SPEEDUP_S12": f'{lad["S12"]["solve_ms"] / lad["S15d"]["solve_ms"]:.1f}',
        "PERIOD": f'{meta["control_period_ms"]:.0f}',
        "PER_TERM_KO": f'{sc["keepout"]["per_term_us"]:.2f}',
        "PER_TERM_CTRL": f'{sc["ctrl"]["per_term_us"]:.2f}',
        "ACC_BYTES": f'{meta["per_term_bytes"] // 8:,}',
        "N_POINTS": f'{meta["n_keepout_points"]}',
        "TERMS_S13": f'{meta13["cost_terms_max"]}',
        "TERMS_S15": f'{meta["cost_terms_max"]}',
        "KO_US_S13": f'{[g for g in t13["braced"]["groups"] if g["group"] == "keepout"][0]["us"]:.0f}',
        "KO_PER_TERM_S13": f'{[g for g in t13["braced"]["groups"] if g["group"] == "keepout"][0]["per_term_us"]:.2f}',
        "CONTAINER_SHARE": f'{100 * sc["ctrl"]["per_term_us"] / sc["keepout"]["per_term_us"]:.0f}',
        "KO_SHARE_S13": f'{100 * [g for g in t13["braced"]["groups"] if g["group"] == "keepout"][0]["us"] / t13["braced"]["full_us"]:.0f}',
        "NODE_S13": f'{t13["braced"]["full_us"]:.0f}',
        "NODE_S15": f'{t15["braced"]["full_us"]:.0f}',
        "DYN_S15": f'{t15["braced"]["bare_us"]:.0f}',
        "DYN_SHARE_S15": f'{100 * t15["braced"]["bare_us"] / t15["braced"]["full_us"]:.0f}',
        "CONE_US": f'{[g for g in t15["braced"]["groups"] if g["group"] == "cones"][0]["us"]:.0f}',
        "CONE_SHARE": f'{100 * [g for g in t15["braced"]["groups"] if g["group"] == "cones"][0]["us"] / t15["braced"]["full_us"]:.0f}',
        "INSITU_S13": f'{st13["problem_calcDiff"] / st13["horizon"]:.0f}',
        "INSITU_S15": f'{st15["problem_calcDiff"] / st15["horizon"]:.0f}',
        "SWEEP_S13": f'{st13["problem_calcDiff"] / 1000:.1f}',
        "SWEEP_S15": f'{st15["problem_calcDiff"] / 1000:.1f}',
        "MB_S13": f'{meta13["costdata_mb_per_50_horizon"]:.0f}',
        "MB_S15": f'{meta["costdata_mb_per_50_horizon"]:.0f}',
        "ALLOC_S13": f'{meta13["createData_ms"]:.0f}',
        "ALLOC_S15": f'{meta["createData_ms"]:.0f}',
        "ACT_PY_CALC": f'{pc["actuation_py_calc_us"]:.2f}',
        "ACT_PY_DIFF": f'{pc["actuation_py_calcDiff_us"]:.2f}',
        "ACT_CPP_CALC": f'{pc["actuation_cpp_calc_us"]:.2f}',
        "ACT_CPP_DIFF": f'{pc["actuation_cpp_calcDiff_us"]:.2f}',
        "ACT_STOCK": f'{pc["stock_actuation_calcDiff_us"]:.2f}',
        "NODE_NOCONE": f'{pc["node_nocones_us"]:.0f}',
        "NODE_NOFORCE": f'{pc["node_nocones_noforce_us"]:.0f}',
        "SOLVER_CALCDIFF": f'{sv["calcDiff_us"] / 1000:.1f}',
        "SOLVER_BACKWARD": f'{sv["backwardPass_us"] / 1000:.1f}',
        "SOLVER_ROLLOUT": f'{sv["tryStep_us"] / 1000:.2f}',
        "TRIALS": f'{lad["S15c"]["trials_median"]:.0f}',
        "CRO_VERSION": meta["crocoddyl"],
        "PIN_VERSION": meta["pinocchio"],
        "CPU": meta["cpu_model"],
        "NCPU": f'{meta["n_cpu"]}',
        "COMMIT": meta["commit"][:9],
        "NV": f'{meta["nv"]}',
        "NU": f'{meta["nu"]}',
        "NDX": f'{meta["ndx"]}',
    })
    for k, v in parts.items():
        body = body.replace(f"<!--{k}-->", v)
    left = sorted(set(re.findall(r"<!--([A-Z_0-9]+)-->", body)))
    if left:
        print("  WARNING: unfilled placeholders", left)

    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{a.title}</title>\n'
            + style_from(os.path.join(DOCS, a.style_from))
            + '\n</head>\n<body><div class="wrap">\n\n' + body
            + '\n</div></body>\n</html>\n')
    dst = os.path.join(DOCS, a.out)
    open(dst, "w").write(html)
    print(f"wrote {dst} ({len(html)} bytes)")


if __name__ == "__main__":
    main()
