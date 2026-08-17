#!/usr/bin/env python3
"""Assemble the matrix-free Delassus docpage from delassus.json.

Every number and both figures come out of the run directory, so the page cannot
drift from the measurement. Inline SVG rather than PNGs, and the stylesheet is
lifted from an existing page rather than copied, for the same reasons as
simple_page.py -- see its docstring.

usage: croco_delassus_page.py --json runs/2026-08-06_session14/delassus.json \
                              --body ../docs/lean/_body_delassus_2026-08-06.html \
                              --out 2026-08-06_delassus.html
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs", "lean")
sys.path.insert(0, DOCS)

from simple_page import style_from                      # noqa: E402

# `th` in the shared stylesheet is text-transform:uppercase, which turns "µs"
# into "MS" -- i.e. into milliseconds, three orders of magnitude wrong.  Any
# unit that appears in a header has to opt out of the transform.
US = '<span style="text-transform:none">µs</span>'
# SVG has no <sub>; a shifted tspan is the equivalent.
SUB = '<tspan baseline-shift="sub" font-size="8">c</tspan>'


# ------------------------------------------------------------------ fig 1 --
def fig_budget(d):
    """Where one braced node's 342 us goes, as a single stacked bar.

    The point of the picture is proportion, not magnitude: the slice a
    matrix-free operator could touch has to be visible next to the slice that
    dominates, and a table of microseconds does not make that comparison for
    the reader.
    """
    st = {r["node"]: r for r in d["stages"]}["braced"]
    pr = {r["node"]: r for r in d["nodes"]}["braced"]

    total = pr["calc_full"] + pr["calcdiff_full"]
    dela = (st["fwd_delassus_build"] + st["fwd_delassus_llt"] +
            st["fwd_delassus_solve1"] + st["kkt_delassus_solve_nc"])
    minv_gemm = st["kkt_minv_dense"] + st["kkt_gemms"] + \
        st["fwd_cholesky_decompose"] + st["fwd_minv_solve1"]
    costs = pr["calc_costs"] + pr["calcdiff_costs"]
    other = total - dela - minv_gemm - costs

    segs = [("cost stack (86 keep-out points, cones, limits)", costs, "var(--s2)"),
            ("kinematics, RNEA derivatives, actuation, contacts", other, "var(--s4)"),
            ("M⁻¹ block + Schur GEMMs", minv_gemm, "var(--s1)"),
            ("Delassus", dela, "var(--s3)")]

    W, H, PAD, BAR, TOP = 760, 128, 8, 46, 30
    x, out = PAD, [f'<svg viewBox="0 0 {W} {H}" role="img" '
                   f'aria-label="one braced node, {total:.0f} microseconds, '
                   f'by stage">']
    span = W - 2 * PAD
    for label, us, col in segs:
        w = span * us / total
        out.append(f'<rect x="{x:.1f}" y="{TOP}" width="{max(w, 0.7):.1f}" '
                   f'height="{BAR}" fill="{col}" fill-opacity="0.85"/>')
        if w > 44:
            out.append(f'<text x="{x + w / 2:.1f}" y="{TOP + BAR / 2 + 4:.0f}" '
                       f'font-size="11" text-anchor="middle" fill="var(--bg)" '
                       f'font-weight="600">{100 * us / total:.0f}%</text>')
        x += w
    out.append(f'<text x="{PAD}" y="14" font-size="11" fill="var(--ink3)">'
               f'one braced node, calc + calcDiff = {total:.0f} µs</text>')

    # Callout on the Delassus slice: at 2% of the bar there is no room to label
    # it in place, and it is the one segment the reader came for.  Leader line
    # up and to the left so the text stays inside the viewBox.
    xd = PAD + span * (total - dela / 2) / total
    out.append(f'<line x1="{xd:.1f}" y1="{TOP}" x2="{xd:.1f}" y2="{TOP - 9}" '
               f'stroke="var(--s3)" stroke-width="1.2"/>')
    out.append(f'<text x="{xd - 5:.1f}" y="{TOP - 12}" font-size="12" '
               f'text-anchor="end" fill="var(--s3)" font-weight="600">'
               f'Delassus: {dela:.1f} µs ({100 * dela / total:.1f}%)</text>')

    # Legend: two fixed columns rather than a width heuristic, because the
    # heuristic overlapped as soon as a label changed length.
    for i, (label, us, col) in enumerate(segs):
        lx = PAD + (i % 2) * (span / 2)
        ly = TOP + BAR + 22 + (i // 2) * 17
        out.append(f'<rect x="{lx}" y="{ly - 8}" width="9" height="9" '
                   f'fill="{col}" fill-opacity="0.85"/>')
        out.append(f'<text x="{lx + 14}" y="{ly}" font-size="10.5" '
                   f'fill="var(--ink3)">{label} — {us:.0f} µs</text>')
    out.append("</svg>")
    return "\n".join(out)


# ------------------------------------------------------------------ fig 2 --
def fig_crossover(d):
    """Per-node cost of both routes against constraint count, log-log.

    Log-log because the two curves have different exponents -- that is the
    whole content of the figure -- and a linear axis would show one flat line
    and one that leaves the page.
    """
    import math
    rows = d["sweep"]
    W, H = 760, 330
    L, R, T, B = 58, 250, 22, 42
    xs = [r["nc"] for r in rows]
    ys = [v for r in rows for v in (r["mf_total"], r["explicit_total"])]
    lx0, lx1 = math.log10(min(xs) * 0.85), math.log10(max(xs) * 1.15)
    ly0, ly1 = math.log10(min(ys) * 0.6), math.log10(max(ys) * 1.6)

    def X(v):
        return L + (math.log10(v) - lx0) / (lx1 - lx0) * (W - L - R)

    def Y(v):
        return H - B - (math.log10(v) - ly0) / (ly1 - ly0) * (H - B - T)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="matrix-free vs '
           f'explicit Delassus per node against constraint count">']
    for v in (1, 10, 100, 1000, 10000, 100000):
        if not (ly0 <= math.log10(v) <= ly1):
            continue
        out.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" '
                   f'stroke="var(--line)" stroke-width="1"/>')
        lbl = f"{v} µs" if v < 1000 else f"{v // 1000} ms"
        out.append(f'<text x="{L - 6}" y="{Y(v) + 3.5:.1f}" font-size="10" '
                   f'text-anchor="end" fill="var(--ink3)">{lbl}</text>')
    for v in (21, 50, 100, 250, 500, 1000):
        if not (lx0 <= math.log10(v) <= lx1):
            continue
        out.append(f'<text x="{X(v):.1f}" y="{H - B + 15}" font-size="10" '
                   f'text-anchor="middle" fill="var(--ink3)">{v}</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" font-size="10.5" '
               f'text-anchor="middle" fill="var(--ink3)">constraint rows '
               f'n<tspan baseline-shift="sub" font-size="8">c</tspan></text>')

    for key, col, name in (("explicit_total", "var(--s1)", "explicit (crocoddyl's)"),
                           ("mf_total", "var(--s3)", "matrix-free")):
        pts = " ".join(f"{X(r['nc']):.1f},{Y(r[key]):.1f}" for r in rows)
        out.append(f'<polyline fill="none" stroke="{col}" stroke-width="2.2" '
                   f'points="{pts}"/>')
        for r in rows:
            out.append(f'<circle cx="{X(r["nc"]):.1f}" cy="{Y(r[key]):.1f}" '
                       f'r="2.6" fill="{col}"/>')
        last = rows[-1]
        out.append(f'<text x="{X(last["nc"]) + 8:.1f}" y="{Y(last[key]) + 4:.1f}" '
                   f'font-size="11.5" fill="{col}" font-weight="600">{name}</text>')

    # The crossing, bracketed by the two rows it falls between.
    lo = max((r for r in rows if r["ratio"] > 1), key=lambda r: r["nc"])
    hi = min((r for r in rows if r["ratio"] < 1), key=lambda r: r["nc"])
    t = (1 - lo["ratio"]) / (hi["ratio"] - lo["ratio"])
    nstar = lo["nc"] + t * (hi["nc"] - lo["nc"])
    out.append(f'<line x1="{X(nstar):.1f}" y1="{T}" x2="{X(nstar):.1f}" '
               f'y2="{H - B}" stroke="var(--ink3)" stroke-width="1" '
               f'stroke-dasharray="3 3"/>')
    out.append(f'<text x="{X(nstar) + 5:.1f}" y="{T + 12}" font-size="10.5" '
               f'fill="var(--ink3)">crossover, n{SUB} ≈ {nstar:.0f}</text>')
    n0 = rows[0]["nc"]
    out.append(f'<line x1="{X(n0):.1f}" y1="{T}" x2="{X(n0):.1f}" y2="{H - B}" '
               f'stroke="var(--s2)" stroke-width="1.4"/>')
    out.append(f'<text x="{X(n0) + 5:.1f}" y="{H - B - 14}" font-size="10.5" '
               f'fill="var(--s2)" font-weight="600">this robot, braced '
               f'(n{SUB} = {n0})</text>')
    out.append("</svg>")
    return "\n".join(out), nstar


# ---------------------------------------------------------------- tables --
def cell(v, fmt="{:.2f}", bad=None, good=None):
    cls = ""
    if bad is not None and bad(v):
        cls = ' class="bad"'
    elif good is not None and good(v):
        cls = ' class="good"'
    return f"<td{cls}>{fmt.format(v)}</td>"


def table_stages(d):
    rows = {r["node"]: r for r in d["stages"]}
    head = ("<tr><th>stage, " + US + "</th>"
            + "".join(f"<th>{n} (n<sub>c</sub>={rows[n]['nc']})</th>"
                      for n in ("approach", "braced"))
            + "<th>Delassus?</th></tr>")
    body = []
    for key, lbl, dela in (
            ("fwd_total", "<b>calc</b> — pinocchio::forwardDynamics", None),
            ("fwd_cholesky_decompose", "&nbsp;&nbsp;LTDL factor of M", False),
            ("fwd_delassus_build", "&nbsp;&nbsp;build J M⁻¹Jᵀ", True),
            ("fwd_delassus_llt", "&nbsp;&nbsp;dense LLT of it", True),
            ("fwd_delassus_solve1", "&nbsp;&nbsp;one G⁻¹ solve (the forces)", True),
            ("fwd_minv_solve1", "&nbsp;&nbsp;one M⁻¹ solve", False),
            ("kkt_total", "<b>calcDiff</b> — getKKTContactDynamicMatrixInverse", None),
            ("kkt_delassus_solve_nc", "&nbsp;&nbsp;G⁻¹ on n<sub>c</sub> columns", True),
            ("kkt_minv_dense", "&nbsp;&nbsp;M⁻¹ as a dense n<sub>v</sub>×n<sub>v</sub> block", False),
            ("kkt_gemms", "&nbsp;&nbsp;three Schur GEMMs", False)):
        mark = "" if dela is None else ("<b>yes</b>" if dela else "no")
        body.append(f"<tr><td>{lbl}</td>"
                    + "".join(f"<td>{rows[n][key]:.2f}</td>"
                              for n in ("approach", "braced"))
                    + f"<td>{mark}</td></tr>")
    tot = []
    for n in ("approach", "braced"):
        r = rows[n]
        tot.append(r["delassus_total"])
    body.append("<tr><td><b>Delassus-attributable</b></td>"
                + "".join(f'<td class="good"><b>{v:.2f}</b></td>' for v in tot)
                + "<td></td></tr>")
    return ('<div class="scroll"><table><thead>' + head +
            "</thead><tbody>" + "".join(body) + "</tbody></table></div>")


def table_operator(d):
    rows = {r["node"]: r for r in d["operator"]}
    out = ['<div class="scroll"><table><thead><tr><th>' + US + '</th>']
    for n in ("approach", "braced"):
        out.append(f'<th>{n} matrix-free</th><th>{n} explicit</th>')
    out.append("</tr></thead><tbody>")
    for lbl, ka, kb in (
            ("setup, per configuration", "mf_setup", "explicit_setup"),
            ("one G x", ("mf", "apply1"), ("explicit", "apply1")),
            ("one G⁻¹ x", ("mf", "solve1"), ("explicit", "solve1"))):
        out.append(f"<tr><td>{lbl}</td>")
        for n in ("approach", "braced"):
            r = rows[n]
            a = r[ka] if isinstance(ka, str) else r[ka[0]][ka[1]]
            b = r[kb] if isinstance(kb, str) else r[kb[0]][kb[1]]
            out.append(f"<td>{a:.2f}</td><td>{b:.2f}</td>")
        out.append("</tr>")
    out.append("<tr><td><b>setup + the solves crocoddyl asks for</b></td>")
    for n in ("approach", "braced"):
        r = rows[n]
        out.append(f'<td class="bad"><b>{r["mf_at_croco_pattern"]:.1f}</b></td>'
                   f'<td class="good"><b>{r["explicit_at_croco_pattern"]:.1f}</b></td>')
    out.append("</tr><tr><td>break-even, in G⁻¹ solves</td>")
    for n in ("approach", "braced"):
        r = rows[n]
        k = r["breakeven_solves"]
        txt = "never" if k < 0 else f"{k:.0f}"
        out.append(f'<td colspan="2">{txt} '
                   f'(crocoddyl asks for {r["croco_solves_per_node"]})</td>')
    out.append("</tr></tbody></table></div>")
    return "".join(out)


def table_dropin(d):
    out = ['<div class="scroll"><table><thead><tr><th>node</th><th>block</th>'
           '<th>right-hand sides</th>'
           f'<th>crocoddyl {US}</th><th>matrix-free {US}</th>'
           '<th>ratio</th><th>agreement</th></tr></thead><tbody>']
    for r in d["dropin"]:
        out.append(f'<tr><td>{r["node"]}</td><td>{r["what"]}</td>'
                   f'<td>{r["n_rhs"]}</td><td>{r["route_a"]:.1f}</td>'
                   f'<td>{r["route_b"]:.1f}</td>'
                   f'<td class="bad">{r["ratio"]:.2f}×</td>'
                   f'<td>{r["rel_diff"]:.0e}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_sweep(d):
    out = ['<div class="scroll"><table><thead><tr>'
           '<th>n<sub>c</sub></th>'
           f'<th>setup mf {US}</th><th>setup explicit {US}</th>'
           f'<th>one solve mf {US}</th><th>one solve explicit {US}</th>'
           f'<th>per node mf {US}</th><th>per node explicit {US}</th>'
           '<th>mf / explicit</th>'
           '<th>memory mf</th><th>memory explicit</th>'
           '<th>|G−G<sub>ref</sub>|</th></tr></thead><tbody>']
    for r in d["sweep"]:
        kb = lambda b: (f"{b / 1024:.1f} KB" if b < 1 << 20
                        else f"{b / (1 << 20):.1f} MB")
        cls = ' class="good"' if r["ratio"] < 1 else ' class="bad"'
        star = ('&nbsp;<span style="color:var(--s2)">← this robot</span>'
                if r["extra"] == 0 else "")
        out.append(f'<tr><td style="white-space:nowrap">{r["nc"]}{star}</td><td>{r["mf_setup"]:.1f}</td>'
                   f'<td>{r["explicit_setup"]:.1f}</td>'
                   f'<td>{r["mf_solve"]:.2f}</td>'
                   f'<td>{r["explicit_solve"]:.2f}</td>'
                   f'<td>{r["mf_total"]:.0f}</td>'
                   f'<td>{r["explicit_total"]:.0f}</td>'
                   f'<td{cls}>{r["ratio"]:.2f}×</td>'
                   f'<td>{kb(r["mf_bytes"])}</td><td>{kb(r["explicit_bytes"])}</td>'
                   f'<td>{r["rel_err"]:.0e}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_profile(d):
    out = ['<div class="scroll"><table><thead><tr><th>node</th>'
           '<th>n<sub>c</sub></th><th>cost terms</th>'
           f'<th>calc full {US}</th><th>calc dynamics {US}</th>'
           f'<th>calcDiff full {US}</th><th>calcDiff dynamics {US}</th>'
           f'<th>calcDiff cost stack {US}</th>'
           '<th>dynamics share</th></tr></thead><tbody>']
    for r in d["nodes"]:
        out.append(f'<tr><td>{r["node"]}</td><td>{r["nc"]}</td>'
                   f'<td>{r["n_costs"]}</td>'
                   f'<td>{r["calc_full"]:.1f}</td><td>{r["calc_dyn"]:.1f}</td>'
                   f'<td>{r["calcdiff_full"]:.1f}</td>'
                   f'<td>{r["calcdiff_dyn"]:.1f}</td>'
                   f'<td class="bad">{r["calcdiff_costs"]:.1f}</td>'
                   f'<td>{100 * r["dyn_share"]:.0f}%</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title",
                    default="The matrix-free Delassus, measured inside a "
                            "crocoddyl MPC — S14")
    ap.add_argument("--style-from", default="2026-08-06_croco_loop.html")
    a = ap.parse_args()

    d = json.load(open(a.json))
    body = open(a.body).read()
    fig2, nstar = fig_crossover(d)
    parts = {
        "FIG_BUDGET": fig_budget(d),
        "FIG_CROSSOVER": fig2,
        "TABLE_PROFILE": table_profile(d),
        "TABLE_STAGES": table_stages(d),
        "TABLE_OPERATOR": table_operator(d),
        "TABLE_DROPIN": table_dropin(d),
        "TABLE_SWEEP": table_sweep(d),
    }
    # Scalars the prose quotes, so the prose cannot drift from the JSON either.
    st = {r["node"]: r for r in d["stages"]}
    op = {r["node"]: r for r in d["operator"]}
    pr = {r["node"]: r for r in d["nodes"]}
    bg = d["budget"]
    worst = max(d["dropin"], key=lambda r: r["ratio"])
    best_agree = max(r["rel_diff"] for r in d["dropin"])
    parts.update({
        "N_STAR": f"{nstar:.0f}",
        "MPC_STEP": f"{bg['mpc_step_ms']:.1f}",
        "MPC_FLOOR": f"{bg['braced']['floor_ms']:.1f}",
        "MPC_SAVING": f"{bg['braced']['saving_ms']:.1f}",
        # Two different accountings, both honest, and the page has to say which
        # is which: DELA_SHARE_NODE is one calc plus one calcDiff (what the
        # figure draws), DELA_SHARE counts calc twice for a line-search rollout
        # (the upper bound §8 uses to bound the whole step).
        "DELA_SHARE_NODE": f"{100 * st['braced']['delassus_total'] / (pr['braced']['calc_full'] + pr['braced']['calcdiff_full']):.1f}",
        "DELA_SHARE": f"{100 * bg['braced']['share_of_node']:.1f}",
        "DELA_US": f"{st['braced']['delassus_total']:.1f}",
        "STAGE_SHARE": f"{100 * st['braced']['delassus_share_of_stages']:.0f}",
        "ERR_BRACED": f"{op['braced']['rel_err']:.1e}",
        "ERR_APPROACH": f"{op['approach']['rel_err']:.1e}",
        "SETUP_MF": f"{op['braced']['mf_setup']:.2f}",
        "SETUP_EX": f"{op['braced']['explicit_setup']:.2f}",
        "SOLVE_MF": f"{op['braced']['mf']['solve1']:.2f}",
        "SOLVE_EX": f"{op['braced']['explicit']['solve1']:.2f}",
        "BREAKEVEN": f"{op['braced']['breakeven_solves']:.0f}",
        "CROCO_SOLVES": f"{op['braced']['croco_solves_per_node']}",
        "WORST_RATIO": f"{worst['ratio']:.1f}",
        "DROPIN_AGREE": f"{best_agree:.0e}",
        "COSTS_US": f"{pr['braced']['calcdiff_costs']:.0f}",
        "NODE_US": f"{pr['braced']['calc_full'] + pr['braced']['calcdiff_full']:.0f}",
        "COST_SHARE": f"{100 * (1 - pr['braced']['dyn_share']):.0f}",
        "MEM_RATIO": f"{d['sweep'][-1]['explicit_bytes'] / d['sweep'][-1]['mf_bytes']:.0f}",
        "MEM_NC": f"{d['sweep'][-1]['nc']}",
        "SWEEP_ERR": f"{max(r['rel_err'] for r in d['sweep']):.1e}",
        "PIN_VERSION": d["meta"]["pinocchio"],
        "CRO_VERSION": d["meta"]["crocoddyl"],
    })
    for k, v in parts.items():
        body = body.replace(f"<!--{k}-->", v)
    import re
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
