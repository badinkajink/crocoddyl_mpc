#!/usr/bin/env python3
"""Assemble the S16 multithreading docpage from runs/2026-08-13_session16/.

Same contract as croco_speed_page.py: every number and figure comes out of the
run directory.  The before/after pair here is two PROCESSES rather than two
configurations -- which libcrocoddyl is mapped is decided by LD_PRELOAD at exec
time -- so every JSON records the library it was measured against and the page
prints it, because "which build is this" is not otherwise recoverable from a
table of milliseconds.

usage: croco_threads_page.py --dir runs/2026-08-13_session16 \
                             --body ../docs/lean/_body_threads_2026-08-13.html \
                             --out 2026-08-13_mpc_threads.html
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

MS = '<span style="text-transform:none">ms</span>'


def load(run_dir):
    d = {}
    for name in os.listdir(run_dir):
        if name.endswith(".json"):
            d[name[:-5]] = json.load(open(os.path.join(run_dir, name)))
    return d


# ------------------------------------------------------------------ fig 1 --
def fig_threads(d):
    """Step time against thread count, against ideal 1/n and the sequential floor.

    Two things have to be visible at once: the measured curve flattening, and WHY
    it flattens.  The backward pass cannot be parallelised, so it is drawn as a
    floor no thread count crosses, and ideal 1/n is drawn behind the measurement
    so the gap between them is the picture rather than a number in a caption.
    """
    rows = d["threads"]["threads"]
    seq = d.get("stage", {}).get("sequential_ms")
    W, H = 760, 320
    L, R, T, B = 58, 150, 22, 42
    xs = [max(r["effective"], 1) for r in rows]
    y1 = max(r["solve_ms"] for r in rows) * 1.10
    x1 = max(xs)

    def X(v):
        return L + (v - 1) / max(x1 - 1, 1) * (W - L - R)

    def Y(v):
        return H - B - v / y1 * (H - B - T)

    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="MPC step time '
           f'against thread count">']
    step = max(1, int(round(y1 / 6)))
    for gy in range(0, int(y1) + 1, step):
        out.append(f'<line x1="{L}" y1="{Y(gy):.1f}" x2="{W - R}" '
                   f'y2="{Y(gy):.1f}" stroke="var(--rule)" stroke-width="0.7"/>')
        out.append(f'<text x="{L - 7}" y="{Y(gy) + 3.5:.1f}" font-size="10" '
                   f'text-anchor="end" fill="var(--ink3)">{gy}</text>')
    for n in xs:
        out.append(f'<text x="{X(n):.1f}" y="{H - B + 15}" font-size="10" '
                   f'text-anchor="middle" fill="var(--ink3)">{n}</text>')
    out.append(f'<text x="{L - 7}" y="{T + 2}" font-size="10" '
               f'text-anchor="end" fill="var(--ink3)">ms</text>')
    out.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 8}" font-size="10.5" '
               f'text-anchor="middle" fill="var(--ink3)">threads</text>')

    period = d["meta"]["meta"]["control_period_ms"]
    if period < y1:
        out.append(f'<line x1="{L}" y1="{Y(period):.1f}" x2="{W - R}" '
                   f'y2="{Y(period):.1f}" stroke="var(--bad)" '
                   f'stroke-width="1.3" stroke-dasharray="4 3"/>')
        out.append(f'<text x="{W - R - 4:.0f}" y="{Y(period) - 6:.1f}" '
                   f'font-size="10" text-anchor="end" fill="var(--bad)">'
                   f'{period:.0f} ms control period</text>')

    base = rows[0]["solve_ms"]
    ideal = " ".join(f'{"M" if i == 0 else "L"}{X(n):.1f},{Y(base / n):.1f}'
                     for i, n in enumerate(xs))
    out.append(f'<path d="{ideal}" fill="none" stroke="var(--ink3)" '
               f'stroke-width="1.2" stroke-dasharray="3 3"/>')
    out.append(f'<text x="{X(xs[-1]) + 8:.1f}" y="{Y(base / xs[-1]) + 4:.1f}" '
               f'font-size="10.5" fill="var(--ink3)">ideal 1/n</text>')

    if seq:
        out.append(f'<line x1="{L}" y1="{Y(seq):.1f}" x2="{W - R}" '
                   f'y2="{Y(seq):.1f}" stroke="var(--s3)" stroke-width="1.2"/>')
        out.append(f'<text x="{W - R - 4:.0f}" y="{Y(seq) + 14:.1f}" '
                   f'font-size="10" text-anchor="end" fill="var(--s3)">'
                   f'sequential floor (backward pass + 1 rollout) {seq:.1f} ms</text>')

    path = " ".join(
        f'{"M" if i == 0 else "L"}{X(max(r["effective"], 1)):.1f},'
        f'{Y(r["solve_ms"]):.1f}' for i, r in enumerate(rows))
    out.append(f'<path d="{path}" fill="none" stroke="var(--s2)" '
               f'stroke-width="2.2"/>')
    for r in rows:
        out.append(f'<circle cx="{X(max(r["effective"], 1)):.1f}" '
                   f'cy="{Y(r["solve_ms"]):.1f}" r="3.4" fill="var(--s2)"/>')
    last = rows[-1]
    out.append(f'<text x="{X(max(last["effective"], 1)) + 8:.1f}" '
               f'y="{Y(last["solve_ms"]) + 4:.1f}" font-size="10.5" '
               f'fill="var(--s2)">measured</text>')
    out.append("</svg>")
    return "\n".join(out)


def table_threads(d):
    rows = d["threads"]["threads"]
    period = d["meta"]["meta"]["control_period_ms"]
    base = rows[0]["solve_ms"]
    out = ['<div class="scroll"><table><thead><tr><th>threads</th>'
           f'<th>step {MS}</th><th>p95 {MS}</th><th>Hz</th><th>speed-up</th>'
           '<th>efficiency</th><th>reach err mm</th><th>survives</th>'
           '</tr></thead><tbody>']
    for r in rows:
        n = max(r["effective"], 1)
        cls = "good" if r["p95_ms"] <= period else ""
        out.append(f'<tr><td>{n}</td>'
                   f'<td>{r["solve_ms"]:.2f}</td>'
                   f'<td class="{cls}">{r["p95_ms"]:.2f}</td>'
                   f'<td>{r["hz"]:.0f}</td>'
                   f'<td>{base / r["solve_ms"]:.2f}×</td>'
                   f'<td>{100 * (base / r["solve_ms"]) / n:.0f}%</td>'
                   f'<td>{r["reach_mm"]:.1f}</td>'
                   f'<td class="{"bad" if r["fell"] else "good"}">'
                   f'{"FELL" if r["fell"] else "upright"}</td></tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def table_stage(d):
    if "stage" not in d:
        return ""
    n = d["stage"]["nthreads"]
    out = ['<div class="scroll"><table><thead><tr><th>stage of one step</th>'
           f'<th>1 thread {MS}</th><th>{n} threads {MS}</th><th>speed-up</th>'
           '<th>parallel in crocoddyl?</th></tr></thead><tbody>']
    for r in d["stage"]["stage"]:
        out.append(f'<tr><td>{r["stage"]}</td><td>{r["t1_ms"]:.2f}</td>'
                   f'<td>{r["tn_ms"]:.2f}</td>'
                   f'<td>{r["t1_ms"] / max(r["tn_ms"], 1e-9):.2f}×</td>'
                   f'<td class="{"good" if r["parallel"] else "bad"}">'
                   f'{"yes" if r["parallel"] else "no — sequential"}</td>'
                   f'</tr>')
    out.append("</tbody></table></div>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(
        HERE, "runs", "2026-08-13_session16"))
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title",
                    default="Multithreading the crocoddyl MPC — S16")
    ap.add_argument("--style-from", default="2026-08-13_mpc_speed.html")
    a = ap.parse_args()

    d = load(a.dir)
    body = open(a.body).read()
    rows = d["threads"]["threads"]
    meta = d["meta"]["meta"]
    best = min(rows, key=lambda r: r["solve_ms"])
    base = rows[0]

    parts = {
        "FIG_THREADS": fig_threads(d),
        "TABLE_THREADS": table_threads(d),
        "TABLE_STAGE": table_stage(d),
        "T1_MS": f'{base["solve_ms"]:.1f}',
        "T1_P95": f'{base["p95_ms"]:.1f}',
        "BEST_N": f'{max(best["effective"], 1)}',
        "BEST_MS": f'{best["solve_ms"]:.1f}',
        "BEST_P95": f'{best["p95_ms"]:.1f}',
        "BEST_HZ": f'{best["hz"]:.0f}',
        "BEST_SPEEDUP": f'{base["solve_ms"] / best["solve_ms"]:.2f}',
        "BEST_EFF": (f'{100 * (base["solve_ms"] / best["solve_ms"]) / max(best["effective"], 1):.0f}'),
        "MAX_N": f'{max(max(r["effective"], 1) for r in rows)}',
        "PERIOD": f'{meta["control_period_ms"]:.0f}',
        "NCPU": f'{meta["n_cpu"]}',
        "CPU": meta["cpu_model"],
        "CRO_VERSION": meta["crocoddyl"],
        "PIN_VERSION": meta["pinocchio"],
        "COMMIT": meta["commit"][:9],
        "LIBPATH": meta.get("libcrocoddyl", "unknown"),
        "MT": "on" if meta.get("multithreading") else "off",
    }
    # The 35-node sweep is the DEPLOYED configuration; the 50-node one has more
    # thread points and is what the figure and table show.  They are different
    # measurements and the page must not quote one as the other.
    if "threads_h35" in d:
        h = d["threads_h35"]["threads"]
        hb = min(h, key=lambda r: r["solve_ms"])
        parts.update({
            "H35_N": f'{max(hb["effective"], 1)}',
            "H35_MS": f'{hb["solve_ms"]:.1f}',
            "H35_P95": f'{hb["p95_ms"]:.1f}',
            "H35_HZ": f'{hb["hz"]:.0f}',
            "H35_T1_MS": f'{h[0]["solve_ms"]:.1f}',
            "H35_T1_P95": f'{h[0]["p95_ms"]:.1f}',
            "H35_SPEEDUP": f'{h[0]["solve_ms"] / hb["solve_ms"]:.2f}',
        })
    if "stage" in d:
        parts["SEQ_MS"] = f'{d["stage"]["sequential_ms"]:.1f}'
        parts["STAGE_N"] = f'{d["stage"]["nthreads"]}'
        for r in d["stage"]["stage"]:
            if "calcDiff" in r["stage"]:
                parts["CD_1"] = f'{r["t1_ms"]:.2f}'
                parts["CD_N"] = f'{r["tn_ms"]:.2f}'
                parts["CD_SPEEDUP"] = f'{r["t1_ms"] / max(r["tn_ms"], 1e-9):.2f}'
            if "rollout" in r["stage"]:
                parts["RO_1"] = f'{r["t1_ms"]:.2f}'
                parts["RO_N"] = f'{r["tn_ms"]:.2f}'
            if "backwardPass" in r["stage"]:
                parts["BW_1"] = f'{r["t1_ms"]:.2f}'
                parts["BW_N"] = f'{r["tn_ms"]:.2f}'
    for k, v in parts.items():
        body = body.replace(f"<!--{k}-->", str(v))
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
