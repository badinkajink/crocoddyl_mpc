#!/usr/bin/env python3
"""Assemble the S17 robustness docpage from runs/2026-08-14_session17/.

Same contract as croco_speed_page.py and croco_threads_page.py: every number
and every figure in the page comes out of the run directory, so the page cannot
drift from the measurement without the file changing under it.

usage: croco_robust_page.py --dir runs/2026-08-14_session17 \\
                            --body ../docs/lean/_body_robust_2026-08-14.html \\
                            --out 2026-08-14_robustness.html
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs", "lean")
sys.path.insert(0, DOCS)

MS = '<span style="text-transform:none">ms</span>'


def load(run_dir):
    d = {}
    for name in os.listdir(run_dir):
        if name.endswith(".json"):
            try:
                d[name[:-5]] = json.load(open(os.path.join(run_dir, name)))
            except Exception:                                  # noqa: BLE001
                pass
    return d


def pct(r):
    return f"{100 * r:.0f}%"


def cls(r):
    return "good" if r >= 0.9 else ("bad" if r < 0.5 else "")


# ------------------------------------------------------------------ fig 1 --
def fig_fall(d):
    """Where the CoM goes, on the seed that falls and the seed that does not.

    The whole diagnosis is one picture: the failing run's CoM leaves the back of
    the foot polygon 0.4 s in and never comes back, and the brace it was going
    to lean on does not arrive until 2.4 s.  The polygon's rear edge is drawn
    because that is the line the argument is about, and the brace-force trace is
    underneath so the timing is visible rather than asserted.
    """
    tr = d.get("trace", {}).get("trace")
    if not tr:
        return ""
    sup = tr["support"]
    W, H = 760, 330
    L, R, T, B = 62, 120, 18, 78
    runs = [("s15_fall", "var(--bad)",
             f"S15, seed {tr['s15_fall']['seed']}"),
            ("s15_ok", "var(--s3)", f"S15, seed {tr['s15_ok']['seed']}"),
            ("fixed_was_fall", "var(--good)",
             f"fixed, seed {tr['fixed_was_fall']['seed']}")]
    tmax = max(max(tr[k]["t"]) for k, _, _ in runs if k in tr)
    lo, hi = sup["lo"][0], sup["hi"][0]
    ylo, yhi = min(lo - 0.09, -0.02), hi + 0.05

    def X(v):
        return L + v / tmax * (W - L - R)

    def Y(v):
        return H - B - (v - ylo) / (yhi - ylo) * (H - B - T)

    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="centre of mass '
         f'against time for a surviving and a falling seed">']
    # the support polygon as a band
    o.append(f'<rect x="{L}" y="{Y(hi):.1f}" width="{W - L - R}" '
             f'height="{Y(lo) - Y(hi):.1f}" fill="var(--panel)"/>')
    for v, lab in ((hi, "toe"), (lo, "heel")):
        o.append(f'<line x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" '
                 f'stroke="var(--rule)" stroke-width="1"/>')
        o.append(f'<text x="{W - R + 6}" y="{Y(v) + 3.5:.1f}" font-size="10" '
                 f'fill="var(--ink3)">{lab} {v:.3f}</text>')
    o.append(f'<text x="{L - 8}" y="{T + 8}" font-size="10" text-anchor="end" '
             f'fill="var(--ink3)">CoM x [m]</text>')
    for gt in range(0, int(tmax) + 1):
        o.append(f'<line x1="{X(gt):.1f}" y1="{T}" x2="{X(gt):.1f}" '
                 f'y2="{H - B}" stroke="var(--rule)" stroke-width="0.6"/>')
        o.append(f'<text x="{X(gt):.1f}" y="{H - B + 14}" font-size="10" '
                 f'text-anchor="middle" fill="var(--ink3)">{gt}</text>')
    o.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - B + 30}" font-size="10.5"'
             f' text-anchor="middle" fill="var(--ink3)">time [s]</text>')
    for gy in [round(ylo + i * (yhi - ylo) / 4, 2) for i in range(5)]:
        o.append(f'<text x="{L - 8}" y="{Y(gy) + 3.5:.1f}" font-size="10" '
                 f'text-anchor="end" fill="var(--ink3)">{gy:.2f}</text>')
    for key, col, lab in runs:
        if key not in tr:
            continue
        r = tr[key]
        p = " ".join(f'{"M" if i == 0 else "L"}{X(t):.1f},{Y(c[0]):.1f}'
                     for i, (t, c) in enumerate(zip(r["t"], r["com"])))
        o.append(f'<path d="{p}" fill="none" stroke="{col}" stroke-width="2"/>')
        o.append(f'<text x="{W - R + 6}" y="{Y(r["com"][-1][0]) + 3.5:.1f}" '
                 f'font-size="10.5" fill="{col}">{lab}</text>')
    # brace force, as a strip along the bottom, on its own scale
    r = tr.get("fixed_was_fall")
    if r:
        fmax = max(max(r["brace"]), 1.0)
        y0, hgt = H - 44, 30
        p = " ".join(f'{"M" if i == 0 else "L"}{X(t):.1f},'
                     f'{y0 + hgt - f / fmax * hgt:.1f}'
                     for i, (t, f) in enumerate(zip(r["t"], r["brace"])))
        o.append(f'<path d="{p}" fill="none" stroke="var(--s2)" '
                 f'stroke-width="1.4"/>')
        o.append(f'<text x="{W - R + 6}" y="{y0 + 12}" font-size="10" '
                 f'fill="var(--s2)">brace {fmax:.0f} N</text>')
    o.append("</svg>")
    return "\n".join(o)


# ------------------------------------------------------------------ fig 2 --
def fig_grid(d):
    """The table in plan view, one target per marker, coloured by survival.

    Two panels because the stance offset is the variable being argued about and
    a table of 25 rows hides it.  The table outline and the robot's feet are
    drawn to scale, since "askew" only means anything relative to them.
    """
    g = d.get("grid")
    if not g:
        return ""
    rows = [r for r in g["rows"] if "survive" in r]
    if not rows:
        return ""
    stances = sorted({r["stance_dy"] for r in rows}, reverse=True)
    W, H = 760, 300
    pw = W / len(stances)
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="survival by reach '
         f'target, in plan view, for two stance offsets">']
    # world -> panel: x right, y up
    x0, x1 = 0.0, 1.75
    y0, y1 = -0.45, 0.45

    for pi, dy in enumerate(stances):
        ox = pi * pw

        def X(v, ox=ox):
            return ox + 40 + (v - x0) / (x1 - x0) * (pw - 70)

        def Y(v):
            return 40 + (y1 - v) / (y1 - y0) * (H - 90)

        o.append(f'<text x="{ox + pw / 2:.0f}" y="20" font-size="11.5" '
                 f'text-anchor="middle" fill="var(--ink2)">'
                 f'stance dy = {dy * 1000:+.0f} mm</text>')
        # table top: x 0.50..1.68, y -0.2975..0.2975
        o.append(f'<rect x="{X(0.50):.1f}" y="{Y(0.2975):.1f}" '
                 f'width="{X(1.68) - X(0.50):.1f}" '
                 f'height="{Y(-0.2975) - Y(0.2975):.1f}" fill="var(--panel)" '
                 f'stroke="var(--rule)"/>')
        # feet, shifted by the stance
        for sgn in (+1, -1):
            fy = sgn * 0.2646 + dy
            o.append(f'<rect x="{X(0.185):.1f}" y="{Y(fy + 0.04):.1f}" '
                     f'width="{X(0.388) - X(0.185):.1f}" '
                     f'height="{Y(fy - 0.04) - Y(fy + 0.04):.1f}" '
                     f'fill="none" stroke="var(--ink3)" stroke-width="1"/>')
        for r in rows:
            if r["stance_dy"] != dy:
                continue
            tx, ty, _ = r["target"]
            s = r["survive"]
            col = ("var(--good)" if s >= 0.85 else
                   "var(--s2)" if s >= 0.5 else "var(--bad)")
            o.append(f'<circle cx="{X(tx):.1f}" cy="{Y(ty):.1f}" r="10" '
                     f'fill="{col}" fill-opacity="0.85"/>')
            o.append(f'<text x="{X(tx):.1f}" y="{Y(ty) + 3.5:.1f}" '
                     f'font-size="9" text-anchor="middle" fill="#fff">'
                     f'{100 * s:.0f}</text>')
        o.append(f'<text x="{X(0.19):.1f}" y="{Y(0.0) + 3.5:.1f}" '
                 f'font-size="9.5" text-anchor="middle" '
                 f'fill="var(--ink3)">robot</text>')
    o.append("</svg>")
    return "\n".join(o)


# ------------------------------------------------------------------ fig 3 --
def fig_sim2real(d):
    """Survival against each thing a deployment has to get right."""
    s = d.get("sim2real", {}).get("sim2real")
    if not s:
        return ""
    groups = [("state estimate", ["est2mm", "est5mm", "est10mm", "est20mm",
                                  "enc_only"]),
              ("friction", ["mu50", "mu75", "mu150"]),
              ("table pose", ["table_x10", "table_x20", "table_x40",
                              "table_y20", "table_y40"]),
              ("touchdown timing", ["late-10", "late-5", "late5", "late10"])]
    by = {r["name"]: r for r in s}
    base = by.get("baseline")
    prof = "q0.02"
    W = 760
    rowh, gap = 22, 26
    n = sum(len(g[1]) for g in groups) + len(groups)
    H = 46 + n * rowh + len(groups) * 8
    L, R = 150, 90
    o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="survival against '
         f'each sim-to-real axis">']
    if base:
        b = base["rate"][prof][0]
        o.append(f'<line x1="{L + b * (W - L - R):.1f}" y1="26" '
                 f'x2="{L + b * (W - L - R):.1f}" y2="{H - 10}" '
                 f'stroke="var(--s3)" stroke-width="1.2" '
                 f'stroke-dasharray="4 3"/>')
        o.append(f'<text x="{L + b * (W - L - R):.1f}" y="18" font-size="10" '
                 f'text-anchor="middle" fill="var(--s3)">'
                 f'nothing degraded {pct(b)}</text>')
    y = 38
    for title, names in groups:
        o.append(f'<text x="8" y="{y + 11}" font-size="11" '
                 f'fill="var(--ink2)">{title}</text>')
        y += gap
        for nm in names:
            r = by.get(nm)
            if not r:
                continue
            v = r["rate"][prof][0]
            col = ("var(--good)" if v >= 0.8 else
                   "var(--s2)" if v >= 0.4 else "var(--bad)")
            o.append(f'<text x="{L - 8}" y="{y + 11}" font-size="10.5" '
                     f'text-anchor="end" fill="var(--ink2)">{nm}</text>')
            o.append(f'<rect x="{L}" y="{y + 2}" width="{max(v, 0.004) * (W - L - R):.1f}" '
                     f'height="{rowh - 8}" fill="{col}" rx="2"/>')
            o.append(f'<text x="{L + max(v, 0.004) * (W - L - R) + 6:.1f}" '
                     f'y="{y + 11}" font-size="10" fill="var(--ink3)">'
                     f'{pct(v)}</text>')
            y += rowh
        y += 8
    o.append("</svg>")
    return "\n".join(o)


# ----------------------------------------------------------------- tables --
def table_weights(d):
    w = d.get("weights", {}).get("weights")
    if not w:
        return ""
    o = ['<div class="scroll"><table><thead><tr><th>configuration</th>'
         '<th>what it changes</th><th>plan reach mm</th><th>nominal</th>'
         '<th>survives 0.02 rad</th></tr></thead><tbody>']
    for r in w:
        if not r.get("planned"):
            continue
        rows = r["rows"]
        nom = [x for x in rows if not x.get("q_noise", x.get("dist"))]
        sv = r.get("survive", 0.0)
        what = ", ".join(f"{k} = {v:g}" for k, v in r["weights"].items()) \
            or "the S15 defaults"
        o.append(f'<tr><td><code>{r["name"]}</code></td><td>{what}</td>'
                 f'<td>{r.get("plan_reach_mm", float("nan")):.1f}</td>'
                 f'<td>{"ok" if (nom and nom[0]["ok"]) else "&mdash;"}</td>'
                 f'<td class="{cls(sv)}">{pct(sv)}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def table_matrix(d, key, caption_cols=("nominal", "q0.02", "winch1")):
    m = d.get(key, {}).get("matrix")
    if not m:
        return ""
    o = ['<div class="scroll"><table><thead><tr><th>plan</th><th>settle</th>']
    for c in caption_cols:
        o.append(f"<th>{c}</th>")
    o.append(f"<th>step {MS}</th></tr></thead><tbody>")
    for r in m:
        o.append(f'<tr><td><code>{r["tag"]}</code></td>'
                 f'<td>{r["settle"]}</td>')
        for c in caption_cols:
            v = r["rate"].get(c)
            if v is None:
                o.append("<td>&mdash;</td>")
            else:
                o.append(f'<td class="{cls(v[0])}">{pct(v[0])} '
                         f'<span style="color:var(--ink3)">of {v[1]}</span></td>')
        ms = r.get("solve_ms")
        o.append(f'<td>{ms:.1f}</td></tr>' if ms else '<td>&mdash;</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def table_horizon(d):
    m = d.get("horizon", {}).get("matrix")
    if not m:
        return ""
    o = ['<div class="scroll"><table><thead><tr><th>nodes</th><th>node dt</th>'
         f'<th>preview</th><th>step {MS}</th><th>p95 {MS}</th>'
         '<th>survives</th></tr></thead><tbody>']
    for r in sorted(m, key=lambda r: (r["preview_s"], r["dt_scale"])):
        v = r["rate"].get("q0.02") or [float("nan"), 0]
        o.append(f'<tr><td>{r["horizon"]}</td>'
                 f'<td>{20 * r["dt_scale"]:.0f} ms</td>'
                 f'<td>{r["preview_s"]:.2f} s</td>'
                 f'<td>{r["solve_ms"]:.1f}</td><td>{r["p95_ms"]:.1f}</td>'
                 f'<td class="{cls(v[0])}">{pct(v[0])}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def table_grid(d):
    g = d.get("grid")
    if not g:
        return ""
    o = ['<div class="scroll"><table><thead><tr><th>target x, y, z</th>'
         '<th>stance dy</th><th>mode</th><th>admissible</th>'
         '<th>plan reach mm</th><th>nominal</th><th>survives winch</th>'
         '<th>median reach mm</th></tr></thead><tbody>']
    for r in sorted(g["rows"], key=lambda r: (r["target"][0], r["target"][1],
                                              r["target"][2], -r["stance_dy"])):
        t = r["target"]
        sv = r.get("survive")
        o.append(f'<tr><td>{t[0]:.3f}, {t[1]:+.3f}, {t[2]:.3f}</td>'
                 f'<td>{1000 * r["stance_dy"]:+.0f} mm</td>'
                 f'<td>{r["mode"] or "&mdash;"}</td>'
                 f'<td>{r["n_admissible"]}</td>'
                 f'<td>{r.get("plan_reach_mm", float("nan")):.1f}</td>'
                 f'<td>{"ok" if r.get("nominal_ok") else "&mdash;"}</td>'
                 f'<td class="{cls(sv) if sv is not None else ""}">'
                 f'{pct(sv) if sv is not None else "&mdash;"}</td>'
                 f'<td>{r.get("reach_mm", float("nan")):.1f}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def table_sim2real(d):
    s = d.get("sim2real", {}).get("sim2real")
    if not s:
        return ""
    o = ['<div class="scroll"><table><thead><tr><th>degraded</th>'
         '<th>what it is</th><th>nominal</th><th>survives 0.02 rad</th>'
         '<th>why it failed</th></tr></thead><tbody>']
    for r in s:
        v = r["rate"].get("q0.02") or [float("nan"), 0]
        nom = r["rate"].get("nominal") or [float("nan"), 0]
        why = {}
        for x in r["rows"]:
            if not x["ok"]:
                why[x["why"]] = why.get(x["why"], 0) + 1
        kw = ", ".join(f"{k} = {v2}" for k, v2 in r["kw"].items()) or "&mdash;"
        o.append(f'<tr><td><code>{r["name"]}</code></td>'
                 f'<td style="font-size:.86em">{kw}</td>'
                 f'<td class="{cls(nom[0])}">{pct(nom[0])}</td>'
                 f'<td class="{cls(v[0])}">{pct(v[0])}</td>'
                 f'<td>{", ".join(f"{k} x{n}" for k, n in sorted(why.items())) or "&mdash;"}</td>'
                 f'</tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def table_attitude(d):
    """The base estimate decomposed.  Position, velocity, rate and attitude are
    four different sensors on a real robot and the table has to say which."""
    s = d.get("sim2real", {}).get("sim2real")
    if not s:
        return ""
    order = [("base_p2mm", "base position, 2 mm"),
             ("base_p10mm", "base position, 10 mm"),
             ("base_v10", "base linear velocity, 10 mm/s"),
             ("base_v50", "base linear velocity, 50 mm/s"),
             ("base_w10", "base angular rate, 10 mrad/s"),
             ("base_rp2mrad", "roll/pitch, 2 mrad (0.11&deg;)"),
             ("base_rp10mrad", "roll/pitch, 10 mrad (0.57&deg;)"),
             ("base_yaw2mrad", "yaw, 2 mrad"),
             ("base_yaw10mrad", "yaw, 10 mrad"),
             ("base_yaw50mrad", "yaw, 50 mrad (2.9&deg;)"),
             ("base_pv_good", "all four at 2 mm / 2 mrad")]
    by = {r["name"]: r for r in s}
    o = ['<div class="scroll"><table><thead><tr><th>degraded, alone</th>'
         '<th>nominal</th><th>survives 0.02 rad</th>'
         '<th>median reach mm</th></tr></thead><tbody>']
    base = by.get("baseline")
    if base:
        v = base["rate"]["q0.02"]
        rm = sorted(r["reach_mm"] for r in base["rows"] if r["ok"])
        o.append(f'<tr><td><b>nothing</b></td><td>ok</td>'
                 f'<td class="{cls(v[0])}">{pct(v[0])}</td>'
                 f'<td>{rm[len(rm) // 2] if rm else float("nan"):.1f}</td></tr>')
    for nm, lab in order:
        r = by.get(nm)
        if not r:
            continue
        v = r["rate"]["q0.02"]
        nom = r["rate"]["nominal"][0]
        rm = sorted(x["reach_mm"] for x in r["rows"] if x["ok"])
        o.append(f'<tr><td>{lab}</td>'
                 f'<td>{"ok" if nom else "&mdash;"}</td>'
                 f'<td class="{cls(v[0])}">{pct(v[0])}</td>'
                 f'<td>{rm[len(rm) // 2] if rm else float("nan"):.1f}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


ABLATE_WHY = {
    "AB_none": "everything",
    "AB_nocones": "the friction and wrench cones",
    "AB_nokeepout": "the table box keep-out",
    "AB_nocom": "the CoM support barrier",
    "AB_nohold": "the brace hold term",
    "AB_noland": "the landing-spot ramp",
    "AB_nojointlim": "the joint-limit barrier",
}


def table_ablate(d):
    m = d.get("ablate", {}).get("matrix")
    if not m:
        return ""
    base = next((r for r in m if r["tag"] == "AB_none"), None)
    o = ['<div class="scroll"><table><thead><tr><th>cost group</th>'
         '<th>nominal</th><th>0.02 rad</th><th>winch</th><th>verdict</th>'
         '</tr></thead><tbody>']
    for r in m:
        q = r["rate"].get("q0.02", [float("nan"), 0])
        w = r["rate"].get("winch1", [float("nan"), 0])
        nom = r["rate"].get("nominal", [0, 0])[0]
        if r["tag"] == "AB_none":
            note = "&mdash;"
        elif not nom:
            note = "<b>necessary</b> &mdash; fails from the nominal state"
        elif base and (q[0] < base["rate"]["q0.02"][0] - 0.1
                       or w[0] < base["rate"]["winch1"][0] - 0.1):
            note = "necessary"
        else:
            note = "no measurable effect"
        o.append(f'<tr><td>{ABLATE_WHY.get(r["tag"], r["tag"])}</td>'
                 f'<td>{"ok" if nom else "<span class=\'bad\'>falls</span>"}</td>'
                 f'<td class="{cls(q[0])}">{pct(q[0])}</td>'
                 f'<td class="{cls(w[0])}">{pct(w[0])}</td>'
                 f'<td>{note}</td></tr>')
    o.append("</tbody></table></div>")
    return "".join(o)


def videos(d):
    v = d.get("videos", {}).get("videos")
    if not v:
        return ""
    o = ['<div class="grid2">']
    for r in v:
        o.append(f'<figure><video controls muted playsinline preload="metadata" '
                 f'src="media/{r["file"]}"></video>'
                 f'<figcaption>{r["caption"]}</figcaption></figure>')
    o.append("</div>")
    return "".join(o)


FIGS = dict(fall=fig_fall, grid=fig_grid, sim2real=fig_sim2real)
TABLES = dict(weights=table_weights, grid=table_grid, sim2real=table_sim2real,
              horizon=table_horizon, attitude=table_attitude,
              ablate=table_ablate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-14_session17")
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title",
                    default="The lean was never robust, and the noise was not why")
    ap.add_argument("--style-from", default="2026-08-05_crocoddyl.html")
    args = ap.parse_args()

    d = load(args.dir)
    body = open(os.path.join(DOCS, os.path.basename(args.body))).read() \
        if os.path.exists(os.path.join(DOCS, os.path.basename(args.body))) \
        else open(args.body).read()
    for name, fn in FIGS.items():
        body = body.replace(f"<!--FIG:{name}-->", fn(d))
    for name, fn in TABLES.items():
        body = body.replace(f"<!--TABLE:{name}-->", fn(d))
    body = body.replace("<!--VIDEOS-->", videos(d))
    for key, cols in (("matrix", ("nominal", "q0.02", "winch1")),
                      ("launch", ("nominal", "q0.02", "winch1")),
                      ("combo", ("nominal", "q0.02", "winch1"))):
        body = body.replace(f"<!--TABLE:{key}-->", table_matrix(d, key, cols))

    from simple_page import style_from                          # noqa: E402
    html = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{args.title}</title>\n'
            + style_from(os.path.join(DOCS, args.style_from))
            + '\n</head>\n<body><div class="wrap">\n\n' + body
            + '\n</div></body>\n</html>\n')
    dst = os.path.join(DOCS, args.out)
    open(dst, "w").write(html)
    print(f"wrote {dst}  ({len(html)} bytes)")
    left = [t for t in ("FIG", "TABLE") if f"<!--{t}:" in html]
    if left:
        import re
        print("  UNFILLED:", sorted(set(re.findall(r"<!--(?:FIG|TABLE):"
                                                   r"([a-z_0-9]+)-->", html))))


if __name__ == "__main__":
    main()
