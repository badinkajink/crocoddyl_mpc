#!/usr/bin/env python3
"""Figures for the S14 "why the closed loop fails" page.

Same visual system as croco_figs (inline SVG painted with the page's CSS
variables, colour never the only channel), and it imports that module's Axes so
there is one implementation of the frame, not two.

usage: croco_why_figs.py --why WHY.json [--why-cad WHY2.json]
                         [--surfaces brace_surfaces.json] [--out figs.json]
                         [--inject page.html]
"""

import argparse
import json
import os

import numpy as np

from croco_figs import Axes, esc, SER, DASH, inject, W

ROLE_LABEL = {"brace": "planned brace", "brace_arm": "bracing arm, elsewhere",
              "reach_arm": "reaching arm", "torso": "torso / pelvis",
              "legs": "legs", "other": "other"}
ROLE_COLOR = {"brace": "var(--s2)", "brace_arm": "var(--s1)",
              "reach_arm": "var(--s4)", "torso": "var(--s5)",
              "legs": "var(--s3)", "other": "var(--ink3)"}
MODE_SHORT = {"palm": "palm", "elbow": "elbow", "elbow+forearm": "elb+fore",
              "elbow+palm": "elb+palm", "forearm+palm": "fore+palm",
              "elbow+forearm+palm": "elb+fore+palm", "legs_only": "legs only"}


def _cells(why):
    """[(target, mode, cell)] in grid order."""
    out = []
    for k, c in why["cells"].items():
        t, mode = k.split("|")
        out.append((float(t), mode, c))
    return sorted(out, key=lambda r: (r[0], r[1]))


def _bucket(load):
    """Collapse the load dict onto the six roles."""
    acc = {}
    for k, v in load.items():
        acc.setdefault("brace" if k.startswith("brace:") else k, 0.0)
        acc["brace" if k.startswith("brace:") else k] += v
    return acc


# --------------------------------------------------------------- scoreboard --
def fig_scoreboard(why):
    """The grid, scored on what the PLANT did rather than on admissibility."""
    targets, modes = why["targets"], [m for m in why["modes"] if m != "legs_only"]
    cells = {(t, m): c for t, m, c in _cells(why)}
    cw, ch, x0, y0 = 118, 48, 128, 34
    w = x0 + cw * len(targets) + 12
    h = y0 + ch * len(modes) + 30
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="closed-loop grid">']
    for j, t in enumerate(targets):
        g.append(f'<text x="{x0 + cw * j + cw / 2:.0f}" y="{y0 - 10}" font-size="10.5" '
                 f'text-anchor="middle" fill="var(--ink3)">x = {t:.3f}</text>')
    for i, mode in enumerate(modes):
        y = y0 + ch * i
        g.append(f'<text x="{x0 - 8}" y="{y + ch / 2 + 4:.0f}" font-size="10.5" '
                 f'text-anchor="end" fill="var(--ink2)">{esc(MODE_SHORT.get(mode, mode))}</text>')
        for j, t in enumerate(targets):
            x = x0 + cw * j
            c = cells.get((t, mode))
            if c is None:
                g.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" '
                         f'height="{ch - 4}" fill="var(--line)" opacity="0.25" rx="3"/>')
                g.append(f'<text x="{x + cw / 2:.0f}" y="{y + ch / 2 + 4:.0f}" '
                         f'font-size="9.5" text-anchor="middle" fill="var(--ink3)">'
                         f'not certified</text>')
                continue
            oc = c.get("outcome",
                       "fell_braced" if c["fell"] else
                       ("stalled" if c["reach_end"] > 60 else "reached"))
            fill, op, lab = {
                "reached":       ("var(--s2)", 0.30, f"reached {c['reach_end']:.0f} mm"),
                "stalled":       ("var(--s4)", 0.30, f"stalled {c['reach_end']:.0f} mm short"),
                "fell_braced":   ("var(--s5)", 0.36, "fell under the brace"),
                "fell_approach": ("var(--s5)", 0.20, "fell before the brace"),
            }[oc]
            g.append(f'<rect x="{x + 2}" y="{y + 2}" width="{cw - 4}" height="{ch - 4}" '
                     f'fill="{fill}" opacity="{op}" rx="3"/>')
            g.append(f'<text x="{x + cw / 2:.0f}" y="{y + 16:.0f}" font-size="10" '
                     f'text-anchor="middle" fill="var(--ink)">{esc(lab)}</text>')
            acc = _bucket(c["load"])
            tot = sum(acc.values())
            edge = c.get("inboard")
            if tot <= 1.0:
                sub1, sub2 = "no table load at all", ""
            else:
                sub1 = f"brace carries {100 * acc.get('brace', 0) / tot:.0f}% of it"
                sub2 = (f"{edge:.0f} mm in from the edge"
                        if edge is not None else "")
            g.append(f'<text x="{x + cw / 2:.0f}" y="{y + 29:.0f}" font-size="8.5" '
                     f'text-anchor="middle" fill="var(--ink3)">{esc(sub1)}</text>')
            if sub2:
                g.append(f'<text x="{x + cw / 2:.0f}" y="{y + 40:.0f}" '
                         f'font-size="8.5" text-anchor="middle" '
                         f'fill="var(--ink3)">{esc(sub2)}</text>')
    g.append(f'<text x="4" y="{h - 16}" font-size="9" fill="var(--ink3)">'
             f'each cell says what happened, what fraction of the table load went '
             f'through the planned brace, and how far in from the edge it sat</text>')
    g.append(f'<text x="4" y="{h - 5}" font-size="9" fill="var(--ink3)">'
             f'&quot;inboard&quot; is the force-weighted distance from the load to the '
             f'nearest table edge — the slab is 595 mm wide</text>')
    g.append("</svg>")
    return "".join(g)


# --------------------------------------------------------------- load split --
def fig_load_split(why):
    """Who actually carries the table load, per cell."""
    rows = [(t, m, c) for t, m, c in _cells(why)
            if sum(_bucket(c["load"]).values()) > 1.0]
    bh, gap, x0 = 17, 5, 158
    h = 34 + len(rows) * (bh + gap) + 20
    bw = W - x0 - 82
    g = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="table load by role">']
    tot_max = max(sum(_bucket(c["load"]).values()) for _, _, c in rows)
    # SQUARE-ROOT width.  One cell puts 513 N through its torso and several put
    # 8-13 N through a brace; on a linear axis the second group is a hairline and
    # the figure says only "one cell is big", which is the least interesting
    # thing in it.  sqrt keeps the ordering and the ratios readable, and the
    # newton total is printed on every row so nothing is hidden behind the scale.
    def wpx(v):
        return np.sqrt(max(v, 0.0) / tot_max) * bw
    for i, (t, mode, c) in enumerate(rows):
        y = 30 + i * (bh + gap)
        g.append(f'<text x="{x0 - 6}" y="{y + bh - 4:.0f}" font-size="9.5" '
                 f'text-anchor="end" fill="var(--ink2)">'
                 f'{t:.3f} · {esc(MODE_SHORT.get(mode, mode))}</text>')
        acc = _bucket(c["load"])
        tot = sum(acc.values())
        cx, run = x0, 0.0
        for role in ("brace", "brace_arm", "reach_arm", "torso", "legs", "other"):
            v = acc.get(role, 0.0)
            if v <= 0:
                continue
            nxt = wpx(run + v)
            g.append(f'<rect x="{x0 + wpx(run):.1f}" y="{y}" '
                     f'width="{max(nxt - wpx(run), 0.8):.1f}" '
                     f'height="{bh}" fill="{ROLE_COLOR[role]}" opacity="0.85"/>')
            run += v
        g.append(f'<text x="{x0 + wpx(tot) + 5:.1f}" y="{y + bh - 4:.0f}" '
                 f'font-size="9" fill="var(--ink3)">{tot:.0f} N</text>')
    g.append(f'<text x="4" y="14" font-size="9.5" fill="var(--ink3)">'
             f'mean normal force on the table over the braced phase, by which part of the '
             f'robot delivers it</text>')
    g.append(f'<text x="4" y="25" font-size="9" fill="var(--ink3)">'
             f'bar width is √force so the 8 N rows stay legible beside the 513 N one; '
             f'the total is printed on each row</text>')
    g.append("</svg>")
    return "".join(g)


# The series stylesheet has no `.key`/`.dot` rules -- croco_figs emits markup for
# them and it has always rendered as one run-together line of text.  Rather than
# edit a stylesheet three published pages share, the legends here carry their own
# inline style, which also survives being screenshotted out of the page.
def _key(items):
    sp = ("display:inline-flex;align-items:center;gap:.35rem;"
          "margin:0 .9rem .25rem 0;font-size:.82rem;color:var(--ink3)")
    dot = "width:.62rem;height:.62rem;border-radius:50%;flex:0 0 auto;background:"
    return ('<div style="margin:.35rem 0 0">'
            + "".join(f'<span style="{sp}"><i style="{dot}{c}"></i>{esc(lab)}</span>'
                      for c, lab in items) + "</div>")


def key_of(labels):
    """Legend in the page's own idiom, for charts whose series converge.

    reach_trace and gap_trace both end with three of their lines within a few
    pixels of each other -- direct end-labels there stack into an unreadable
    pile no matter how they are nudged, and the honest fix is a legend rather
    than a cleverer nudge.
    """
    return _key([(SER[i % 5], lab) for i, lab in enumerate(labels)])


def key_roles():
    return _key([(ROLE_COLOR[r], ROLE_LABEL[r]) for r in
                 ("brace", "brace_arm", "reach_arm", "torso", "legs")])


# ---------------------------------------------------------------- table map --
def fig_table_map(why, table=(1.09, 0.0, 0.59, 0.2975)):
    """Top view of the tabletop with every carrying contact of every cell.

    Drawn at TRUE ASPECT.  The point of the figure is that the load sits at the
    lateral edge of a slab that is 1180 mm long and only 595 mm wide, and a
    stretched y axis would make a 5 mm margin look like a comfortable one.
    """
    cx, cy, hx, hy = table
    xlim = (cx - hx - 0.06, cx + hx + 0.06)
    ylim = (cy - hy - 0.07, cy + hy + 0.07)
    pad = (48, 20, 22, 36)
    plot_w = W - pad[0] - pad[1]
    h = int(round(plot_w * (ylim[1] - ylim[0]) / (xlim[1] - xlim[0]))) + pad[2] + pad[3]
    ax = Axes(xlim, ylim, w=W, h=h,
              xlabel="world x [m] — the robot stands at x ≈ 0.2, the table runs away from it",
              ylabel="world y [m], top view, true aspect", pad=pad,
              yticks=[-0.2, 0.0, 0.2])
    ax.parts.append(
        f'<rect x="{ax.X(cx - hx):.1f}" y="{ax.Y(cy + hy):.1f}" '
        f'width="{ax.X(cx + hx) - ax.X(cx - hx):.1f}" '
        f'height="{ax.Y(cy - hy) - ax.Y(cy + hy):.1f}" fill="var(--ink3)" '
        f'opacity="0.10" stroke="var(--line)"/>')
    for s in (-1, 1):
        ax.parts.append(
            f'<line x1="{ax.X(cx - hx):.1f}" y1="{ax.Y(cy + s * (hy - 0.008)):.1f}" '
            f'x2="{ax.X(cx + hx):.1f}" y2="{ax.Y(cy + s * (hy - 0.008)):.1f}" '
            f'stroke="var(--s5)" stroke-width="1" stroke-dasharray="4 3" opacity="0.8"/>')
    ax.parts.append(
        f'<text x="{ax.X(cx):.1f}" y="{ax.Y(cy + hy) - 5:.1f}" font-size="9" '
        f'text-anchor="middle" fill="var(--s5)">dashed: 8 mm in from the rail</text>')
    tx, ty = why["targets"], why["target_yz"][0]
    for t in tx:
        ax.parts.append(
            f'<path d="M {ax.X(t) - 4:.1f},{ax.Y(ty):.1f} l 4,-4 l 4,4 l -4,4 Z" '
            f'fill="none" stroke="var(--ink2)" stroke-width="1"/>')
    ax.parts.append(
        f'<text x="{ax.X(tx[-1]) + 8:.1f}" y="{ax.Y(ty) + 3.5:.1f}" font-size="9" '
        f'fill="var(--ink2)">reach targets</text>')
    for _, _, c in _cells(why):
        for r in c["rows"][c["n_approach"]::3]:
            for con in r["con"]:
                if con["fn"] < 5.0 or con["where"] not in ("top", "edge"):
                    continue
                role = "brace" if con["site"] else con["role"]
                rr = 1.8 + 3.4 * np.sqrt(min(con["fn"], 300.0) / 300.0)
                ax.parts.append(
                    f'<circle cx="{ax.X(con["pos"][0]):.1f}" '
                    f'cy="{ax.Y(con["pos"][1]):.1f}" r="{rr:.1f}" '
                    f'fill="{ROLE_COLOR.get(role, "var(--ink3)")}" opacity="0.45"/>')
    return ax.svg("contact positions on the tabletop")


# --------------------------------------------------------------- the stall ---
def fig_reach_trace(why, targets=None):
    """Plant reach error against the plan's, per target."""
    ax = Axes((0, 200), (0, 420), h=254, xlabel="node (dt = 20 ms)",
              ylabel="hand-to-target error [mm]", pad=(52, 24, 34, 34))
    ax.vspan(120, 200, "contacts on (braced phase)")
    rows = [(t, m, c) for t, m, c in _cells(why) if not c["fell"]]
    seen = {}
    for t, mode, c in rows:
        seen.setdefault(t, []).append((mode, c))
    labels = []
    for i, (t, group) in enumerate(sorted(seen.items())):
        best = min(group, key=lambda g: g[1]["reach_end"])
        c = best[1]
        ax.line([r["k"] for r in c["rows"]],
                [min(r["reach"], 415) for r in c["rows"]],
                SER[i % 5], DASH[i % 5])
        labels.append(f"x = {t:.2f} ({MODE_SHORT.get(best[0], best[0])})")
        if "reach_plan" in c["rows"][0]:
            ax.line([r["k"] for r in c["rows"]],
                    [min(r["reach_plan"], 415) for r in c["rows"]],
                    SER[i % 5], "1 3", width=1.0, opacity=0.6)
    ax.parts.append('<text x="52" y="18" font-size="9" fill="var(--ink3)">'
                    'solid: MuJoCo · dotted: the plan the MPC is tracking · '
                    'one line per target, its best surviving mode</text>')
    return ax.svg("reach error, plan against plant") + key_of(labels)


GAP_PICKS = [(1.050, "elbow", "elbow"),          # a brace that works
             (1.050, "palm", "palm"),            # a brace that hooks the rail
             (1.150, "elbow+palm", "elbow")]      # a brace that never arrives


def fig_gap_trace(why, picks=GAP_PICKS):
    """Brace-surface clearance: what the plan asks for, what the plant gets."""
    ax = Axes((100, 200), (-25, 70), h=254,
              xlabel="node (dt = 20 ms) — contacts switch on at 120",
              ylabel="clearance from the brace surface to the tabletop [mm]",
              pad=(52, 24, 34, 34))
    ax.vspan(120, 200, "braced phase")
    ax.hline(0, "var(--ink)", "4 3")
    cells = {(t, m): c for t, m, c in _cells(why)}
    labels, n = [], 0
    for t, mode, site in picks:
        c = cells.get((t, mode))
        if c is None or site not in c["gap_braced"]:
            continue
        rows = [r for r in c["rows"] if r["k"] >= 100]
        ax.line([r["k"] for r in rows],
                [float(np.clip(r["gap"][site], -25, 70)) for r in rows],
                SER[n % 5], DASH[n % 5])
        labels.append(f"x = {t:.2f}, {MODE_SHORT.get(mode, mode)} — {site} surface")
        if "gap_plan" in rows[0]:
            ax.line([r["k"] for r in rows],
                    [float(np.clip(r["gap_plan"][site], -25, 70)) for r in rows],
                    SER[n % 5], "1 3", width=1.0, opacity=0.65)
        n += 1
    ax.parts.append('<text x="52" y="18" font-size="9" fill="var(--ink3)">'
                    'solid: MuJoCo · dotted: the plan the MPC tracks · '
                    'zero is the wood</text>')
    return ax.svg("brace clearance over the braced phase") + key_of(labels)


# ------------------------------------------------------------ hand section ---
def _silhouette(P, ax, n=90):
    """Envelope of a point cloud in the (x, y) plane, as an SVG path.

    Per-x-bin min/max of y, with empty bins filled by interpolation so a part
    whose vertices cluster on feature edges (every one of these is a machined
    part, so they all do) does not come out serrated.
    """
    x = P[:, 0]
    edges = np.linspace(x.min(), x.max(), n + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n - 1)
    xs = 0.5 * (edges[:-1] + edges[1:])
    top = np.full(n, np.nan)
    bot = np.full(n, np.nan)
    for b in range(n):
        s = P[idx == b]
        if len(s):
            top[b], bot[b] = s[:, 1].max(), s[:, 1].min()
    ok = ~np.isnan(top)
    if ok.sum() < 2:
        return ""
    top = np.interp(xs, xs[ok], top[ok])
    bot = np.interp(xs, xs[ok], bot[ok])
    pts = list(zip(xs, top)) + list(zip(xs[::-1], bot[::-1]))
    return "M " + " L ".join(f"{ax.X(a):.1f},{ax.Y(b):.1f}" for a, b in pts) + " Z"


def _rect(ax, x0, x1, y0, y1, **kw):
    st = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in kw.items())
    return (f'<rect x="{ax.X(x0):.1f}" y="{ax.Y(y1):.1f}" '
            f'width="{ax.X(x1) - ax.X(x0):.1f}" '
            f'height="{ax.Y(y0) - ax.Y(y1):.1f}" {st}/>')


OLD_PROXY = [("wrist pad", -20.0, 63.0, -42.5, 42.5, True),
             ("gripper box", 54.0, 139.0, -31.5, 31.5, False)]
NEW_PROXY = [("wrist pad", -20.0, 63.0, -28.0, 28.0, False),
             ("flange", 54.0, 79.0, -31.5, 31.5, False),
             ("gripper body", 56.0, 139.0, -19.0, 15.0, False)]


def fig_hand_section(cad, old=-42.5, new=-31.5):
    """The hand seen along the brace direction: CAD against both proxy sets.

    `cad` is {part: Nx3 points in the wrist_yaw_link frame [mm]}; the brace
    direction is body -y, so DOWN on this figure is toward the wood.  It is an
    envelope rather than a section: the fingers sit +-54..106 mm off the
    mid-plane in z and are projected in, because the question the figure answers
    is "how far down does ANY of this reach", which is what decides when MuJoCo
    calls a palm brace a contact.
    """
    ax = Axes((-32, 238), (-58, 40), w=W, h=250,
              xlabel="distance along the wrist axis [mm] — wrist joint at 0, finger tips at 225",
              ylabel="toward the tabletop [mm]", pad=(50, 14, 46, 34),
              yticks=[-40, -20, 0, 20])
    ax.parts.append(_rect(ax, -32, 238, -58, old, fill="var(--s4)", opacity="0.10"))
    for yv, col, lab in ((old, "var(--s4)", "tabletop the OLD proxies touch"),
                         (new, "var(--s2)", "tabletop the CAD touches")):
        ax.parts.append(
            f'<line x1="{ax.X(-32):.1f}" y1="{ax.Y(yv):.1f}" '
            f'x2="{ax.X(238):.1f}" y2="{ax.Y(yv):.1f}" stroke="{col}" '
            f'stroke-width="1.4"/>')
        ax.parts.append(
            f'<text x="{ax.X(236):.1f}" y="{ax.Y(yv) - 4:.1f}" font-size="9" '
            f'text-anchor="end" fill="{col}">{lab} ({yv:+.1f} mm)</text>')
    # the dimension the whole thing is about
    xd = 14.0
    ax.parts.append(
        f'<line x1="{ax.X(xd):.1f}" y1="{ax.Y(old):.1f}" x2="{ax.X(xd):.1f}" '
        f'y2="{ax.Y(new):.1f}" stroke="var(--ink)" stroke-width="1.2"/>')
    ax.parts.append(
        f'<text x="{ax.X(xd) + 6:.1f}" y="{(ax.Y(old) + ax.Y(new)) / 2 + 3.5:.1f}" '
        f'font-size="10.5" fill="var(--ink)">11.0 mm of hardware that is not there</text>')

    for P in cad.values():
        d = _silhouette(P, ax)
        if d:
            ax.parts.append(f'<path d="{d}" fill="var(--s1)" opacity="0.42" '
                            f'stroke="var(--s1)" stroke-width="0.7"/>')
    for nm, x0, x1, y0, y1, rounded in OLD_PROXY:
        rx = abs(ax.Y(y0) - ax.Y(y1)) / 2 if rounded else 2
        ax.parts.append(_rect(ax, x0, x1, y0, y1, fill="none", stroke="var(--s4)",
                              stroke_width="1.5", stroke_dasharray="5 3",
                              rx=f"{rx:.0f}"))
    for nm, x0, x1, y0, y1, rounded in NEW_PROXY:
        ax.parts.append(_rect(ax, x0, x1, y0, y1, fill="none", stroke="var(--s2)",
                              stroke_width="1.5", rx="2"))
    for lab, col, px in (("filled: CAD", "var(--s1)", 52),
                         ("dashed: proxies as they were", "var(--s4)", 150),
                         ("solid: CAD-faithful proxies", "var(--s2)", 350)):
        ax.parts.append(f'<text x="{px}" y="18" font-size="9.5" fill="{col}">'
                        f'{esc(lab)}</text>')
    return ax.svg("hand envelope against its collision proxies")


# ------------------------------------------------------------ before/after ---
def fig_cad_effect(pre, post):
    """What the CAD-faithful brace surfaces changed, cell by cell."""
    pc = {f"{t:.3f}|{m}": c for t, m, c in _cells(pre)}
    qc = {f"{t:.3f}|{m}": c for t, m, c in _cells(post)}
    keys = [k for k in sorted(set(pc) | set(qc))]
    bh, gap, x0 = 15, 4, 150
    h = 34 + len(keys) * (bh + gap) + 12
    g = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="model fix effect">']
    g.append(f'<text x="{x0}" y="14" font-size="9.5" fill="var(--ink3)">'
             f'closed-loop reach error, clipped at 400 mm · ▲ = toppled</text>')
    bw = W - x0 - 96
    for i, k in enumerate(keys):
        y = 26 + i * (bh + gap)
        t, mode = k.split("|")
        g.append(f'<text x="{x0 - 6}" y="{y + bh - 3:.0f}" font-size="9" '
                 f'text-anchor="end" fill="var(--ink2)">'
                 f'{t} · {esc(MODE_SHORT.get(mode, mode))}</text>')
        for c, col, dy in ((pc.get(k), "var(--ink3)", 0),
                           (qc.get(k), "var(--s2)", bh / 2)):
            if c is None:
                continue
            v = min(c["reach_end"], 400.0)
            g.append(f'<rect x="{x0}" y="{y + dy:.1f}" '
                     f'width="{max(v / 400.0 * bw, 0.8):.1f}" height="{bh / 2 - 1:.1f}" '
                     f'fill="{col}" opacity="0.9"/>')
            if c["fell"]:
                g.append(f'<text x="{x0 + v / 400.0 * bw + 4:.1f}" '
                         f'y="{y + dy + bh / 2 - 2:.1f}" font-size="8" '
                         f'fill="var(--s5)">▲</text>')
    g.append("</svg>")
    return "".join(g)


def key_models():
    return _key([("var(--ink3)", "proxies as they were"),
                 ("var(--s2)", "CAD-faithful proxies")])


# ------------------------------------------------------------------- driver --
def hand_cad():
    """CAD point clouds for the hand, in the wrist_yaw_link frame [mm]."""
    import mujoco
    import contact_select as cs
    import brace_surfaces as bs
    m = mujoco.MjModel.from_xml_path(cs.MODEL)
    arm = cs.BRACE_ARM
    b = cs.bid(m, f"{arm}_wrist_yaw_link")
    wm = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b
          and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH][0]
    P, _ = bs.geom_points(m, wm)
    out = {"wrist housing": P * 1e3}
    for k, V in bs.magpie_parts().items():
        out[k] = V * 1e3
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--why", required=True)
    ap.add_argument("--why-cad")
    ap.add_argument("--out")
    ap.add_argument("--inject")
    args = ap.parse_args()

    pre = json.load(open(args.why))
    figs = {}
    src = pre
    if args.why_cad and os.path.exists(args.why_cad):
        post = json.load(open(args.why_cad))
        figs["cad_effect"] = fig_cad_effect(pre, post) + key_models()
        src = post
        figs["scoreboard_cad"] = fig_scoreboard(post)
    figs["scoreboard"] = fig_scoreboard(pre)
    figs["load_split"] = fig_load_split(pre) + key_roles()
    figs["table_map"] = fig_table_map(pre) + key_roles()
    figs["reach_trace"] = fig_reach_trace(pre)
    figs["gap_trace"] = fig_gap_trace(pre)
    figs["hand_section"] = fig_hand_section(hand_cad())

    if args.out:
        json.dump(figs, open(args.out, "w"))
        print(f"wrote {args.out}  ({len(figs)} figures)")
    if args.inject:
        inject(args.inject, figs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
