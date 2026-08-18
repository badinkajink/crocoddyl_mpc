#!/usr/bin/env python3
"""Build the S13 docpage figures as inline SVG, in the series' own visual system.

Inline SVG rather than PNG for the same reason the S11 figures are: the page has
a light and a dark theme, and a raster figure only ever matches one of them.
Everything here paints with the page's CSS variables (--ink, --line, --s1..--s5),
so a figure re-themes with the page and needs no second export.

Colour is never the only channel.  Every series in a multi-line chart also gets a
dash pattern and a direct end-label, so the charts survive being printed, being
read by someone with a colour vision deficiency, and being screenshotted into a
slide deck where the CSS variables do not follow.

Writes one JSON dict of {figure name: svg string}; the docpage substitutes them
into <!--FIG:name--> placeholders (see `inject`).
"""

import argparse
import json
import os

import numpy as np

W, H = 640, 268
PAD_L, PAD_R, PAD_T, PAD_B = 52, 74, 26, 34
SER = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)"]
DASH = ["none", "6 3", "2 3", "9 3 2 3", "4 2"]


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


class Axes:
    """Minimal linear x/y axes with recessive grid and direct end-labels."""

    def __init__(self, xlim, ylim, w=W, h=H, xlabel="", ylabel="",
                 pad=(PAD_L, PAD_R, PAD_T, PAD_B), yticks=None, xticks=None):
        self.w, self.h = w, h
        self.pl, self.pr, self.pt, self.pb = pad
        self.xlim, self.ylim = xlim, ylim
        self.parts = []
        self.xlabel, self.ylabel = xlabel, ylabel
        self.yticks = yticks
        self.xticks = xticks

    def X(self, x):
        x0, x1 = self.xlim
        return self.pl + (x - x0) / (x1 - x0) * (self.w - self.pl - self.pr)

    def Y(self, y):
        y0, y1 = self.ylim
        return self.h - self.pb - (y - y0) / (y1 - y0) * (self.h - self.pt - self.pb)

    # ------------------------------------------------------------------ marks
    def line(self, xs, ys, color, dash="none", label=None, width=2.0,
             label_dy=0.0, opacity=1.0, label_at=-1):
        """`label_at` picks WHICH point carries the direct label.

        The default (the last point) is right for a time series that fans out.
        It is wrong for a sweep whose series converge at the right-hand end --
        there the labels stack on top of each other and the place they are
        separated is the START, so `label_at=0`.
        """
        pts = " ".join(f"{self.X(x):.1f},{self.Y(y):.1f}" for x, y in zip(xs, ys))
        self.parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" '
            f'stroke-dasharray="{dash}" opacity="{opacity}"/>')
        if label:
            anchor = "start" if label_at != 0 else "end"
            dx = 5 if label_at != 0 else -6
            self.parts.append(
                f'<text x="{self.X(xs[label_at]) + dx:.1f}" '
                f'y="{self.Y(ys[label_at]) + 3.5 + label_dy:.1f}" font-size="10" '
                f'text-anchor="{anchor}" fill="{color}">{esc(label)}</text>')
        return self

    def hline(self, y, color="var(--line)", dash="3 3", label=None, width=1.0):
        self.parts.append(
            f'<line x1="{self.pl}" y1="{self.Y(y):.1f}" x2="{self.w - self.pr}" '
            f'y2="{self.Y(y):.1f}" stroke="{color}" stroke-width="{width}" '
            f'stroke-dasharray="{dash}"/>')
        if label:
            self.parts.append(
                f'<text x="{self.w - self.pr + 4}" y="{self.Y(y) + 3.5:.1f}" '
                f'font-size="9" fill="var(--ink3)">{esc(label)}</text>')
        return self

    def vspan(self, x0, x1, label=None, fill="var(--ink3)", op=0.07):
        self.parts.append(
            f'<rect x="{self.X(x0):.1f}" y="{self.pt}" '
            f'width="{self.X(x1) - self.X(x0):.1f}" '
            f'height="{self.h - self.pt - self.pb}" fill="{fill}" '
            f'opacity="{op}"/>')
        if label:
            self.parts.append(
                f'<text x="{(self.X(x0) + self.X(x1)) / 2:.1f}" y="{self.pt + 11}" '
                f'font-size="9" fill="var(--ink3)" text-anchor="middle">'
                f'{esc(label)}</text>')
        return self

    def dot(self, x, y, color, r=3.2, ring=True):
        if ring:
            self.parts.append(
                f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r + 1.6:.1f}" '
                f'fill="var(--bg)"/>')
        self.parts.append(
            f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" '
            f'fill="{color}"/>')
        return self

    def text(self, x, y, s, color="var(--ink3)", size=9, anchor="start"):
        self.parts.append(
            f'<text x="{self.X(x):.1f}" y="{self.Y(y):.1f}" font-size="{size}" '
            f'fill="{color}" text-anchor="{anchor}">{esc(s)}</text>')
        return self

    # ------------------------------------------------------------------ frame
    def svg(self, aria=""):
        g = []
        yt = self.yticks if self.yticks is not None else _ticks(*self.ylim)
        xt = self.xticks if self.xticks is not None else _ticks(*self.xlim)
        for y in yt:
            g.append(f'<line x1="{self.pl}" y1="{self.Y(y):.1f}" '
                     f'x2="{self.w - self.pr}" y2="{self.Y(y):.1f}" '
                     f'stroke="var(--line)" stroke-width="1" opacity="0.7"/>')
            g.append(f'<text x="{self.pl - 6}" y="{self.Y(y) + 3.5:.1f}" '
                     f'font-size="9" fill="var(--ink3)" text-anchor="end">'
                     f'{_fmt(y)}</text>')
        for x in xt:
            g.append(f'<text x="{self.X(x):.1f}" y="{self.h - self.pb + 13}" '
                     f'font-size="9" fill="var(--ink3)" text-anchor="middle">'
                     f'{_fmt(x)}</text>')
        g.append(f'<line x1="{self.pl}" y1="{self.h - self.pb}" '
                 f'x2="{self.w - self.pr}" y2="{self.h - self.pb}" '
                 f'stroke="var(--line)" stroke-width="1"/>')
        if self.xlabel:
            g.append(f'<text x="{(self.pl + self.w - self.pr) / 2:.0f}" '
                     f'y="{self.h - 3}" font-size="9.5" fill="var(--ink3)" '
                     f'text-anchor="middle">{esc(self.xlabel)}</text>')
        if self.ylabel:
            # Left-anchored ABOVE the plot area, not end-anchored beside the top
            # tick: an end-anchored y-label runs off the left edge of the viewBox
            # for any label longer than the tick gutter, which silently truncates
            # it to "gon [mm]" and "rror [mm]" in the rendered page.
            g.append(f'<text x="2" y="{self.pt - 4}" font-size="9.5" '
                     f'fill="var(--ink3)" text-anchor="start">{esc(self.ylabel)}</text>')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" '
                f'aria-label="{esc(aria)}">' + "".join(g + self.parts) + "</svg>")


def _ticks(a, b, n=5):
    span = b - a
    step = 10.0 ** np.floor(np.log10(span / n))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= n:
            step *= mult
            break
    lo = np.ceil(a / step) * step
    return [round(v, 10) for v in np.arange(lo, b + 1e-9, step)]


def _fmt(v):
    if abs(v) >= 1000 or (v != 0 and abs(v) < 0.01):
        return f"{v:g}"
    s = f"{v:.10g}"
    return s


# --------------------------------------------------------------------------- #
def load(run_dir, name):
    with open(os.path.join(run_dir, name)) as fh:
        return json.load(fh)


def key(ctrls, labels):
    """Legend row in the page's own `.key`/`.dot` idiom.

    Five series is two too many to direct-label at the right edge of a time
    series: the four that fall all end on the floor within 8 px of each other and
    the two that survive end within 5 px, so the labels collide into an unreadable
    stack.  A legend carries identity for all five and the plots direct-label only
    the two that finish the maneuver, which is the selective-labelling rule
    applied where it actually bites.
    """
    out = ['<div class="key">']
    for i, (c, lab) in enumerate(zip(ctrls, labels)):
        out.append(f'<span><i class="dot" style="background:{SER[i % 5]};'
                   f'border-radius:50%"></i>{esc(lab)}</span>')
    out.append("</div>")
    return "".join(out)


def _series(ax, run_dir, ctrls, labels, value, direct=("riccati", "mpc"),
            dy=None):
    for i, (c, lab) in enumerate(zip(ctrls, labels)):
        L = load(run_dir, f"replay_s13_{c}.json")["log"]
        ax.line([r["t"] for r in L], [value(r) for r in L],
                SER[i % 5], DASH[i % 5],
                label=lab if c in direct else None,
                label_dy=(dy or {}).get(c, 0.0))
    return ax


def fig_ladder(run_dir, ctrls, labels):
    """Pelvis height vs time: who stays up, and for how long."""
    ax = Axes((0, 4.0), (0, 1.1), xlabel="time [s]", ylabel="pelvis height [m]")
    ax.hline(0.55, color="var(--warn)", dash="4 3", label="toppled")
    _series(ax, run_dir, ctrls, labels, lambda r: r["pelvis_z"],
            dy={"riccati": -6.0, "mpc": 6.0})
    return ax.svg("pelvis height over time for each controller")


def fig_reach(run_dir, ctrls, labels, n_approach=120):
    ax = Axes((0, 4.0), (0, 700), xlabel="time [s]",
              ylabel="reaching-hand error [mm]")
    ax.vspan(n_approach * 0.02, 4.0, "braced")
    _series(ax, run_dir, ctrls, labels,
            lambda r: min(r["reach_err"] * 1000, 690),
            dy={"riccati": -5.0, "mpc": 6.0})
    return ax.svg("reaching hand error over time")


def fig_margin(run_dir, ctrls, labels, n_approach=120):
    ax = Axes((0, 4.0), (-120, 140), xlabel="time [s]",
              ylabel="CoM margin in the foot polygon [mm]")
    ax.vspan(n_approach * 0.02, 4.0, "braced")
    ax.hline(0.0, color="var(--warn)", dash="4 3", label="edge")
    _series(ax, run_dir, ctrls, labels,
            lambda r: max(min(r["support_margin"] * 1000, 135), -115),
            dy={"riccati": -5.0, "mpc": 6.0})
    return ax.svg("CoM support margin over time")


def fig_brace(run_dir, tag="mpc", n_approach=120):
    L = load(run_dir, f"replay_s13_{tag}.json")["log"]
    plan = load(run_dir, "plan_s13.json")
    ax = Axes((0, 4.0), (0, 160), xlabel="time [s]",
              ylabel="brace normal force [N]")
    ax.vspan(n_approach * 0.02, 4.0, "braced")
    for i, s in enumerate(plan["subset"]):
        # elbow and palm both sit on zero at the right edge, so their labels
        # would print on top of each other; stagger them vertically.
        ax.line([r["t"] for r in L], [min(r[f"F_{s}"], 158) for r in L],
                SER[i % 5], DASH[i % 5], label=s,
                label_dy={"elbow": -6.0, "palm": 6.0}.get(s, 0.0))
    return ax.svg("measured brace contact forces over time")


def certified_site_heights(mode_dir, mode, sites):
    """Height of each bracing site above the tabletop AT q*, straight from MuJoCo.

    Read from the pose file rather than back-computed from a plan report: the S12
    plan JSON predates the brace-height field entirely, and recovering the datum
    from a plan's own braced-phase mean silently re-centres each curve on itself,
    which draws the S12 defect as zero.
    """
    import contact_select as cs
    import mujoco
    m, d = cs.load(ik_margin=0.0)
    q = np.loadtxt(os.path.join(mode_dir, f"q_{mode}.txt"))
    d.qpos[:len(q)] = q
    mujoco.mj_forward(m, d)
    tz = cs.table_top_z(m, d)
    out = {}
    for s in sites:
        body, off = cs.SITES[s]
        out[s] = float(cs.point_world(m, d, body, off)[2] - tz)
    return out


def fig_brace_height(run_dir, s12_dir):
    """The S12 defect and its repair, in one panel per plan."""
    out = []
    for tag, d, title, na, T in (
            ("elbow_forearm", s12_dir, "S12 plan", 60, 2.0),
            ("s13", run_dir, "S13 plan", 120, 4.0)):
        rows = load(d, f"diag_{tag}.json")["rows"]
        plan = load(d, f"plan_{tag}.json")
        cert = certified_site_heights(d, plan["mode"], plan["subset"])
        ax = Axes((0, T), (-30, 30), w=320, h=224,
                  pad=(46, 62, 26, 34), xlabel="time [s]",
                  ylabel="site − certified [mm]")
        ax.vspan(na * plan["dt"], T, "braced")
        ax.hline(0.0, color="var(--good)", dash="4 3")
        ax.parts.append(
            f'<text x="{ax.X(0.05):.1f}" y="{ax.Y(0.0) - 4:.1f}" font-size="9" '
            f'fill="var(--good)">on the table</text>')
        # In the S13 panel all three sites land ON the datum, so their end labels
        # would stack on one another (and on the datum label); spread them.
        for i, s in enumerate(plan["subset"]):
            ax.line([r["k"] * plan["dt"] for r in rows],
                    [max(min((r[f"z_{s}"] - cert[s]) * 1000, 28), -28)
                     for r in rows],
                    SER[i % 5], DASH[i % 5], label=s,
                    label_dy=(i - 1) * 10.0 if len(plan["subset"]) > 2 else 0.0)
        out.append((title, ax.svg(f"brace site height against the certified "
                                  f"landing height, {title}")))
    return out


def fig_mpc_sweep(run_dir):
    mat = load(run_dir, "matrix.json")["mpc_sweep"]
    Hs = sorted({r["mpc_horizon"] for r in mat})
    ax = Axes((0.1, len(Hs) + 0.35), (-25, 235), w=440, h=250,
              pad=(52, 96, 26, 48), ylabel="reach error at brace end [mm]",
              xticks=[], yticks=[0, 50, 100, 150, 200])
    for it, color, dash in ((1, SER[0], "none"), (2, SER[1], "6 3")):
        pts = [(i + 0.5 + 0.34 * (it - 1), r) for i, Hh in enumerate(Hs)
               for r in mat if r["mpc_horizon"] == Hh and r["mpc_iters"] == it]
        for x, r in pts:
            v = min(r["reach_err_at_brace_end"] * 1000, 205)
            ax.dot(x, v, "var(--warn)" if r["fell"] else color)
            if r["fell"]:
                ax.text(x, v - 18, "fell", color="var(--warn)", anchor="middle")
        ax.line([x for x, _ in pts],
                [min(r["reach_err_at_brace_end"] * 1000, 205) for _, r in pts],
                color, dash, width=1.6, opacity=0.75)
        ax.parts.append(
            f'<text x="{ax.X(1.55):.1f}" y="{ax.Y(200 - 26 * (it - 1)):.1f}" '
            f'font-size="10" fill="{color}">{it} iteration'
            f'{"s" if it > 1 else ""} / step</text>')
    for i, Hh in enumerate(Hs):
        ax.parts.append(
            f'<text x="{ax.X(i + 0.67):.1f}" y="{ax.h - 22:.0f}" font-size="9" '
            f'fill="var(--ink3)" text-anchor="middle">H={Hh} '
            f'({Hh * 0.02:.1f} s)</text>')
    return ax.svg("MPC reach error against horizon length")


def fig_robust(run_dir):
    mat = load(run_dir, "matrix.json").get("robust", [])
    pushes = sorted({r.get("push") or 0.0 for r in mat
                     if not r.get("q_noise")})
    ax = Axes((-34, max(pushes) + 10), (0, 300), w=470, h=245,
              pad=(74, 24, 26, 36),
              xlabel="backward push on the pelvis, held 0.2 s at t = 1.4 s [N]",
              ylabel="reach error at brace end [mm]",
              xticks=[0, 40, 80, 120, 160])
    for i, (c, lab) in enumerate((("ff", "feedforward"), ("riccati", "riccati"),
                                  ("mpc", "MPC"))):
        rows = sorted([r for r in mat if r["ctrl"] == c and not r.get("q_noise")],
                      key=lambda r: r.get("push") or 0.0)
        xs = [r.get("push") or 0.0 for r in rows]
        ys = [min(r["reach_err_at_brace_end"] * 1000, 292) for r in rows]
        ax.line(xs, ys, SER[i % 5], DASH[i % 5], label=lab, width=1.8,
                label_at=0, label_dy={"feedforward": 13.0}.get(lab, 0.0))
        for x, y, r in zip(xs, ys, rows):
            ax.dot(x, y, "var(--warn)" if r["fell"] else SER[i % 5], r=3.0)
    return ax.svg("reach error against disturbance size, per controller")


# --------------------------------------------------------------------------- #
# (contact mode) x (reach target) sweep
# --------------------------------------------------------------------------- #
MODE_LABEL = {"legs_only": "legs only", "palm": "palm", "elbow": "elbow",
              "elbow+forearm": "elbow+forearm", "elbow+palm": "elbow+palm (arch)",
              "forearm+palm": "forearm+palm",
              "elbow+forearm+palm": "elbow+forearm+palm"}
MODE_COLOR = {"legs_only": "var(--ink3)", "palm": "var(--s3)",
              "elbow": "var(--s1)", "elbow+forearm": "var(--s4)",
              "elbow+palm": "var(--s2)", "forearm+palm": "var(--s5)",
              "elbow+forearm+palm": "var(--s5)"}
SITE_COLOR = {"foot": "var(--ink3)", "elbow": "var(--s1)", "forearm": "var(--s4)",
              "palm": "var(--s3)", "hip": "var(--s5)", "torso": "var(--s2)"}


def fig_sweep_grid(sweep):
    """The matrix itself: one cell per (mode, target), as a small status grid.

    A heat map would put a number in every cell and imply they are all the same
    kind of number; they are not.  A cell can fail for three different reasons --
    no pose exists, no force distribution exists, or the pose exists and MuJoCo
    falls over anyway -- and which one it is IS the result.  So the glyph carries
    the outcome and the number carries the reach error, and the cells that failed
    early stay blank rather than being coloured zero.
    """
    rows = sweep["rows"]
    modes = [m for m in sweep["modes"]
             if any(r["mode"] == m for r in rows)]
    targets = sweep["targets"]
    cw, ch = 92, 34
    x0, y0 = 148, 40
    w = x0 + cw * len(targets) + 16
    h = y0 + ch * len(modes) + 46
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="contact mode by '
         f'reach target outcome matrix">']
    for j, t in enumerate(targets):
        g.append(f'<text x="{x0 + cw*j + cw/2:.0f}" y="{y0 - 20}" font-size="10" '
                 f'fill="var(--ink)" text-anchor="middle">x = {t:.3f} m</text>')
    by = {(r["mode"], r["target_x"]): r for r in rows}
    for i, mode in enumerate(modes):
        yc = y0 + ch * i + ch / 2
        g.append(f'<text x="{x0 - 10}" y="{yc + 3.5:.0f}" font-size="10" '
                 f'fill="var(--ink)" text-anchor="end">'
                 f'{esc(MODE_LABEL.get(mode, mode))}</text>')
        for j, t in enumerate(targets):
            r = by.get((mode, t))
            xc = x0 + cw * j
            fill, txt, col = "var(--line)", "", "var(--ink3)"
            if r is None or not r["static_admissible"]:
                txt = r.get("static_reason", "") if r else ""
                fill, col = "none", "var(--ink3)"
            elif r.get("mpc_fell"):
                fill, col = "var(--warn)", "var(--bg)"
                txt = "toppled"
            elif "mpc_reach_mm" in r:
                fill = MODE_COLOR.get(mode, "var(--s1)")
                col = "var(--bg)"
                txt = f"{r['mpc_reach_mm']:.1f} mm"
            else:
                txt, fill = "planned", "none"
            g.append(f'<rect x="{xc + 3}" y="{y0 + ch*i + 3}" width="{cw - 6}" '
                     f'height="{ch - 6}" rx="3" fill="{fill}" '
                     f'opacity="{0.20 if fill == "var(--line)" else 0.85}" '
                     f'stroke="var(--line)" stroke-width="1"/>')
            g.append(f'<text x="{xc + cw/2:.0f}" y="{yc + 3.5:.0f}" font-size="9" '
                     f'fill="{col}" text-anchor="middle">{esc(txt)}</text>')
    g.append(f'<text x="6" y="{h - 20}" font-size="9" fill="var(--ink3)">'
             f'filled = certified, planned and replayed in MuJoCo under MPC; the '
             f'number is the reach error at the end of the braced phase.</text>')
    g.append(f'<text x="6" y="{h - 8}" font-size="9" fill="var(--ink3)">'
             f'empty = no admissible static pose, labelled with the test it '
             f'failed.</text>')
    g.append("</svg>")
    return "".join(g)


def fig_sweep_reach(sweep):
    """Reach error vs target distance, per mode -- the sweep's headline curve."""
    rows = [r for r in sweep["rows"] if "mpc_reach_mm" in r]
    modes = [m for m in sweep["modes"] if any(r["mode"] == m for r in rows)]
    ax = Axes((0.88, 1.30), (0, 60), w=520, h=250, pad=(54, 128, 26, 36),
              xlabel="reach target, x [m]", ylabel="reach error in MuJoCo [mm]",
              xticks=sweep["targets"])
    for i, mode in enumerate(modes):
        pts = sorted([r for r in rows if r["mode"] == mode],
                     key=lambda r: r["target_x"])
        xs = [r["target_x"] for r in pts]
        ys = [min(r["mpc_reach_mm"], 58) for r in pts]
        c = MODE_COLOR.get(mode, SER[i % 5])
        ax.line(xs, ys, c, DASH[i % 5], label=MODE_LABEL.get(mode, mode),
                width=1.8, label_dy=(i % 3 - 1) * 4.0)
        for x, y, r in zip(xs, ys, pts):
            ax.dot(x, y, "var(--warn)" if r.get("mpc_fell") else c, r=3.0)
    return ax.svg("MuJoCo reach error against target distance, per contact mode")


def fig_sweep_effort(sweep):
    """Static actuator effort vs target, per mode: WHY a mode runs out.

    The cost the enumeration ranks on, plotted against the axis that makes the
    task hard.  A mode that stops being admissible because its curve crosses the
    torque ceiling failed on STRENGTH; one that simply ends failed on GEOMETRY,
    and the two are different claims about the robot.
    """
    rows = sweep["rows"]
    modes = [m for m in sweep["modes"] if any(r["mode"] == m for r in rows)]
    ax = Axes((0.88, 1.30), (0, 1.30), w=520, h=250, pad=(54, 128, 26, 36),
              xlabel="reach target, x [m]", ylabel="peak |tau| / torque limit",
              xticks=sweep["targets"], yticks=[0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.hline(1.0, color="var(--warn)", dash="4 3", label="limit")
    for i, mode in enumerate(modes):
        pts = sorted([r for r in rows if r["mode"] == mode],
                     key=lambda r: r["target_x"])
        xs = [r["target_x"] for r in pts]
        ys = [r["static_ratio"] for r in pts]
        c = MODE_COLOR.get(mode, SER[i % 5])
        ax.line(xs, ys, c, DASH[i % 5], label=MODE_LABEL.get(mode, mode),
                width=1.8, label_dy=(i % 3 - 1) * 4.0)
        for x, y, r in zip(xs, ys, pts):
            ax.dot(x, y, c if r["static_admissible"] else "var(--warn)", r=3.0)
    return ax.svg("static peak torque ratio against target distance, per mode; "
                  "orange dots are inadmissible cells")


# --------------------------------------------------------------------------- #
# CoM support regions, top view (the S11 plot, applied to crocoddyl output)
# --------------------------------------------------------------------------- #
def _region_panel(g, cell, ox, oy, pw, ph, bounds, title):
    """One top-view panel into an existing SVG part list.

    World x runs RIGHT (away from the robot, toward the table) and world y UP the
    page -- the same orientation as the S11 figure, so the two can be read
    against each other without re-learning the axes.
    """
    x0, x1, y0, y1 = bounds
    sx = lambda x: ox + (x - x0) / (x1 - x0) * pw
    sy = lambda y: oy + ph - (y - y0) / (y1 - y0) * ph

    def poly(pts, stroke, dash="none", fill="none", op=1.0, width=1.6):
        if not pts:
            return
        s = " ".join(f"{sx(p[0]):.1f},{sy(p[1]):.1f}" for p in pts)
        g.append(f'<polygon points="{s}" fill="{fill}" fill-opacity="{op}" '
                 f'stroke="{stroke}" stroke-width="{width}" '
                 f'stroke-dasharray="{dash}"/>')

    g.append(f'<rect x="{ox:.1f}" y="{oy:.1f}" width="{pw:.1f}" height="{ph:.1f}" '
             f'fill="none" stroke="var(--line)" stroke-width="1"/>')
    g.append(f'<text x="{ox:.1f}" y="{oy - 5:.1f}" font-size="9.5" '
             f'fill="var(--ink)">{esc(title)}</text>')

    poly(cell["legs"]["actuated"], "var(--ink3)", "3 3",
         fill="var(--ink3)", op=0.10, width=1.2)
    col = MODE_COLOR.get(cell["mode"], "var(--s1)")
    poly(cell["certified"]["actuated"], col, "none", fill=col, op=0.13)
    if "achieved" in cell:
        poly(cell["achieved"]["actuated"], "var(--ink)", "5 3", width=1.4)

    for p in cell["certified"]["pts"]:
        c = SITE_COLOR.get(p["label"], "var(--ink3)")
        r = 1.8 if p["label"] == "foot" else 2.4 + 2.2 * min(p["fn"] / 200.0, 1.0)
        g.append(f'<circle cx="{sx(p["p"][0]):.1f}" cy="{sy(p["p"][1]):.1f}" '
                 f'r="{r:.1f}" fill="{c}" fill-opacity="0.9"/>')
    cx, cy = cell["certified"]["com"]
    g.append(f'<circle cx="{sx(cx):.1f}" cy="{sy(cy):.1f}" r="4" fill="none" '
             f'stroke="{col}" stroke-width="2"/>')
    if "achieved" in cell:
        ax_, ay_ = cell["achieved"]["com"]
        if x0 <= ax_ <= x1 and y0 <= ay_ <= y1:
            g.append(f'<line x1="{sx(cx):.1f}" y1="{sy(cy):.1f}" '
                     f'x2="{sx(ax_):.1f}" y2="{sy(ay_):.1f}" stroke="var(--ink)" '
                     f'stroke-width="1" stroke-dasharray="2 2"/>')
            g.append(f'<circle cx="{sx(ax_):.1f}" cy="{sy(ay_):.1f}" r="2.6" '
                     f'fill="var(--ink)"/>')
        else:
            # A CoM outside the frame is a toppled robot: say so rather than
            # rescale six panels around one failure.
            g.append(f'<text x="{ox + pw - 4:.1f}" y="{oy + 12:.1f}" '
                     f'font-size="8" fill="var(--warn)" text-anchor="end">'
                     f'CoM off frame</text>')
    fmt = lambda v: "n/a" if v is None else f"{1000*v:+.0f}"
    g.append(f'<text x="{ox + 4:.1f}" y="{oy + ph - 16:.1f}" font-size="8" '
             f'fill="var(--ink3)">legs {fmt(cell["legs"]["margin"])} '
             f'&#8594; certified {fmt(cell["certified"]["margin"])} mm</text>')
    if "achieved" in cell:
        got = "+".join(cell.get("achieved_subset", [])) or "no arm contact"
        col2 = ("var(--warn)" if got != "+".join(cell["subset"])
                else "var(--ink3)")
        g.append(f'<text x="{ox + 4:.1f}" y="{oy + ph - 5:.1f}" font-size="8" '
                 f'fill="{col2}">MuJoCo: {esc(got)}, '
                 f'{fmt(cell["achieved"]["margin"])} mm</text>')


def fig_regions(regions, picks, cols=3, pw=214, ph=182):
    """Small multiples of the S11 CoM-region plot, one per chosen cell.

    Each panel carries three things: the legs-only region (grey, dashed) for
    scale, the region the OFFLINE plan certified (filled), and the region of the
    pose MuJoCo actually ends the braced phase in (black dashed), with the two
    CoMs joined by a tick so the drift between them is a length rather than a
    comparison of two numbers.
    """
    cells = {(c["target_x"], c["mode"]): c for c in regions["cells"]}
    chosen = [cells[k] for k in picks if k in cells]
    if not chosen:
        raise RuntimeError("no region cells matched the requested picks")

    # Bounds from the CERTIFIED geometry only.  Including the achieved CoM puts a
    # toppled robot's centre of mass half a metre outside the extent, and every
    # panel then draws its region as a small blob in a lot of white space.
    allpts = []
    for c in chosen:
        allpts += c["legs"]["actuated"] + c["certified"]["actuated"]
        allpts += [c["certified"]["com"]]
        allpts += [p["p"][:2] for p in c["certified"]["pts"]]
    A = np.array(allpts)
    cx, cy = (A[:, 0].min() + A[:, 0].max()) / 2, (A[:, 1].min() + A[:, 1].max()) / 2
    half = max(np.ptp(A[:, 0]), np.ptp(A[:, 1])) / 2 * 1.10
    bounds = (cx - half, cx + half, cy - half, cy + half)

    rows = (len(chosen) + cols - 1) // cols
    gapx, gapy = 22, 34
    w = 12 + cols * (pw + gapx)
    h = 20 + rows * (ph + gapy)
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="static-equilibrium '
         f'CoM regions, certified against achieved, per contact mode and target">']
    for i, c in enumerate(chosen):
        ox = 12 + (i % cols) * (pw + gapx)
        oy = 22 + (i // cols) * (ph + gapy)
        title = f"x={c['target_x']:.3f}  {MODE_LABEL.get(c['mode'], c['mode'])}"
        _region_panel(g, c, ox, oy, pw, ph, bounds, title)
    g.append("</svg>")
    return "".join(g)


def fig_region_margin(regions):
    """Certified vs achieved CoM margin, per cell -- the region plot as a number.

    The panels show six cells; this shows all of them, and it is where the
    systematic part lives: whether MuJoCo's pose keeps the margin the offline
    plan certified is a question about the whole sweep, not about the six cells
    that fit on a page.
    """
    cells = [c for c in regions["cells"] if "achieved" in c
             and c["achieved"]["margin"] is not None
             and c["certified"]["margin"] is not None]
    ax = Axes((0, 260), (0, 260), w=380, h=300, pad=(56, 28, 26, 40),
              xlabel="certified CoM margin at q* [mm]",
              ylabel="CoM margin of the pose MuJoCo reaches [mm]",
              xticks=[0, 50, 100, 150, 200, 250],
              yticks=[0, 50, 100, 150, 200, 250])
    ax.parts.append(f'<line x1="{ax.X(0):.1f}" y1="{ax.Y(0):.1f}" '
                    f'x2="{ax.X(260):.1f}" y2="{ax.Y(260):.1f}" '
                    f'stroke="var(--line)" stroke-width="1" '
                    f'stroke-dasharray="4 3"/>')
    ax.parts.append(f'<text x="{ax.X(196):.1f}" y="{ax.Y(214):.1f}" '
                    f'font-size="9" fill="var(--ink3)">y = x</text>')
    for c in cells:
        col = MODE_COLOR.get(c["mode"], "var(--s1)")
        ax.dot(1000 * c["certified"]["margin"], 1000 * c["achieved"]["margin"],
               col, r=3.4)
    return ax.svg("achieved against certified CoM support margin, one dot per "
                  "mode and target")


# Six cells that carry the argument: the easy target where the brace is
# decoration, the middle target where it starts to matter, and the two far ones
# where the legs-only region has gone NEGATIVE and a brace is the only thing
# holding the pose.
REGION_PICKS = [(0.905, "legs_only"), (0.905, "elbow+forearm"),
                (1.050, "palm"), (1.050, "elbow+forearm+palm"),
                (1.150, "elbow"), (1.250, "elbow+palm")]


def fig_region_duty(regions):
    """How much of the braced phase each planned contact was actually loaded.

    The region plots are about the pose; this is about whether the contact that
    pose assumes ever happens.  Duty is the fraction of braced nodes on which
    MuJoCo puts more than 5 N through the site, so 100% is a brace that is there
    for the whole phase and 0% is a plan drawing force through thin air.
    """
    rows = []
    for c in regions["cells"]:
        for s, v in (c.get("contact_duty") or {}).items():
            rows.append((c["target_x"], c["mode"], s, v))
    if not rows:
        raise RuntimeError("no contact_duty in regions.json")
    w = 560
    rowh = 17
    h = 46 + rowh * len(rows)
    bx, bw = 268, 200
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="fraction of the '
         f'braced phase each planned contact actually carried load">']
    for i, (x, mode, site, v) in enumerate(rows):
        y = 30 + i * rowh
        g.append(f'<text x="{bx - 8}" y="{y + 9}" font-size="9" fill="var(--ink)" '
                 f'text-anchor="end">x={x:.3f} · {esc(MODE_LABEL.get(mode, mode))} '
                 f'· {esc(site)}</text>')
        g.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="11" rx="2" '
                 f'fill="var(--line)" opacity="0.35"/>')
        col = SITE_COLOR.get(site, "var(--s1)")
        g.append(f'<rect x="{bx}" y="{y}" width="{max(bw * v, 1.2):.1f}" '
                 f'height="11" rx="2" fill="{col}" opacity="0.9"/>')
        g.append(f'<text x="{bx + bw + 6}" y="{y + 9}" font-size="9" '
                 f'fill="var(--ink3)">{v:.0%}</text>')
    g.append(f'<text x="{bx}" y="20" font-size="9" fill="var(--ink3)">'
             f'fraction of braced nodes with &gt; 5 N through the site</text>')
    g.append("</svg>")
    return "".join(g)


def fig_keepout_bench(bench):
    """C++ against Python for the keep-out activation, as a two-bar comparison."""
    py, cc = bench["python"], bench["cpp"]
    items = [("problem calcDiff\n(200-node sweep)", py["calcdiff_ms"], cc["calcdiff_ms"]),
             ("MPC step\n(H=50, 1 iteration)", py["mpc_ms"], cc["mpc_ms"])]
    w, h = 460, 190
    top = max(max(a, b) for _, a, b in items) * 1.18
    g = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="keep-out activation '
         f'cost, Python against C++">']
    bx, by, bw, bh = 176, 26, 250, 44
    for i, (lab, a, b) in enumerate(items):
        y = by + i * (bh + 42)
        head = lab.split("\n")
        g.append(f'<text x="{bx - 10}" y="{y + 12}" font-size="9.5" '
                 f'fill="var(--ink)" text-anchor="end">{esc(head[0])}</text>')
        g.append(f'<text x="{bx - 10}" y="{y + 24}" font-size="8.5" '
                 f'fill="var(--ink3)" text-anchor="end">{esc(head[1])}</text>')
        for j, (v, col, name) in enumerate(((a, "var(--ink3)", "python"),
                                            (b, "var(--s1)", "c++"))):
            yy = y + j * 20
            g.append(f'<rect x="{bx}" y="{yy}" width="{bw * v / top:.1f}" '
                     f'height="15" rx="2" fill="{col}" opacity="0.85"/>')
            g.append(f'<text x="{bx + bw * v / top + 6:.1f}" y="{yy + 12}" '
                     f'font-size="9" fill="var(--ink3)">{v:.0f} ms  {name}</text>')
    g.append(f'<line x1="{bx + bw * 20.0 / top:.1f}" y1="{by - 8}" '
             f'x2="{bx + bw * 20.0 / top:.1f}" y2="{h - 22}" '
             f'stroke="var(--warn)" stroke-width="1" stroke-dasharray="3 3"/>')
    g.append(f'<text x="{bx + bw * 20.0 / top + 5:.1f}" y="{h - 10}" '
             f'font-size="9" fill="var(--warn)">20 ms control period</text>')
    g.append("</svg>")
    return "".join(g)


def inject(html_path, figs):
    with open(html_path) as fh:
        s = fh.read()
    for name, svg in figs.items():
        s = s.replace(f"<!--FIG:{name}-->", svg)
    with open(html_path, "w") as fh:
        fh.write(s)
    missing = [n for n in figs if f"<!--FIG:{n}-->" in s]
    return missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-06_session13")
    ap.add_argument("--s12-dir", default="runs/2026-08-05_session12/croco")
    ap.add_argument("--sweep", default="runs/2026-08-06_session13/sweep")
    ap.add_argument("--out", default="../docs/lean/media/figs_2026-08-06.json")
    ap.add_argument("--inject", default=None)
    args = ap.parse_args()

    ctrls = ["hold", "position", "ff", "riccati", "mpc"]
    labels = ["hold", "position", "feedforward", "riccati", "MPC"]
    figs = {}
    figs["key"] = key(ctrls, labels)
    figs["ladder"] = fig_ladder(args.dir, ctrls, labels)
    figs["reach"] = fig_reach(args.dir, ctrls, labels)
    figs["margin"] = fig_margin(args.dir, ctrls, labels)
    figs["brace"] = fig_brace(args.dir)
    for (title, svg), name in zip(fig_brace_height(args.dir, args.s12_dir),
                                  ("bh_s12", "bh_s13")):
        figs[name] = svg
    try:
        figs["mpc_sweep"] = fig_mpc_sweep(args.dir)
    except Exception as e:                                   # noqa: BLE001
        print(f"  mpc_sweep skipped: {e}")
    try:
        figs["robust"] = fig_robust(args.dir)
    except Exception as e:                                   # noqa: BLE001
        print(f"  robust skipped: {e}")

    # ---- S13b: the C++ keep-out, the (mode x target) sweep, the CoM regions --
    for name, fn in (
            ("keepout_bench", lambda: fig_keepout_bench(
                load(args.dir, "bench_keepout.json"))),
            ("sweep_grid", lambda: fig_sweep_grid(load(args.sweep, "sweep.json"))),
            ("sweep_reach", lambda: fig_sweep_reach(load(args.sweep, "sweep.json"))),
            ("sweep_effort", lambda: fig_sweep_effort(load(args.sweep, "sweep.json"))),
            ("regions", lambda: fig_regions(load(args.sweep, "regions.json"),
                                            REGION_PICKS)),
            ("region_margin", lambda: fig_region_margin(
                load(args.sweep, "regions.json"))),
            ("region_duty", lambda: fig_region_duty(
                load(args.sweep, "regions.json"))),
    ):
        try:
            figs[name] = fn()
        except Exception as e:                               # noqa: BLE001
            print(f"  {name} skipped: {e}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(figs, fh)
    print(f"wrote {args.out}  ({len(figs)} figures)")
    if args.inject:
        missing = inject(args.inject, figs)
        print(f"injected into {args.inject}"
              + (f"; MISSING placeholders for {missing}" if missing else ""))


if __name__ == "__main__":
    main()
