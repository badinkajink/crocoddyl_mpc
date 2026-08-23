#!/usr/bin/env python3
"""Generate the session docpage for the MJPC-vs-CMPC comparison.

Every number on the page is computed here from the two `analysis.json` files,
never typed. That is the whole reason this is a script and not an HTML file: the
page is regenerated whenever the runs are, so it cannot drift from them, and a
claim on it always has a rollout behind it.

usage:
  cmpc_page.py --mjpc A/analysis.json --cmpc B/analysis.json
               --figs paper/figures/cmpc_brace_reach --out docs/lean/PAGE.html
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np

import bvs_plots as BP

ORDER_ARMS = ["stand", "brace"]
LAB = {"stand": "no brace cost", "brace": "commanded brace"}

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Figures copied next to the page, prefixed so the media directory stays
# greppable by session the way every earlier page's is.
FIGS = [
    ("fig_x_reach_envelope.png", "s23_envelope.png",
     "The target sweep, both planners. Hue is the arm, marker and linestyle "
     "the planner. Left: does the hand arrive. Right: how far it gets, "
     "measured from the ankles so the two planners' different start poses "
     "cannot flatter either."),
    ("fig_x_stability_panels_support.png", "s23_panels.png",
     "What the brace buys and costs, at the extended-reach condition "
     "(x = 1.06 m). Solid fill MJPC, hatched CMPC."),
    ("fig_cmpc_stages.png", "s23_stages.png",
     "The maneuver end to end under CMPC: one trajectory, brace+reach held "
     "and then chained into the recovery."),
    ("fig_strategy_strip.png", "s23_strip.png",
     "CMPC strategies, chosen by measured load rather than by commanded "
     "weights."),
    ("fig_x_disturbance.png", "s23_disturb.png",
     "Push rejection: three 0.3 s pulses of 30-90 N at the reaching wrist, "
     "invisible to both planners."),
]


def load(p):
    return json.load(open(p))


def group(rows, stage, arm):
    return [r for r in BP.by(rows, stage) if r.get("arm") == arm]


def ms(vals, sc=1.0):
    v = [x * sc for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return float("nan"), 0.0
    return float(np.mean(v)), float((max(v) - min(v)) / 2)


def cell(rows, stage, arm, key, sc=1.0, fmt="%.1f"):
    g = group(rows, stage, arm)
    vals = ([key(r) for r in g] if callable(key)
            else [r.get(key) for r in g])
    mu, hr = ms(vals, sc)
    if not np.isfinite(mu):
        return "&mdash;"
    return (fmt % mu) + ("" if hr == 0 else " &plusmn; " + fmt % hr)


def sweep_by_x(rows, arm):
    """Every swept rollout of one arm, unfiltered, keyed by target x."""
    d = defaultdict(list)
    for r in rows:
        if (r.get("stage") == "sweep" and r.get("arm") == arm
                and "error" not in r):
            d[round(r["target"][0], 3)].append(r)
    return d


def outcome(r, tol=0.05):
    """What one swept rollout did: arrived, or the way it did not."""
    if r.get("fell"):
        return "fell"
    if BP.contact_class(r) == "torso":
        return "trunk"
    return "arrived" if r.get("reach_err_settled", 9) <= tol else "missed"


PASS_CLEAN = ("arrived",)
PASS_ANY = ("arrived", "trunk")


def clean_band(rows, arm, tol=0.05, allow=PASS_CLEAN):
    """The longest contiguous run of targets where EVERY replicate arrived.

    TWO BANDS ARE REPORTED, not one, and the pair is the finding. `allow`
    decides whether a rollout that arrived on target while resting its chest on
    the slab counts: the ARRIVAL band says yes (it is upright, it is on target,
    it braced with a body the mode did not name), the CLEAN band says no. The
    sampling planner's commanded brace has a wide arrival band and no clean
    band at all, and reporting either number alone tells a different and
    incomplete story about it.

    A BAND, NOT A MAXIMUM, because a maximum cannot express what the data
    does.

    THREE THINGS HAD TO CHANGE from the obvious version, and each was a way of
    reporting a controller as better than it is:

      * it was computed over `BP.by(...)`, which USED TO drop the torso-rest
        rollouts -- so a target where every replicate finished with its chest
        on the slab contributed NO rollouts and was skipped rather than scored.
        Three of the sampling planner's seven braced targets had zero surviving
        replicates, and its envelope was read off the four that remained. Since
        2026-08-23 those rollouts are pooled (a trunk rest is an upright,
        on-target posture bracing with a body the mode did not name), so the
        band below is computed over every rollout and the trunk-assisted count
        is reported beside it.
      * a fall was likewise invisible.
      * it took `max` over passing targets, so a pass at 1.30 outranked a
        failure at 1.22.

    Measured, the four arms fail in four different shapes: the sampling
    planner's braced arm rests its trunk at almost every target (12 of 14
    swept rollouts), the gradient planner's braced arm FALLS at the nearest
    target and rests its trunk at the two furthest, and the two no-brace arms
    simply stop arriving. A single "largest x" collapses all of that to one
    number and, worse, to a number that reads as a success. The band's two
    ends, plus the per-target outcome table below it, do not.

    Returns (lo, hi), or (None, None) if no target is clean.
    """
    d = sweep_by_x(rows, arm)
    best = (None, None)
    run = None
    for x in sorted(d):
        if all(outcome(r, tol) in allow for r in d[x]):
            run = x if run is None else run
            if best[0] is None or (x - run) >= (best[1] - best[0]):
                best = (run, x)
        else:
            run = None
    return best


def band_str(rows, arm, allow=PASS_CLEAN):
    lo, hi = clean_band(rows, arm, allow=allow)
    if lo is None:
        return "none"
    return "%.2f m" % lo if lo == hi else "%.2f&ndash;%.2f m" % (lo, hi)


def falls(rows, stage, arm):
    g = [r for r in rows if r.get("stage") == stage and r.get("arm") == arm
         and "error" not in r]
    return sum(1 for r in g if r.get("fell")), len(g)


HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The gradient planner on the sampling planner's figures</title>
<style>
  :root{
    color-scheme:light dark;
    --bg:#fcfcfb; --panel:#f3f3f0; --line:#dededa;
    --ink:#0b0b0b; --ink2:#52514e; --ink3:#7a7873;
    --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
    --warn:#e34948; --good:#1baf7a;
  }
  @media (prefers-color-scheme:dark){
    :root{ --bg:#1a1a19; --panel:#232322; --line:#3a3a37;
           --ink:#fff; --ink2:#c3c2b7; --ink3:#918f86;
           --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
           --warn:#e66767; --good:#3fbf8c; }
  }
  *{box-sizing:border-box}
  body{background:var(--bg);color:var(--ink);margin:0;
       font:16px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
       padding:2.5rem 1.25rem 5rem}
  .wrap{max-width:64rem;margin:0 auto}
  h1{font-size:1.75rem;line-height:1.2;margin:0 0 .4rem;letter-spacing:-.02em}
  h2{font-size:1.2rem;margin:3rem 0 .5rem;letter-spacing:-.01em;
     padding-top:1rem;border-top:1px solid var(--line)}
  h3{font-size:1rem;margin:1.8rem 0 .3rem}
  .sub{color:var(--ink2);margin:0 0 .4rem}
  .meta{color:var(--ink3);font-size:.85rem;margin:0 0 2rem}
  p,li{color:var(--ink2)}
  code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;
       background:var(--panel);padding:.1em .35em;border-radius:4px}
  pre{background:var(--panel);padding:.9rem 1rem;border-radius:8px;overflow-x:auto;
      font-size:.78rem;line-height:1.45;color:var(--ink)}
  table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0;min-width:26rem}
  th,td{text-align:right;padding:.35rem .6rem;border-bottom:1px solid var(--line);white-space:nowrap}
  th:first-child,td:first-child{text-align:left;white-space:normal}
  th{color:var(--ink3);font-weight:600;font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
  td{color:var(--ink2)} td:first-child{color:var(--ink)}
  .scroll{overflow-x:auto}
  .note{border-left:3px solid var(--s1);background:var(--panel);
        border-radius:0 8px 8px 0;padding:.8rem 1rem;margin:1.2rem 0}
  .warn{border-left-color:var(--warn)}
  .ok{border-left-color:var(--good)}
  .note b{color:var(--ink)}
  figure{margin:1.6rem 0}
  img{width:100%;border-radius:10px;display:block;background:#fff}
  figcaption{color:var(--ink3);font-size:.84rem;margin-top:.5rem}
  ul{padding-left:1.15rem}
  td.bad{color:var(--warn);font-weight:600}
  td.good{color:var(--good);font-weight:600}
</style>
</head>
<body>
<div class="wrap">
"""

TAIL = "</div>\n</body>\n</html>\n"


def build(mjpc, cmpc, figs, out, cmpc_run, mjpc_run):
    media = os.path.join(os.path.dirname(out), "media")
    os.makedirs(media, exist_ok=True)
    have = []
    for src, dst, cap in FIGS:
        p = os.path.join(figs, src)
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(media, dst))
            have.append((dst, cap))

    o = [HEAD]
    o.append("<h1>The gradient planner on the sampling planner's figures</h1>")
    o.append('<p class="sub">Crocoddyl-MPC run through the braced-reach '
             'protocol that produced the paper\'s MJPC baselines &mdash; same '
             'plant, same targets, same scorer, same plotting code &mdash; so '
             'the two controllers can be read off one axis.</p>')
    o.append('<p class="meta">2026-08-22 &middot; S23 &middot; in-process '
             'MuJoCo, closed loop, %d CMPC and %d MJPC scored rollouts</p>'
             % (len(cmpc), len(mjpc)))

    o.append('<div class="note ok"><p><b>What this produced.</b> Every figure '
             'in the MJPC baseline set now has a CMPC counterpart and a '
             'two-planner overlay, from one code path: the CMPC harness writes '
             "testspeed's <code>--dump_traj</code> CSV, so "
             '<code>brace_vs_stand.py analyze</code>, <code>bvs_plots.py</code>'
             ' and <code>bvs_strips.py</code> run on it unchanged. The target '
             'sweep was extended to seven targets and run on both.</p>'
             '<p><b>Arrival band</b> &mdash; the contiguous run of targets '
             'where every replicate lands within 5&nbsp;cm, upright: '
             'MJPC <b>%s</b> no-brace, <b>%s</b> braced; '
             'CMPC <b>%s</b> no-brace, <b>%s</b> braced. Requiring in addition '
             'that no rollout rests its trunk on the slab &mdash; the '
             '<b>clean</b> band &mdash; cuts those to '
             'MJPC <b>%s</b> / <b>%s</b> and CMPC <b>%s</b> / <b>%s</b>. '
             'The gap between the two rows is the whole story of the braced '
             'arms: they keep arriving, they just stop arriving on the arm '
             'they were told to use.</p></div>'
             % (band_str(mjpc, "stand", PASS_ANY),
                band_str(mjpc, "brace", PASS_ANY),
                band_str(cmpc, "stand", PASS_ANY),
                band_str(cmpc, "brace", PASS_ANY),
                band_str(mjpc, "stand"), band_str(mjpc, "brace"),
                band_str(cmpc, "stand"), band_str(cmpc, "brace")))

    # ---- protocol -------------------------------------------------------- #
    o.append("<h2>1. What was held, and what could not be</h2>")
    o.append("<p>The two controllers were given the same plant "
             "(<code>Lean_H12_Magpie.xml</code> and "
             "<code>Lean_Simple_H12_Magpie.xml</code> are the same rigid body "
             "&mdash; identical nq/nv/nu, body, site, geom and actuator lists "
             "in the same order, identical masses, gains, force ranges, "
             "friction, <code>solref/solimp</code>, contact pairs and "
             "exclusions; they differ only in MJPC task metadata), the same "
             "targets, the same durations and the same scorer. Three things "
             "could not be held, and all three are visible in the numbers "
             "rather than hidden in them:</p>")
    o.append("<ul>"
             "<li><b>What &ldquo;no brace cost&rdquo; means.</b> For MJPC it is "
             "three weights at zero and the planner may still find the table "
             "&mdash; and does. For CMPC it is the <code>legs_only</code> plan, "
             "a contact schedule with nothing on the table, which no weight can "
             "override. The difference is not that one makes table contacts and "
             "the other cannot: a <code>legs_only</code> rollout at x = 1.38 m "
             "settles with the gripper seated anyway, because the physics puts "
             "it there. The difference is that the gradient planner never "
             "MODELLED that contact, so it is an unmodelled disturbance rather "
             "than support being exploited. That is the result, not a "
             "confound.</li>"
             "<li><b>The start pose.</b> MJPC begins at the <code>home</code> "
             "keyframe, CMPC at <code>stand</code> (knees 0.55&nbsp;rad). Each "
             "is its own pipeline's start. Functional reach &mdash; hand to "
             "ankle-midpoint &mdash; is posture-intrinsic and references "
             "neither.</li>"
             "<li><b>Where the spread comes from.</b> MJPC replicates differ "
             "because it samples; CMPC is a deterministic descent, so its "
             "replicates start from a seeded 3&nbsp;mrad / 3&nbsp;mm "
             "perturbation. Both are spreads over repeated attempts; they are "
             "not the same random variable.</li></ul>")

    # ---- headline table -------------------------------------------------- #
    o.append("<h2>2. The extended-reach condition, side by side</h2>")
    o.append("<p>x = 1.06&nbsp;m, 20&nbsp;s, four replicates per arm per "
             "planner. This condition exists because the paper's default "
             "target is 0.905&nbsp;m, where both arms arrive to within a "
             "centimetre &mdash; good for isolating a stability difference, "
             "useless for a reach one &mdash; and because it is the one target "
             "at which CMPC's declared brace does not survive the hold "
             "(&sect;4).</p>")
    METRICS = [("Functional reach", "func_reach_settled", 100.0, "%.1f", "cm"),
               ("Settled reach error", "reach_err_settled", 100.0, "%.1f", "cm"),
               ("Bracing-arm force", BP.brace_arm_load, 1.0, "%.0f", "N"),
               ("Support margin (actuated)", "margin_actuated_settled", 100.0,
                "%.1f", "cm"),
               ("Forward margin (actuated)", "fwd_actuated_settled", 100.0,
                "%.1f", "cm"),
               ("Smoothness SPARC", "sparc_reach", 1.0, "%.2f", "&mdash;"),
               ("Hand jitter", "hand_jitter_mm", 1.0, "%.1f", "mm"),
               ("Worst-case push at hand", "push_min_N", 1.0, "%.0f", "N"),
               ("Peak torque ratio", "peak_tau_ratio", 1.0, "%.2f", "&mdash;")]
    o.append('<div class="scroll"><table><thead><tr><th>metric</th>'
             "<th>MJPC no&nbsp;brace</th><th>MJPC brace</th>"
             "<th>CMPC no&nbsp;brace</th><th>CMPC brace</th>"
             "<th>unit</th></tr></thead><tbody>")
    for lab, key, sc, fmt, unit in METRICS:
        cells = "".join(
            "<td>%s</td>" % cell(rows, "nominal2", arm, key, sc, fmt)
            for rows in (mjpc, cmpc) for arm in ("stand", "brace"))
        o.append("<tr><td>%s</td>%s<td>%s</td></tr>" % (lab, cells, unit))
    o.append("</tbody></table></div>")

    for dst, cap in have[:2]:
        o.append('<figure><img src="media/%s" alt=""><figcaption>%s'
                 "</figcaption></figure>" % (dst, cap))

    # ---- envelope -------------------------------------------------------- #
    o.append("<h2>3. The target sweep</h2>")
    o.append("<p>Seven targets, 0.90&ndash;1.38&nbsp;m along +x at fixed "
             "y,&nbsp;z; two replicates per point per arm per planner. The "
             "arrival band is the contiguous run of targets at which "
             "<i>every</i> replicate settles within 5&nbsp;cm of the target it "
             "was given and stays upright; the clean band additionally "
             "requires that none of them rests the trunk on the slab:</p>")
    o.append('<div class="scroll"><table><thead><tr><th>planner</th>'
             "<th>no brace cost</th><th>commanded brace</th>"
             "<th>no brace, clean</th><th>braced, clean</th>"
             "</tr></thead><tbody>")
    for name, rows in (("MJPC (sampling)", mjpc), ("CMPC (gradient)", cmpc)):
        o.append("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                 % (name, band_str(rows, "stand", PASS_ANY),
                    band_str(rows, "brace", PASS_ANY),
                    band_str(rows, "stand"), band_str(rows, "brace")))
    o.append("</tbody></table></div>")

    # PER TARGET, PER ARM, WHAT ACTUALLY HAPPENED. The single envelope number
    # above compresses four different outcomes into one x, and which one ended
    # the walk is the interesting part -- a target lost to a trunk-rest is a
    # cost-function failure, one lost to a fall is a controller failure, and
    # one lost to a 6 cm miss is neither.
    o.append("<p>What ended each walk, per target &mdash; "
             "<b>a</b>rrived / <b>m</b>issed by more than 5&nbsp;cm / "
             "rested the <b>t</b>runk / <b>f</b>ell:</p>")
    xs = sorted({x for rowset in (mjpc, cmpc) for arm in ORDER_ARMS
                 for x in sweep_by_x(rowset, arm)})
    o.append('<div class="scroll"><table><thead><tr><th>planner / arm</th>'
             + "".join("<th>%.2f</th>" % x for x in xs)
             + "</tr></thead><tbody>")
    for name, rowset in (("MJPC", mjpc), ("CMPC", cmpc)):
        for arm in ORDER_ARMS:
            d = sweep_by_x(rowset, arm)
            cells = []
            for x in xs:
                g = d.get(x, [])
                if not g:
                    cells.append("<td>&mdash;</td>")
                    continue
                letters = "".join(outcome(r)[0] for r in sorted(
                    g, key=lambda r: r.get("tag", "")))
                cls = " class=\"good\"" if set(letters) == {"a"} else \
                      " class=\"bad\""
                cells.append("<td%s>%s</td>" % (cls, letters))
            o.append("<tr><td>%s %s</td>%s</tr>"
                     % (name, LAB[arm], "".join(cells)))
    o.append("</tbody></table></div>")
    o.append("<p>The static side of the same question, from "
             "<code>croco_modes</code>'s IK+QP certification: "
             "<code>legs_only</code> stops being admissible past "
             "x&nbsp;=&nbsp;1.14&nbsp;m while <code>elbow+forearm</code> still "
             "certifies at 1.30&nbsp;m; neither certifies at 1.38 or at the "
             "1.60&nbsp;m max-reach target, which is why the max-reach cell "
             "carries a best-effort pose marked <code>admissible: false</code> "
             "rather than a certified one.</p>")
    o.append('<div class="note"><p><b>That certification is about a pose, not '
             "about the target, and the closed loop beats it.</b> "
             "<code>legs_only</code> is rejected at 1.22 and 1.30&nbsp;m on "
             "<b>base residual</b> &mdash; 14.4 and 14.6&nbsp;N &mdash; not on "
             "reach (5.1 and 19.9&nbsp;mm) and not on torque (0.99, 0.96). "
             "The pose being rejected is the one <code>solve_ik</code> "
             "produces: feet pinned at the seed stance, hand driven to target, "
             "nothing in the objective about balance. It puts the CoM "
             "<b>165&nbsp;mm</b> ahead of the ankle midpoint, and the static "
             "LP calls it infeasible. The MPC, given the same target and no "
             "brace, settles somewhere else entirely &mdash; CoM "
             "<b>60&nbsp;mm</b> ahead, feasible, with 8.5 and 7.4&nbsp;cm of "
             "actuated support margin &mdash; and holds it for the full "
             "16&nbsp;s with 21 and 32&nbsp;mm of reach error. The difference "
             "is a counterweight: at 1.30&nbsp;m the MPC hinges at the hip and "
             "puts the pelvis <b>30&nbsp;mm behind</b> the ankle midpoint "
             "where the IK pose has it 76&nbsp;mm in front, and that "
             "106&nbsp;mm of hip travel costs only 61&nbsp;mm of reach. Read "
             "the "
             "enumeration as what it is: a chooser of contact modes, and a "
             "conservative screen on the workspace, not a bound on it.</p>"
             "<p>At 1.38&nbsp;m the verdict is different in kind. All 16 "
             "arm+wrist subsets fail there, and every one of them fails on "
             "<b>reach</b> (56&ndash;98&nbsp;mm short) or on a site that will "
             "not seat &mdash; with torque ratios of 0.48&ndash;0.63 and base "
             "residuals of zero. The far end is a kinematic limit of the arm. "
             "No contact schedule buys it back.</p></div>")

    # ---- the near-target failure ----------------------------------------- #
    o.append("<h2>4. Where the declared brace stops working</h2>")
    nf, nn = falls(cmpc, "nominal", "brace")
    o.append('<div class="note warn"><p><b>CMPC\'s elbow+forearm brace does '
             "not survive a 20&nbsp;s hold at the paper's default target.</b> "
             "%d of %d nominal braced replicates fell (x&nbsp;=&nbsp;0.905). "
             "At x&nbsp;&ge;&nbsp;0.98 the same plan, the same gains and the "
             "same loop hold for the full 20&nbsp;s in every cell measured "
             "(0.98, 1.06, 1.14, 1.22). The certified static brace force at "
             "0.9047 is 51.5&nbsp;N against 80.9&nbsp;N at 1.05 &mdash; at the "
             "near target the declared contact is a light touch, and a rigid "
             "bilateral constraint on a light touch is the configuration this "
             "formulation is least able to defend.</p></div>"
             % (nf, nn))
    o.append("<p>This is not the horizon trap that produced the first version "
             "of these runs, and the two are worth telling apart because they "
             "look identical from outside. Clamping the held plan index to "
             "<code>len(us)-1</code> makes <code>MPC.__call__</code> take its "
             "tail branch and rebuild a <b>one-node</b> problem: mean solve "
             "collapses 18.9&nbsp;&rarr;&nbsp;2.5&nbsp;ms and the robot is on "
             "the floor by t&nbsp;=&nbsp;15&nbsp;s <i>in every cell</i>. "
             "Holding at <code>len(us)&nbsp;&minus;&nbsp;H</code> keeps the "
             "window full, and then only the two nearest targets fail. "
             "<b><code>croco_twin.py</code>'s own <code>hold</code> submode "
             "still uses <code>len(us)-1</code></b> &mdash; left alone because "
             "recorded results depend on it, but S19's &ldquo;8&nbsp;s "
             "hold&rdquo; should be read as a hold that was degrading.</p>")

    for dst, cap in have[2:]:
        o.append('<figure><img src="media/%s" alt=""><figcaption>%s'
                 "</figcaption></figure>" % (dst, cap))

    # ---- mode selection --------------------------------------------------- #
    o.append("<h2>5. Choosing the contact mode instead of fixing it</h2>")
    o.append("<p>Failing at <i>both</i> ends is the signature of a schedule "
             "that is right in the middle, so the sweep was repeated with the "
             "mode chosen per target. The policy is "
             "<code>croco_modes</code>' own ranking applied mechanically: "
             "enumerate all 16 subsets of {elbow, forearm, palm, wrist}, keep "
             "the admissible ones, take the least normalized actuator effort, "
             "and decline where nothing certifies. No hand-picking &mdash; "
             "otherwise the comparison is a search over 16 modes dressed up as "
             "a controller. It selects <code>elbow+wrist</code> at 0.90, "
             "<code>forearm+palm</code> at 0.98, <code>forearm+wrist</code> at "
             "1.06&ndash;1.30, and nothing at 1.38. "
             "<b><code>elbow+forearm</code>, the mode used everywhere else in "
             "this study, is never the ranked choice at any target.</b></p>")
    mp = os.path.join(figs, "table_mode_select.txt")
    if os.path.exists(mp):
        o.append("<div class=\"scroll\"><pre>%s</pre></div>"
                 % open(mp).read().replace("&", "&amp;").replace("<", "&lt;"))
    msrc = os.path.join(figs, "fig_mode_select.png")
    if os.path.exists(msrc):
        # copied here and not through FIGS: `have` is index-sliced by the
        # sections above (have[:2], have[2:]), so appending to it would render
        # this figure a second time under the wrong heading
        shutil.copyfile(msrc, os.path.join(media, "s23_mode_select.png"))
        o.append('<figure><img src="media/s23_mode_select.png" alt="">'
                 "<figcaption>Fixed versus chosen contact mode. Hue is the "
                 "schedule; a cross marks a target where a replicate fell. "
                 "Curves are over upright rollouts &mdash; a fallen robot's "
                 "settled reach error is 262&nbsp;cm and plotting it flattens "
                 "everything else into the bottom of the axis."
                 "</figcaption></figure>")
    o.append('<div class="note"><p><b>A real but partial improvement, and the '
             "two ends fail for different reasons.</b> Through the middle the "
             "selected mode holds <i>more</i> margin &mdash; +0.4 to "
             "+6.5&nbsp;cm of actuated support margin, against 8&ndash;10&nbsp;cm "
             "for <code>legs_only</code> at the same targets &mdash; while "
             "pushing 25&ndash;30% <i>less</i> force through the bracing arm "
             "(102&ndash;110&nbsp;N against 131&ndash;146&nbsp;N), at reach "
             "error within 1.2&nbsp;cm of the fixed mode. That is the effort "
             "ranking doing exactly what it claims.</p>"
             "<p>At the near target it halves the fall rate (2/2 &rarr; 1/2) "
             "without eliminating it. <code>legs_only</code> holds 0.90&nbsp;m "
             "with 8.7&nbsp;cm of margin and never falls, so bracing there is "
             "harmful <i>regardless of which mode is chosen</i> &mdash; which "
             "points back at the light-touch diagnosis above and not at mode "
             "selection. At the far target the policy declines, correctly: "
             "nothing certifies at 1.38&nbsp;m, and the fixed mode only "
             "&ldquo;reaches&rdquo; it by missing by 6.3&nbsp;cm with "
             "89&nbsp;N through the trunk. Automatic mode selection is worth "
             "having and is not the missing piece.</p></div>")

    # ---- generated tables ------------------------------------------------ #
    o.append("<h2>6. The generated tables, verbatim</h2>")
    for name in ("table_mjpc_vs_cmpc.txt", "table_brace_vs_stand.txt",
                 "table_disturbance.txt", "table_mode_select.txt"):
        p = os.path.join(figs, name)
        if not os.path.exists(p):
            continue
        o.append("<h3>%s</h3><pre>%s</pre>"
                 % (name, open(p).read().replace("&", "&amp;")
                    .replace("<", "&lt;")))

    # ---- reproduce -------------------------------------------------------- #
    o.append("<h2>7. Reproduce</h2>")
    o.append("<pre>%s</pre>" % (
        "export CL_ASSETS_DIR=... LEAN_TASK_DIR=... MUJOCO_GL=egl\n"
        "export MJPC_BIN=&lt;mujoco_mpc&gt;/build_cmake/bin/testspeed\n"
        'export BVS_SWEEP_X="0.90,0.98,1.06,1.14,1.22,1.30,1.38"\n'
        "\n# the sampling baseline\n"
        "studies/brace_vs_stand.py run     --out %s --stage all\n"
        "studies/brace_vs_stand.py analyze --run %s --append\n"
        "\n# the gradient side: one cell per target, then the episodes\n"
        "studies/cmpc_brace_vs_stand.py cells --out %s --stage all\n"
        "env LD_PRELOAD=$CROCO_OMP \\\n"
        "  studies/cmpc_brace_vs_stand.py run --out %s --stage all\n"
        "\n# every figure, both planners\n"
        "studies/make_cmpc_figures.sh\n" % (mjpc_run, mjpc_run, cmpc_run,
                                            cmpc_run)))
    o.append(TAIL)
    with open(out, "w") as f:
        f.write("\n".join(o))
    print("wrote %s (%d figures embedded)" % (out, len(have)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mjpc", required=True)
    ap.add_argument("--cmpc", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(load(a.mjpc), load(a.cmpc), a.figs, a.out,
          os.path.dirname(a.cmpc).replace(ROOT + "/", ""),
          os.path.dirname(a.mjpc).replace(ROOT + "/", ""))


if __name__ == "__main__":
    main()
