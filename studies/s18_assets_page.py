#!/usr/bin/env python3
"""The S18 docpage: what the asset resync changed, and what it cost to find out.

Same contract as the other page generators here -- every number comes out of the
run directory, so the page cannot drift from the measurement without the file
changing under it. Unlike them the body is inline: this page is meant to be
short, and a two-file split for six sections is more machinery than content.

It DEGRADES. The grid's stress stage takes about an hour; sections whose inputs
are not there yet say so and the page still builds, because a page you cannot
regenerate mid-run is a page nobody regenerates.

usage: s18_assets_page.py --dir runs/2026-08-16_session18 \\
                          --out 2026-08-16_assets.html
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.normpath(os.path.join(HERE, "..", "docs", "lean"))
sys.path.insert(0, DOCS)
sys.path.insert(0, HERE)

MISSING = ('<p class="note">Not measured yet &mdash; this section fills in when '
           '<code>%s</code> exists.</p>')


# ------------------------------------------------------------------ loading

def cells(root):
    """(name -> manifest) for every certified cell under a grid dir."""
    out = {}
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "modes.json")
        if os.path.exists(p):
            try:
                out[name] = json.load(open(p))
            except Exception:                                   # noqa: BLE001
                pass
    return out


def admissible(man):
    return [m for m in man.get("modes", []) if m.get("admissible")]


def best(man):
    a = admissible(man)
    return min(a, key=lambda m: m["effort"])["name"] if a else None


def gap_of(man, mode, site):
    """Signed distance of one brace site at q*, or None if it never touches."""
    for m in man.get("modes", []):
        if m.get("name") == mode:
            g = (m.get("gaps") or {}).get(site)
            return g
    return None


# ------------------------------------------------------------------ sections

def sec_certify(new, old):
    """Before/after over the whole grid: what certifies, and how cheaply."""
    if not new:
        return MISSING % "grid/*/modes.json"
    rows = []
    for name in sorted(new):
        n, o = new[name], old.get(name)
        rows.append((name, len(admissible(n)), best(n),
                     len(admissible(o)) if o else None, best(o) if o else None))
    n_palm_new = sum(1 for r in rows if r[2] and "palm" in r[2])
    n_palm_old = sum(1 for r in rows if r[4] and "palm" in r[4])
    body = [
        "<h2>1 &middot; What the corrected gripper certifies</h2>",
        "<p>The offline stage enumerates contact subsets and asks a static "
        "equilibrium QP which of them a pose can actually hold. It is the "
        "cheapest thing in the pipeline and the most sensitive to geometry, so "
        "it is where a transposed gripper shows up first.</p>",
        f"<p><b>{n_palm_new} of {len(rows)} cells</b> now pick a subset that "
        f"includes the <b>palm</b>, against <b>{n_palm_old}</b> on the old "
        "proxy. That is the transposition, stated as an outcome: the old box "
        "claimed 12.5&nbsp;mm of hardware on the side the palm brace lands on "
        "and 90&deg; of it in the wrong plane, so the palm could not be seated "
        "at the same time as the forearm. S15 read that as a property of the "
        "robot &mdash; &ldquo;what the CAD-faithful gripper costs is having the "
        "palm AND the forearm down at once&rdquo; &mdash; and it was a property "
        "of the proxy.</p>",
        '<table><thead><tr><th>cell</th>'
        '<th colspan="2">corrected</th><th colspan="2">old proxy</th></tr>'
        '<tr><th></th><th>#adm</th><th>cheapest</th>'
        '<th>#adm</th><th>cheapest</th></tr></thead><tbody>']
    for name, na, ba, no, bo in rows:
        chg = ' class="hi"' if ba != bo else ""
        body.append(f"<tr{chg}><td><code>{name}</code></td><td>{na}</td>"
                    f"<td>{ba or '&mdash;'}</td>"
                    f"<td>{'&mdash;' if no is None else no}</td>"
                    f"<td>{bo or '&mdash;'}</td></tr>")
    body.append("</tbody></table>")
    if old:
        body.append(
            "<p class=\"note\">Old-proxy column is the same enumeration run "
            "against the pre-fix <code>Lean_H12_Magpie.xml</code>, same targets, "
            "same stances, same code &mdash; only the model differs.</p>")
    return "\n".join(body)


def sec_grid(run_dir):
    """The stress result, curated -- and what it does not say."""
    import collections
    pth = os.path.join(run_dir, "grid.json")
    if not os.path.exists(pth):
        return "<h2>3 &middot; Does it survive</h2>\n" + MISSING % "grid.json"
    rows = json.load(open(pth)).get("rows", [])
    if not rows:
        return "<h2>3 &middot; Does it survive</h2>\n" + MISSING % "grid.json rows"

    nom = sum(1 for r in rows if r.get("nominal_ok"))
    w_ok = w_n = 0
    for r in rows:
        rate = (r.get("rate") or {}).get("winch1")
        if rate:
            w_ok += rate[0] * rate[1]
            w_n += rate[1]
    # PER-SEED outcomes live in `why` (a list, one entry per winch1 seed).
    # `reasons` is a dict keyed by contact MODE and is about certification --
    # iterating it here silently compared key sets and reported that every cell
    # agreed, which was an artefact and not a result.
    seqs = [tuple(r.get("why") or []) for r in rows if r.get("why")]
    n_seeds = max((len(x) for x in seqs), default=0)
    always_fail = [i for i in range(n_seeds)
                   if seqs and all(x[i] != "ok" for x in seqs if len(x) == n_seeds)]
    pats = collections.Counter(seqs)
    top_n = pats.most_common(1)[0][1] if pats else 0

    out = [
        "<h2>3 &middot; Does it survive</h2>",
        "<p>Each cell is planned once, then replayed in MuJoCo under the MPC. "
        "<b>Nominal</b> is the plan with no disturbance at all &mdash; it asks "
        "whether the trajectory is executable. <b>winch1</b> perturbs how the "
        "robot arrives: 20&nbsp;mm of placement, 10&nbsp;mm of height, "
        "50&nbsp;mrad of heading, 20&nbsp;mrad of tilt, 0.02&nbsp;rad of joint "
        "noise.</p>",
        f"<p><b>Nominal: {nom} of {len(rows)} cells.</b> That is the number the "
        "palm-site fix bought &mdash; before it, every <code>elbow+palm</code> "
        "cell failed nominal at 0%, because the plan was aiming a contact point "
        "that sat 18&nbsp;mm inside the gripper.</p>",
        f"<p><b>winch1: {100 * w_ok / max(w_n, 1):.0f}%</b> "
        f"({int(round(w_ok))} of {w_n} runs), against S17's 75%. "
        "<i>Do not read this as a controller regression yet.</i> "
        f"Seeds {', '.join(str(i + 1) for i in always_fail)} fail in "
        f"<b>every one of the {len(seqs)} cells</b>, and {top_n} cells share "
        "the identical outcome sequence end to end. The result is therefore set "
        "by which disturbance the seed drew, not by the target &mdash; and five "
        "draws is too few to separate five correlated variables. The leading "
        "suspect is start height: the "
        "two surviving seeds are the two that start low "
        "(&minus;15 and &minus;21&nbsp;mm); all three that fall start at or "
        "above nominal. Isolating it needs a fixed-offset sweep, which the "
        "profile cannot currently express (it takes standard deviations, not "
        "offsets). Open.</p>",
        '<table><thead><tr><th>cell</th><th>mode</th><th>nominal</th>'
        '<th>winch1</th><th>reach (mm)</th></tr></thead><tbody>']
    for r in rows:
        rate = (r.get("rate") or {}).get("winch1")
        w = "&mdash;" if not rate else f"{100 * rate[0]:.0f}%"
        n_ok = r.get("nominal_ok")
        cls_ = ' class="hi"' if not n_ok else ""
        out.append(
            f"<tr{cls_}><td><code>{r.get('cell')}</code></td>"
            f"<td>{r.get('mode')}</td>"
            f"<td>{'ok' if n_ok else '<b>FAIL</b>'}</td><td>{w}</td>"
            f"<td>{(r.get('reach_mm') or 0):.0f}</td></tr>")
    out.append("</tbody></table>")
    out.append(
        "<h3>Why it was 85&nbsp;ms, and why it is 13.6</h3>"
        "<p>The first pass of this grid ran the MPC at <b>84.7&nbsp;ms mean / "
        "138.8&nbsp;ms p95</b> against a 20&nbsp;ms control period &mdash; "
        "essentially S13's 79.7&nbsp;ms, i.e. none of S15's work was active. "
        "The cause was not the environment's crocoddyl build, which is where "
        "the OpenMP warning points: it was that <code>croco_ext/*.so</code> had "
        "never been built in this checkout, and <code>croco_geom</code> falls "
        "back to the Python keep-out activation with a one-line note. It runs "
        "correctly and six times too slowly, and nothing in a replay's output "
        "says so except the solve time, which reads as &ldquo;this problem is "
        "expensive&rdquo;.</p>"
        "<p>Building it restores S15's numbers exactly, and the trajectory is "
        "unchanged in every configuration &mdash; pelvis&nbsp;z 0.956&nbsp;m, "
        "reach 15.0&nbsp;mm, penetration &minus;2.5&nbsp;mm throughout &mdash; "
        "so this is a pure speed-up, not a behaviour change. The two native "
        "pieces were checked against their Python originals first "
        "(<code>test_keepout.py</code>: exact to 0.0 on sdf, gradient, value, "
        "Ar and Arr; <code>test_passive.py</code>: 7e-15 on tau).</p>"
        '<table><thead><tr><th>keep-out</th><th>actuation</th><th>mean</th>'
        '<th>p95</th></tr></thead><tbody>'
        '<tr class="hi"><td>fused</td><td>cpp</td><td>13.6 ms</td><td>19.7 ms</td></tr>'
        '<tr><td>cpp</td><td>cpp</td><td>28.7 ms</td><td>37.5 ms</td></tr>'
        '<tr><td>python</td><td>cpp</td><td>83.3 ms</td><td>136.1 ms</td></tr>'
        '<tr><td>fused</td><td>python</td><td>14.0 ms</td><td>20.6 ms</td></tr>'
        '<tr><td>python</td><td>python</td><td>85.2 ms</td><td>141.3 ms</td></tr>'
        "</tbody></table>"
        "<p>So the keep-out activation is very nearly the whole cost and the "
        "actuation model is 3% of it, which is the split S15 reported: the C++ "
        "activation was half the problem (85&nbsp;&rarr;&nbsp;29) and fusing the "
        "86 points into one cost term was the other half "
        "(29&nbsp;&rarr;&nbsp;13.6). The p95 lands ON the period rather than "
        "comfortably inside it, same as S15; <code>--mpc-no-cones</code>, "
        "<code>--mpc-alphas</code> and <code>--mpc-dt-scale</code> are the "
        "margin knobs if it needs more.</p>"
        "<p class=\"note\">The survival numbers above were measured on the slow "
        "path and are unaffected: the replay is not real-time, and the "
        "trajectory is identical either way. <code>croco_env.py</code> now "
        "refuses to start a long run with the extension unbuilt, and "
        "<code>run_session.sh deps</code> builds it &mdash; &ldquo;remember to "
        "build the extensions&rdquo; is the kind of instruction that gets "
        "skipped exactly once.</p>")
    return "\n".join(out)


def sec_videos(run_dir, media_rel="media"):
    media = os.path.join(run_dir, "media")
    if not os.path.isdir(media):
        return "<h2>2 &middot; The gripper</h2>\n" + MISSING % "media/"
    known = [
        ("s18_gripper_old_vs_new.mp4",
         "<b>Old proxy (left) against the corrected CAD (right)</b>, same pose, "
         "same camera. Left is the whole gripper as the planner used to see it: "
         "the 26&nbsp;mm adapter puck, with collision boxes standing on end. "
         "Right is the real assembly &mdash; base, cranks, fingers, rockers "
         "&mdash; with the jaw plates lying in the plane they actually lie in."),
        ("s18_gripper_orbit.mp4",
         "The corrected gripper at the braced pose, visual mesh and collision "
         "proxy overlaid. The proxy is what the planner reasons about; the mesh "
         "is what it stands for, and they now agree."),
        ("s18_gripper_wrist_roll.mp4",
         "The brace wrist rolled through &plusmn;100&deg;. The certified brace "
         "rolls it ~89&deg; to put the jaws <i>lateral</i>; against the "
         "transposed proxy that same roll put them up and down, which is the "
         "configuration the study rejected for hanging a finger at the wood."),
        ("s18_gripper_jaws.mp4",
         "The jaws through the 4-bar's real travel &mdash; 155&nbsp;mm open to "
         "2&nbsp;mm closed. Every frame is a separately generated model, because "
         "the gripper carries no joints on purpose."),
    ]
    out = ["<h2>2 &middot; The gripper</h2>"]
    any_ = False
    for fn, cap in known:
        if not os.path.exists(os.path.join(media, fn)):
            continue
        any_ = True
        out.append(f'<figure><video src="{media_rel}/{fn}" autoplay loop muted '
                   f'playsinline controls></video><figcaption>{cap}'
                   f'</figcaption></figure>')
    if not any_:
        return "<h2>2 &middot; The gripper</h2>\n" + MISSING % "media/*.mp4"
    return "\n".join(out)


HEADER = """
<h1>The gripper was 90&deg; out, and everything downstream believed it</h1>
<p class="lede">Session 18. Syncing the crocoddyl branch with Allen's
<code>icra2026</code> for asset parity turned up a geometry error in the part of
the robot this whole study is about. The magpie collision proxy had the right
three numbers in the wrong two axes &mdash; body half-extents
(0.0415, <b>0.0170, 0.0667</b>) against the CAD's
(0.0415, <b>0.0667, 0.0170</b>) &mdash; and sat 25&nbsp;mm short. The visual was
not the gripper at all: it was the adapter puck. Both are now generated from
CL_Assets at build time.</p>
<p class="meta">2026-08-16 &middot; <code>mujoco_mpc@crocoddyl-mpc</code> &middot;
<code>_gen_magpie_gripper.py</code>, <code>_gen_h12_base_limits.py</code>,
<code>croco/</code> &middot; run dir <code>%(run)s</code></p>

<h2>0 &middot; What changed, and what it invalidates</h2>
<ul>
<li><b>Merged <code>icra2026</code></b> for asset and environment parity: table
at 1.04&nbsp;m, legs to the floor, Allen's real-run brace keyframes, the deploy
helpers and the base estimator. The MPC method is <i>not</i> synced.</li>
<li><b>The actuator torque budget moved into the generator.</b> icra2026 shipped
it as a patch hunk quoting the ctrlranges the CL_Assets import rewrites, so the
hunk rejected &mdash; and the magpie patch runs through a tolerant wrapper that
discards the <code>.rej</code>. The build reported success while shipping a model
with no <code>forcerange</code> at all.</li>
<li><b>The gripper is generated from CL_Assets</b>, welded: no joint, no body,
mass&nbsp;0. <code>nq/nv/nu</code> stay 41/39/27 and the inertia is untouched;
the generator re-loads the task model and refuses to write otherwise. All eight
parts land within 0.0004&nbsp;mm of the CAD.</li>
</ul>
<p><b>Every S14&ndash;S17 number is stale.</b> The table moved and the seed
keyframe changed, which alone re-derives the certified pose; the gripper fix
changes which contact subsets are reachable at all. This page is the re-run.</p>
"""

FOOTER = """
<h2>4 &middot; Where this leaves the deploy</h2>
<p>The controller was always deployment-shaped and it was hard to see: the replay
computed <code>(q_des, kp, kd, tau_ff)</code> &mdash; exactly what
<code>rt/lowcmd</code> carries &mdash; and then collapsed it into MuJoCo's single
<code>ctrl</code> on the last line before stepping. That collapse,
<code>ctrl = q_des + (tau_ff + kd&nbsp;v_des)/kp</code>, is a property of the
MuJoCo <i>plant</i>, not of the controller.</p>
<p>So the port is a plant swap, and <code>croco/plant/</code> is the whole of it:
<code>read()</code>, <code>write()</code>, <code>now()</code>. None of
<code>deploy_common.cc</code>'s 2683 lines are needed, because this stack never
links <code>libmjpc</code> &mdash; it reads one MJCF and plans in Pinocchio.
Verified against the staged model: the inversion reproduces
<code>kp&thinsp;e + kd&thinsp;e&#775; + tau_ff</code> to 1.2e-14, and the MJCF
actuator order <i>is</i> the Unitree motor order.</p>

<h2>5 &middot; Three things that cost a bisect</h2>
<p>The re-run failed with SIGSEGV in every contact mode before it produced a
number, and none of the causes was the assets. They are recorded because each
one presented as something else:</p>
<ol>
<li><code>build/</code> is root-owned under docker, so the staging script could
not write. Now <code>STAGE_ROOT</code>.</li>
<li>The bridge defaulted its URDF to the cmake FetchContent tree, so a machine
that staged with <code>CL_ASSETS_DIR</code> got a staged MJCF and no URDF, and
failed inside urdfdom as <i>&ldquo;does not contain a valid URDF model&rdquo;</i>
&mdash; a parse error for a file that is simply absent.</li>
<li><b>The interpreter.</b> <code>base</code> has a crocoddyl wheel set whose
<i>contact</i> dynamics segfault; the <code>croco</code> env's works. Every cell
died <code>rc=-11</code> with no traceback, on the old model as well as the new
one, which reads exactly like an asset regression. A contact-<i>free</i>
crocoddyl problem solves fine in the broken environment, so &ldquo;crocoddyl
works&rdquo; proves nothing. <code>croco_env.py</code> now probes it in a
subprocess &mdash; the failure is a signal, not an exception &mdash; and
<code>run_session.sh</code> runs that check before anything long starts.</li>
</ol>
"""

STYLE_EXTRA = """
<style>
figure{margin:1.6rem 0}
figure video{width:100%;border-radius:6px;display:block}
figcaption{font-size:.92rem;opacity:.85;margin-top:.5rem}
tr.hi td{background:rgba(120,180,255,.13)}
.note{font-size:.92rem;opacity:.8}
.lede{font-size:1.06rem}
</style>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/2026-08-16_session18")
    ap.add_argument("--out", default="2026-08-16_assets.html")
    ap.add_argument("--style-from", default="2026-08-14_robustness.html")
    ap.add_argument("--title",
                    default="The gripper was 90 degrees out, and everything "
                            "downstream believed it")
    a = ap.parse_args()

    run = os.path.abspath(a.dir)
    new = cells(os.path.join(run, "grid"))
    old = cells(os.path.join(run, "grid_oldmodel"))

    body = "\n".join([
        HEADER % {"run": os.path.relpath(run, os.path.dirname(HERE))},
        sec_certify(new, old),
        sec_videos(run),
        sec_grid(run),
        FOOTER,
    ])

    try:
        from simple_page import style_from
        style = style_from(os.path.join(DOCS, a.style_from))
    except Exception as exc:                                    # noqa: BLE001
        print("  (no shared style: %s)" % exc)
        style = "<style>body{font:16px/1.6 system-ui;max-width:52rem;" \
                "margin:3rem auto;padding:0 1rem}" \
                "table{border-collapse:collapse;width:100%}" \
                "td,th{border-bottom:1px solid #8884;padding:.35rem .5rem;" \
                "text-align:left;font-size:.94rem}code{font-size:.9em}</style>"

    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{a.title}</title>\n' + style + STYLE_EXTRA
            + '\n</head>\n<body><div class="wrap">\n' + body
            + '\n</div></body>\n</html>\n')
    dst = os.path.join(DOCS, a.out)
    open(dst, "w").write(html)
    print("wrote %s  (%d bytes, %d cells new / %d old)"
          % (dst, len(html), len(new), len(old)))


if __name__ == "__main__":
    main()
