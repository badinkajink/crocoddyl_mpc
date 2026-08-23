#!/usr/bin/env python3
"""Does CHOOSING the contact mode beat using elbow+forearm everywhere?

WHY THIS EXISTS.  The CMPC sweep ran one braced mode -- `elbow+forearm` -- at
every target, because that is the mode the study certified at the nominal pose.
It falls at 0.90 m and it falls at 1.38 m, and a fall at BOTH ends is the
signature of a mode that is right in the middle and wrong outside it, not of a
planner that cannot brace.  So: enumerate every arm+wrist subset at each target,
take the least-effort ADMISSIBLE one, and fly that instead.

THE POLICY IS MECHANICAL, on purpose.  "Least normalized actuator effort among
the admissible" is croco_modes' own ranking, unchanged -- no target-by-target
hand-picking, or the comparison would be a search over 16 modes dressed up as a
controller.  Where nothing certifies the policy DECLINES, and that is a result:
at 1.38 m every one of the 16 subsets misses the target by 56-98 mm in IK with
torque ratios of 0.48-0.63, so the far failure is kinematic reach and no contact
schedule can buy it back.

WHAT IT SHARES WITH THE SWEEP.  Same episode driver, same 16 s, same 2 seeds,
same jitter, same CSV format -- `cmpc_brace_vs_stand` is imported, not copied,
so the rows drop straight into `brace_vs_stand.analyse_one` next to the
incumbent's and the two are scored by identical code.

usage:
    cmpc_mode_select.py plans --out runs/2026-08-23_cmpc_modesel
    cmpc_mode_select.py run   --out runs/2026-08-23_cmpc_modesel
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cmpc_brace_vs_stand as CB
import brace_vs_stand as BVS


def tag_of(mode):
    """Filesystem tag for a mode name: croco_run writes plan_<tag>.json."""
    return mode.replace("+", "_")


def load_picks(out):
    return json.load(open(os.path.join(out, "picks.json")))


# --------------------------------------------------------------------------- #
def cmd_plans(a):
    picks = load_picks(a.out)
    for cell, p in sorted(picks.items()):
        if not p["best"]:
            print("%-6s (no admissible mode -- nothing to solve)" % cell)
            continue
        cdir = os.path.join(a.out, "cells", cell)
        tag = tag_of(p["best"])
        path = os.path.join(cdir, "plan_%s.json" % tag)
        if os.path.exists(path) and not a.force:
            print("%-6s %-22s cached" % (cell, p["best"]))
            continue
        cmd = [sys.executable, os.path.join(HERE, "croco_run.py"),
               "--dir", cdir, "--tag", tag, "--dt", "0.02",
               "--contact-kp", "%g" % CB.CONTACT_KP,
               "--reach-rot", "auto", "--w-reach-rot", "1e-2",
               "--mode", p["best"], "--start", "stand",
               "--n-approach", "120", "--n-braced", "80"]
        t = time.time()
        with open(os.path.join(cdir, "solve_%s.log" % tag), "w") as fh:
            rc = subprocess.run(cmd, stdout=fh,
                                stderr=subprocess.STDOUT).returncode
        print("%-6s %-22s rc=%d  %.1f s" % (cell, p["best"], rc,
                                            time.time() - t), flush=True)


def cmd_run(a):
    picks = load_picks(a.out)
    import contact_select as cs
    tau_lim = cs.torque_limits(cs.load(ik_margin=0.0)[0])
    plant = CB.make_recording_plant(tau_lim)
    manifest = []
    for cell, p in sorted(picks.items()):
        if not p["best"]:
            continue
        cdir = os.path.join(a.out, "cells", cell)
        tag = tag_of(p["best"])
        if not os.path.exists(os.path.join(cdir, "plan_%s.json" % tag)):
            print("%-6s no plan -- skipped" % cell)
            continue
        t = time.time()
        print("       building OCP %s / %s ..." % (cell, tag), end="",
              flush=True)
        try:
            fl = CB.Flight(cdir, tag)
        except Exception as exc:                                # noqa: BLE001
            print(" FAILED: %s" % exc, flush=True)
            continue
        print(" %.1f s" % (time.time() - t), flush=True)
        for r in range(BVS.SWEEP_REPS):
            # `_brace` in the name is load-bearing: brace_vs_stand.cmd_analyze
            # reads the arm out of the filename, and without it a braced
            # rollout is scored and plotted as a standing one.
            name = "modesel_%s_brace_r%d" % (cell, r)
            csv_path = os.path.join(a.out, name + ".csv")
            if os.path.exists(csv_path) and not a.force:
                print("  %-26s cached" % name, flush=True)
                continue
            rows, summ = CB.run_episode(fl, plant, BVS.SWEEP_SECONDS, seed=r)
            CB.write_csv(csv_path, rows, fl, p["target"], "brace", r,
                         BVS.SWEEP_SECONDS, "")
            print("  %-26s %5d rows  %.0f s wall  pelvis %.3f  solve %.1f ms"
                  % (name, len(rows), summ["wall_s"], summ["pelvis_z"],
                     summ["solve_ms_mean"] or -1), flush=True)
            # `summ` already carries `seed`; passing it again is a TypeError
            manifest.append(dict(tag=name, cell=cell, mode=p["best"],
                                 target=p["target"], **summ))
        del fl
    if manifest:
        with open(os.path.join(a.out, "manifest_modesel.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        print("wrote manifest_modesel.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("plans", "run"):
        c = sub.add_parser(nm)
        c.add_argument("--out", required=True)
        c.add_argument("--force", action="store_true")
    rp = sub.add_parser("report")
    rp.add_argument("--out", required=True)
    rp.add_argument("--sweep", required=True,
                    help="the incumbent run dir (elbow+forearm at every target)")
    rp.add_argument("--figs", required=True)
    a = ap.parse_args()
    {"plans": cmd_plans, "run": cmd_run, "report": cmd_report}[a.cmd](a)




# --------------------------------------------------------------------------- #
# reporting -- runs under the PLOTTING interpreter (matplotlib), not the croco
# one.  It reads only JSON, so the split is safe.
# --------------------------------------------------------------------------- #
def cmd_report(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import bvs_plots as BP

    sel = json.load(open(os.path.join(a.out, "analysis.json")))
    inc = json.load(open(os.path.join(a.sweep, "analysis.json")))
    picks = load_picks(a.out)
    xmode = {round(p["target"][0], 3): p["best"] for p in picks.values()}

    def group(rows, stage, arm):
        d = {}
        for r in rows:
            if r.get("stage") != stage or r.get("arm") != arm or "error" in r:
                continue
            d.setdefault(round(r["target"][0], 3), []).append(r)
        return d

    series = [
        ("legs_only", group(inc, "sweep", "stand"), "#2a78d6", "-", "o"),
        ("elbow+forearm (fixed)", group(inc, "sweep", "brace"),
         "#eb6834", "-", "s"),
        ("least-effort mode (selected)", group(sel, "modesel", "brace"),
         "#1baf7a", "--", "^"),
    ]

    # ---- table ----------------------------------------------------------- #
    def one(rs, key, sc=1.0):
        v = [r.get(key) for r in rs if r.get(key) is not None]
        return float(np.mean(v)) * sc if v else float("nan")

    lines = ["%-6s %-22s %-9s %8s %8s %8s %7s"
             % ("x [m]", "selected mode", "outcome", "err[cm]", "armN",
                "margin", "fell")]
    tex = [r"\begin{tabular}{llllrrr}", r"\toprule",
           r"$x$ [m] & selected mode & fixed & selected & err [cm] & "
           r"arm [N] & margin [cm] \\", r"\midrule"]
    for x in sorted(set(series[1][1]) | set(series[2][1])):
        rs_i, rs_s = series[1][1].get(x, []), series[2][1].get(x, [])
        mode = xmode.get(x, "--")
        fi = sum(bool(r.get("fell")) for r in rs_i)
        fs = sum(bool(r.get("fell")) for r in rs_s)
        if not rs_s:
            lines.append("%-6.2f %-22s %-9s %8s %8s %8s %7s"
                         % (x, "(none certifies)", "declined", "--", "--",
                            "--", "%d/%d" % (fi, len(rs_i))))
            tex.append(r"%.2f & \emph{none certifies} & %d/%d fell & "
                       r"declined & --- & --- & --- \\"
                       % (x, fi, len(rs_i)))
            continue
        lines.append("%-6.2f %-22s %-9s %8.1f %8.0f %8.1f %7s"
                     % (x, mode,
                        "fell" if fs else "held",
                        one(rs_s, "reach_err_settled", 100.0),
                        np.mean([BP.brace_arm_load(r) for r in rs_s]),
                        one(rs_s, "margin_actuated_settled", 100.0),
                        "%d/%d vs %d/%d" % (fs, len(rs_s), fi, len(rs_i))))
        tex.append(r"%.2f & \texttt{%s} & %d/%d fell & %d/%d fell & %.1f & "
                   r"%.0f & %.1f \\"
                   % (x, mode.replace("+", "+"), fi, len(rs_i), fs, len(rs_s),
                      one(rs_s, "reach_err_settled", 100.0),
                      np.mean([BP.brace_arm_load(r) for r in rs_s]),
                      one(rs_s, "margin_actuated_settled", 100.0)))
    tex += [r"\bottomrule", r"\end{tabular}"]
    body = "\n".join(lines)
    print("\n" + body + "\n")
    with open(os.path.join(a.figs, "table_mode_select.txt"), "w") as f:
        f.write(body + "\n")
    with open(os.path.join(a.figs, "table_mode_select.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")

    # ---- figure ---------------------------------------------------------- #
    fig, ax = plt.subplots(1, 2, figsize=(BP.TEXT, 2.7), layout="constrained")
    KEYS = ("reach_err_settled", "margin_actuated_settled")
    for lab, d, col, ls, mk in series:
        xs = sorted(d)
        if not xs:
            continue
        # THE CURVE IS OVER UPRIGHT ROLLOUTS ONLY, and a target where every
        # replicate fell is a gap in it rather than a point. A fallen robot's
        # settled reach error is 262 cm -- the distance from the target to a
        # machine on the floor -- and plotting it puts the entire comparison
        # into the bottom 3% of the axis to make room for a number that means
        # nothing. The fall is still shown: it is the cross.
        up = {x: [r for r in d[x] if not r.get("fell")] for x in xs}
        for i, key in enumerate(KEYS):
            px = [x for x in xs if up[x]]
            mu = [100.0 * np.mean([r[key] for r in up[x]
                                   if r.get(key) is not None]) for x in px]
            ax[i].plot(px, mu, ls, color=col, marker=mk, ms=4, lw=1.4,
                       label=lab if i == 0 else None, zorder=3)
        for x in xs:
            if not any(r.get("fell") for r in d[x]):
                continue
            for i, key in enumerate(KEYS):
                if up[x]:
                    v = 100.0 * np.mean([r[key] for r in up[x]
                                         if r.get(key) is not None])
                    ax[i].plot([x], [v], "x", color=col, ms=9, mew=2.0,
                               zorder=6)
                else:
                    # ON THE AXIS FLOOR, in axes coordinates. `get_ylim()[0]`
                    # here reads the AUTOSCALE bottom, which is whatever the
                    # data happens to reach and is not 0 -- it put the "both
                    # replicates fell" cross at 0.7 cm, i.e. on top of the
                    # region where real arrivals live.
                    from matplotlib.transforms import blended_transform_factory
                    tf = blended_transform_factory(ax[i].transData,
                                                   ax[i].transAxes)
                    ax[i].plot([x], [0.0], "x", color=col, ms=9, mew=2.0,
                               zorder=6, clip_on=False, transform=tf)
    ax[0].set_ylabel("Settled reach error  [cm]")
    ax[0].set_title("Does it arrive?", loc="left")
    ax[1].set_ylabel("CoM margin, actuated  [cm]")
    ax[1].set_title("How much margin it holds", loc="left")
    for A in ax:
        A.set_xlabel("Commanded target $x$  [m]")
    ax[0].axhline(5.0, color=BP.INK2, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax[0].annotate("5 cm arrival tolerance", xy=(0.03, 5.0),
                   xycoords=("axes fraction", "data"), xytext=(0, 3),
                   textcoords="offset points", color=BP.INK2, fontsize=7)
    ax[0].set_ylim(0, 9)
    ax[0].legend(frameon=False, fontsize=7, loc="upper right")
    ax[1].annotate("× = a replicate fell", xy=(0.03, 0.95),
                   xycoords="axes fraction", ha="left", va="top", fontsize=7,
                   color=BP.INK2)
    BP._save(fig, a.figs, "fig_mode_select")


if __name__ == "__main__":
    main()
