#!/usr/bin/env python3
"""Summarise a twin grid against the in-process replays it is supposed to match.

THE COMPARISON IS THE POINT. Each cell has an in-process MPC replay (S18) and a
DDS run of the same plan (twin_grid.sh). Everything between them is identical --
same OCP, same gains, same clamp, same model -- so the per-cell difference in
final pelvis height is deployment and nothing else. A cell that stands in
process and falls over the wire is a deployment failure; one that falls in both
was never the wire's fault.

Reported alongside: the loop's own health (solve time, overruns, watchdog
trips, state age) per cell, because a survival number with no timing next to it
cannot distinguish "the controller works" from "the machine was quiet".
"""
import json
import os
import re
import sys

import numpy as np

FELL = 0.55                                   # pelvis height that counts as down


def _twin_outcome(path):
    """Last JSON line lean_twin printed, or None."""
    if not os.path.exists(path):
        return None
    for line in reversed(open(path).read().splitlines()):
        m = re.search(r"\{.*\}", line)
        if m:
            try:
                return json.loads(m.group(0))
            except ValueError:
                continue
    return None


def main(out_dir, grid=None):
    grid = grid or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "runs/2026-08-16_session18/grid")
    rows = []
    for f in sorted(os.listdir(out_dir)):
        if not f.endswith(".json") or f == "summary.json":
            continue          # summary.json is this script's own output
        cell = f[:-5]
        d = json.load(open(os.path.join(out_dir, f)))
        s = d["summary"]
        tw = _twin_outcome(os.path.join(out_dir, cell + ".twin.log")) or {}
        # PAIR AGAINST THE STRESS GRID'S NOMINAL ROW, not against a replay.
        # S18 wrote a standalone MPC replay for exactly one cell but a stress
        # sweep for all 26, and the sweep's `nominal` row (seed None, no
        # disturbance) IS the undisturbed in-process run of the same plan --
        # the correct partner for a twin run that also has no disturbance.
        nom = None
        cdir = os.path.join(grid, cell)
        for r in sorted(os.listdir(cdir)) if os.path.isdir(cdir) else []:
            if r.startswith("stress_") and r.endswith(".json"):
                for row in json.load(open(os.path.join(cdir, r)))["rows"]:
                    if row.get("profile") == "nominal" and row.get("seed") is None:
                        nom = row
                        break
        rows.append(dict(cell=cell, z_twin=tw.get("pelvis_z_min_commanded"),
                         fell_inproc=None if nom is None else bool(nom["fell"]),
                         margin_inproc=None if nom is None else nom["margin_mm"],
                         **{k: s.get(k) for k in
                         ("solve_ms_mean", "solve_ms_p95", "overruns",
                          "watchdog_trips", "age_ms_p95", "tau_saturated",
                          "q_clipped", "periods", "recv")}))
    if not rows:
        print("no runs in %s" % out_dir)
        return 1

    print("%-30s %8s %9s %7s %7s %6s %5s %8s" % (
        "cell", "z_twin", "in-proc", "solve", "p95", "over", "wd", "age_p95"))
    for r in rows:
        z = r["z_twin"]
        inp = ("   --   " if r["fell_inproc"] is None
               else ("     FELL" if r["fell_inproc"] else "  upright"))
        print("%-30s %8s %9s %7.1f %7.1f %6d %5d %8.3f%s" % (
            r["cell"], "  --  " if z is None else "%8.4f" % z, inp,
            r["solve_ms_mean"] or 0, r["solve_ms_p95"] or 0,
            r["overruns"] or 0, r["watchdog_trips"] or 0, r["age_ms_p95"] or 0,
            "  FELL" if (z or 1) < FELL else ""))

    zt = np.array([r["z_twin"] for r in rows if r["z_twin"] is not None])
    print("\n%d cells over the wire: %d upright, %d fell"
          % (len(zt), int((zt >= FELL).sum()), int((zt < FELL).sum())))
    both = [(r["z_twin"] >= FELL, not r["fell_inproc"]) for r in rows
            if r["z_twin"] is not None and r["fell_inproc"] is not None]
    if both:
        a = np.array(both)
        print("paired with the in-process nominal run (%d cells):" % len(both))
        print("  upright both %d | over the wire only %d | in process only %d | neither %d"
              % (int((a[:, 0] & a[:, 1]).sum()), int((a[:, 0] & ~a[:, 1]).sum()),
                 int((~a[:, 0] & a[:, 1]).sum()), int((~a[:, 0] & ~a[:, 1]).sum())))
        dis = [r["cell"] for r in rows
               if r["z_twin"] is not None and r["fell_inproc"] is not None
               and (r["z_twin"] >= FELL) != (not r["fell_inproc"])]
        if dis:
            print("  DISAGREE: " + ", ".join(dis))
    wd = sum(r["watchdog_trips"] or 0 for r in rows)
    print("loop health: %d watchdog trips over %d periods, worst age p95 %.3f ms, recv=%s"
          % (wd, sum(r["periods"] or 0 for r in rows),
             max((r["age_ms_p95"] or 0) for r in rows),
             sorted({r["recv"] for r in rows})))
    json.dump(rows, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
