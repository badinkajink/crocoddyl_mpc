#!/usr/bin/env python3
"""Compare strategy variants across a sweep of `enter` episodes.

Reads the chain.json each mjpc_chain.py run writes and reports, per condition:

  braced      did the run establish an arm contact on the slab and keep it
  fell        pelvis below 0.55 m or tilt past 45 deg at any point
  reach gain  t=0 distance minus end-of-brace_reach distance.  THE number:
              a reach residual with no baseline is what made S12 report a
              null result as a margin/reach trade-off.
  churn       control path length / net displacement over the phase.  A
              converged hold is 1-3x; S12's chain ran 14-55x.
  duty        fraction of the phase each link was actually on the table,
              which contacts-at-end silently overstates.

usage: sweep_report.py RUNDIR [RUNDIR ...]
"""
import json
import os
import sys

import numpy as np

LABEL = {
    "s22_control": "22 stock (control)",
    "s24_rfl": "24  + Right Foot Lift 0",
    "s23_fix": "23  + Support Polygon 0, Body Yaw 40",
    "s25_v2": "25  cost refactor",
}


def phase(run, name):
    for p in run["phases"]:
        if p["name"] == name:
            return p
    return None


def summarise(rundir):
    j = json.load(open(os.path.join(rundir, "chain.json")))
    runs = [r for r in j["runs"]["enter"] if not r.get("failed")]
    rows = []
    for r in runs:
        pl, pr = phase(r, "brace_lean"), phase(r, "brace_reach")
        if pr is None:
            continue
        arm = {k: v for k, v in pr["contact_duty"].items() if k != "hip"}
        rows.append(dict(
            seed=r["seed"],
            fell=bool(pr["fell"] or (pl and pl["fell"])),
            braced=bool(arm and not pr["fell"]),
            gain=pr["reach_gain"],
            reach_end=pr["reach_end"],
            margin=pr["margin_end"],
            force=pr["brace_force_end"],
            churn=pr["ctrl_churn"],
            duty=arm,
            contacts=pr["contacts_end"],
        ))
    return j, rows


def main(dirs):
    print("%-38s %6s %6s %19s %8s %7s %7s" %
          ("condition", "braced", "fell", "reach gain [m]", "margin", "churn", "load"))
    print("-" * 100)
    store = {}
    for dd in dirs:
        key = os.path.basename(dd.rstrip("/"))
        j, rows = summarise(dd)
        store[key] = rows
        n = len(rows)
        if not n:
            print("%-38s  no runs" % LABEL.get(key, key))
            continue
        ok = [r for r in rows if r["braced"]]
        g = np.array([r["gain"] for r in ok]) if ok else np.array([np.nan])
        marg = np.array([r["margin"] for r in ok]) if ok else np.array([np.nan])
        churn = np.array([r["churn"] for r in rows])
        load = np.array([r["force"] for r in ok]) if ok else np.array([np.nan])
        print("%-38s %5d/%d %5d/%d  %+.3f [%+.3f,%+.3f] %8.3f %6.1fx %6.0fN" %
              (LABEL.get(key, key), len(ok), n,
               sum(r["fell"] for r in rows), n,
               np.nanmean(g), np.nanmin(g), np.nanmax(g),
               np.nanmean(marg), np.nanmean(churn), np.nanmean(load)))

    print()
    print("Contact duty during brace_reach (mean over braced runs), and contact sets:")
    for key, rows in store.items():
        ok = [r for r in rows if r["braced"]]
        if not ok:
            print("  %-38s -" % LABEL.get(key, key))
            continue
        agg = {}
        for r in ok:
            for k, v in r["duty"].items():
                agg.setdefault(k, []).append(v)
        duty = ", ".join("%s %.0f%%" % (k, 100 * np.mean(v))
                         for k, v in sorted(agg.items(), key=lambda x: -np.mean(x[1])))
        sets = {}
        for r in ok:
            sets[r["contacts"]] = sets.get(r["contacts"], 0) + 1
        print("  %-38s %s" % (LABEL.get(key, key), duty))
        print("  %-38s   sets: %s" % ("", dict(sorted(sets.items(), key=lambda x: -x[1]))))

    print()
    print("Per-seed reach gain (m), '.' = fell:")
    for key, rows in store.items():
        cells = " ".join(("  .   " if r["fell"] else "%+.3f" % r["gain"]) for r in rows)
        print("  %-38s %s" % (LABEL.get(key, key), cells))


if __name__ == "__main__":
    main(sys.argv[1:])
