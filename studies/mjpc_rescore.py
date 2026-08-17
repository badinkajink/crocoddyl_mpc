#!/usr/bin/env python3
"""Re-score MJPC rollouts already on disk, over the WHOLE trajectory.

The first pass scored the last second of each run, which is wrong for this task:
strategy 22 (`h12_simple_forearm_brace`) is an EIGHT-phase sequence --
stand_up -> forearm_brace_lean -> forearm_brace_reach -> forearm_brace_release
-> standback_r1/r2/r3 -> stand_up -- so its final frame is the robot standing
back up, having deliberately let go.  Scoring that frame measures the recovery,
not the brace, and reports "no brace contacts, hand 32 cm short" for a run that
may well have braced correctly in the middle.

This scans every rollout end to end and reports the frame that best answers the
actual question -- did sampling ever get to a pose like the QP's? -- plus the
contact timeline, so a brace that was established and then released is visible
as such rather than as a failure.

usage: mjpc_rescore.py [outdir]
"""
import glob
import json
import os
import sys

import numpy as np
import mujoco

import contact_select as cs
import stability as st
import mjpc_handoff as mh

STRIDE = 20          # score every 20th logged step (dt 0.002 -> 25 Hz)


def scan(m, d, rows, col, target, nq, nu):
    qi = [col["qpos%d" % i] for i in range(nq)]
    ai = [col["afrc%d" % i] for i in range(nu)]
    frames = []
    for row in rows[::STRIDE]:
        f = mh.score_frame(m, d, row[qi], target, row[ai], region=False)
        f["t"] = float(row[col["time"]])
        f["cost"] = float(row[col["cost"]])
        frames.append(f)
    return frames


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "runs/2026-08-04_session11/mjpc_handoff"
    meta = json.load(open(os.path.join(outdir, "handoff.json")))
    target = meta["target"]
    m, d = cs.load(ik_margin=0)

    print("target %s   basis=%s   scoring EVERY frame, stride %d"
          % (np.round(target, 4), cs.TAU_BASIS, STRIDE))
    print("QP reference:")
    for k, v in meta["qp_reference"].items():
        print("   %-22s reach %.4f peak %.3f margin %.4f"
              % (k, v["reach"], v["peak"], v["margin"]))

    allruns = []
    for path in sorted(glob.glob(os.path.join(outdir, "traj_seed*.csv"))):
        hdr, col, rows = mh.load_traj(path)
        frames = scan(m, d, rows, col, target, m.nq, m.nu)
        seed = os.path.basename(path)
        qi = [col["qpos%d" % i] for i in range(m.nq)]
        ai = [col["afrc%d" % i] for i in range(m.nu)]

        upright = [f for f in frames if f["tilt_deg"] < 60 and f["pelvis_z"] > 0.5]
        braced = [f for f in upright if set(f["contacts"]) & {"elbow", "forearm"}]
        any_c = [f for f in upright if f["contacts"]]
        # the pose the run should be judged on: most brace contacts, then closest
        pool = braced or any_c or upright or frames
        best = min(pool, key=lambda f: (-len(f["contacts"]), f["reach"]))
        closest = min(upright or frames, key=lambda f: f["reach"])
        # the region LP runs only on the frames that get reported
        for f in (best, closest, frames[-1]):
            j = int(round(f["t"] / float(rows[1, col["time"]] - rows[0, col["time"]])))
            j = min(max(j, 0), len(rows) - 1)
            f.update(mh.score_frame(m, d, rows[j][qi], target, rows[j][ai],
                                    region=True))
            f["t"] = float(rows[j][col["time"]])

        # contact timeline: first and last time each body was touching
        tl = {}
        for f in frames:
            for c in f["contacts"]:
                tl.setdefault(c, [f["t"], f["t"]])[1] = f["t"]

        print("\n== %s  (%d frames scored, %.1f s)" % (seed, len(frames), frames[-1]["t"]))
        print("   ever braced (elbow or forearm): %s" % ("YES" if braced else "NO"))
        print("   contact timeline: %s"
              % ("  ".join("%s %.1f-%.1f s" % (k, v[0], v[1]) for k, v in sorted(tl.items()))
                 or "none"))
        print("   best braced frame  t=%5.1f s reach %.4f contacts %-22s peak %.3f margin %+.4f"
              % (best["t"], best["reach"], "+".join(best["contacts"]) or "none",
                 best["peak_logged"], best["margin"]))
        print("   closest-reach frame t=%5.1f s reach %.4f contacts %-22s peak %.3f"
              % (closest["t"], closest["reach"],
                 "+".join(closest["contacts"]) or "none", closest["peak_logged"]))
        print("   final frame         t=%5.1f s reach %.4f pelvis_z %.3f tilt %.0f deg"
              % (frames[-1]["t"], frames[-1]["reach"], frames[-1]["pelvis_z"],
                 frames[-1]["tilt_deg"]))
        allruns.append(dict(file=seed, ever_braced=bool(braced), timeline=tl,
                            best=best, closest=closest, final=frames[-1],
                            n_scored=len(frames)))

    # headline comparison
    ref = meta["qp_reference"].get("elbow+forearm+palm") or \
        list(meta["qp_reference"].values())[0]
    print("\n== QP vs MJPC, best braced frame per seed ==")
    print("   %-14s %8s %8s %9s  %s" % ("", "reach", "peak", "margin", "contacts"))
    print("   %-14s %8.4f %8.3f %9.4f  %s"
          % ("QP solve", ref["reach"], ref["peak"], ref["margin"],
             "+".join(ref["subset"])))
    for r in allruns:
        b = r["best"]
        print("   %-14s %8.4f %8.3f %9.4f  %s"
              % (r["file"].replace("traj_", "").replace(".csv", ""),
                 b["reach"], b["peak_logged"], b["margin"],
                 "+".join(b["contacts"]) or "none"))

    dst = os.path.join(outdir, "rescore.json")
    json.dump(dict(target=target, qp_reference=meta["qp_reference"],
                   stride=STRIDE, runs=allruns), open(dst, "w"), indent=1)
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
