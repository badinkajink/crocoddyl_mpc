#!/usr/bin/env python3
"""The Lean Simple experiment: does a requested contact mode get established?

Four questions, one matrix:

  Q1  Does the rewrite reach at all?   NEAR target, mode elbow+forearm, against
      the S12 result on the same robot/table/target (-0.041 m of reach gain).
  Q2  Is the contact mode doing the work, or is the arm just long enough?
      Same target with mode `none` -- no seat term, and all three candidate
      links held OFF the slab by the keepout. If `none` reaches just as well,
      the brace is decoration and every mode result below is uninterpretable.
  Q3  Is the mode STEERABLE? A target far enough that standing cannot get there,
      run under five different requested modes. The claim being tested is that
      the achieved contact set follows the requested one -- that is the whole
      point of making the mode an input.
  Q4  Is any of this a cost result, or is it the search? The same cell run with
      4x the samples and with 2x the horizon. S12 §9 warned that sweeping the
      costs without sweeping the planner attributes the outcome to the wrong
      knob.

usage: simple_matrix.py --out DIR [--seeds 3] [--seconds 20] [--jobs 3]
"""
import argparse
import concurrent.futures as cf
import json
import os

import simple_lean as S

NEAR = "0.9047|-0.2348|1.0982"     # the S11/S12 target
FAR = "1.1500|-0.2348|1.0982"      # 0.245 m further out, same height and side
# Beyond the shipped braced keyframe's own hand (x = 1.247). Added after FAR
# turned out to be reachable with NO table contact at all: if the brace never
# buys reach, the experiment has not yet found the regime where it would.
VFAR = "1.3000|-0.2348|1.0982"

CELLS = [
    # (tag, mode, target, extra numerics, note)
    ("near_brace",  "elbow+forearm", NEAR, "", "Q1 the S12 comparison"),
    ("near_none",   "none",          NEAR, "", "Q2 control: no mode requested"),
    ("far_brace",   "elbow+forearm", FAR,  "", "Q3 the certified mode"),
    ("far_none",    "none",          FAR,  "", "Q2/Q3 control"),
    ("far_forearm", "forearm",       FAR,  "", "Q3 single link"),
    ("far_palm",    "palm",          FAR,  "", "Q3 a different mode"),
    ("far_all",     "elbow+forearm+palm", FAR, "", "Q3 all three"),
    ("far_brace_nosupport", "elbow+forearm", FAR, "brace_support=0",
     "Q3 feet-only support region: does the brace still buy reach?"),
    ("far_brace_samples", "elbow+forearm", FAR,
     "sampling_trajectories=40", "Q4 4x the samples"),
    ("far_brace_horizon", "elbow+forearm", FAR,
     "agent_horizon=2.0", "Q4 2x the horizon"),
    # Q5, added once Q2 came back negative at FAR: is there ANY target where the
    # brace buys reach? VFAR is past the braced keyframe's own hand.
    ("vfar_brace",  "elbow+forearm", VFAR, "", "Q5 beyond the braced keyframe"),
    ("vfar_none",   "none",          VFAR, "", "Q5 control"),
    # Q6: the last few millimetres. The forearm parks 3-5 mm off the slab
    # because at weight 300 that gap is 0.9 cost units against a 10-sample CEM.
    ("far_seatdepth", "elbow+forearm", FAR, "seat_depth=-0.005",
     "Q6 saturate 5 mm inside, i.e. within the seat calibration's uncertainty"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--only", default="", help="comma-separated cell tags")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose CSV already ran the full duration")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    cells = [c for c in CELLS
             if not a.only or c[0] in a.only.split(",")]
    work = []
    for tag, mode, target, num, note in cells:
        for seed in range(a.seeds):
            name = "%s_s%d" % (tag, seed)
            numerics = "reach_target=%s" % target
            if num:
                numerics += "," + num
            work.append((name, tag, mode, seed, numerics, note))

    def complete(path):
        """True if `path` already holds a rollout that ran the full duration.

        This machine has OOM-killed the matrix mid-flight twice, so the driver
        has to be restartable without throwing away the runs that did finish.
        A truncated CSV is NOT complete and is re-run: scoring a rollout that
        stopped at 12 of 20 s as if it had settled is how a half-finished lean
        gets reported as a converged one.
        """
        if not os.path.exists(path):
            return False
        try:
            _, rows, _ = S.load_traj(path)
            return len(rows) > 10 and rows[-1, 0] >= a.seconds - 0.5
        except Exception:                                  # noqa: BLE001
            return False

    def run(item):
        name, tag, mode, seed, numerics, note = item
        csv = os.path.join(a.out, name + ".csv")
        if a.resume and complete(csv):
            return name, tag, mode, seed, numerics, note, 0, "(cached)", csv
        rc, cmd = S.run_one(csv, mode, a.seconds, a.threads, 300.0, "",
                            numerics, "", log=os.path.join(a.out, name + ".log"))
        return name, tag, mode, seed, numerics, note, rc, cmd, csv

    results = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for (name, tag, mode, seed, numerics, note,
             rc, cmd, csv) in ex.map(run, work):
            if rc != 0 or not os.path.exists(csv):
                print("[%s] FAILED rc=%d" % (name, rc), flush=True)
                results.append(dict(tag=tag, cell=name, mode=mode, seed=seed,
                                    error="run failed rc=%d" % rc))
                continue
            out, _, _ = S.score(csv)
            out.update(cell=name, tag=tag, mode=mode, seed=seed, note=note,
                       numerics=numerics, cmd=cmd)
            results.append(out)
            print("[%-22s] req=%-20s got=%-22s gain=%+.3f fell=%-5s trunk=%.3f"
                  % (name, mode, "+".join(out["achieved_mode"]) or "none",
                     out["reach_gain"], out["fell"], out["trunk_gap_min"]),
                  flush=True)

    with open(os.path.join(a.out, "matrix.json"), "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", os.path.join(a.out, "matrix.json"))


if __name__ == "__main__":
    main()
