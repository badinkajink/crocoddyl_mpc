#!/usr/bin/env python3
"""Would the real robot's safety layer have let these rollouts happen?

The study's sim reach numbers come from a plant with no safety layer in front
of it. The hardware's do not: `h1_real_controller.launch.py` starts
`h12_safety_layer` with `relax_safety_split.yaml`, and that node LATCHES an
e-stop on the first `lowstate` sample outside its limits, then publishes
mode 0 / kp = kd = 0 -- the robot goes limp and the run is over. So a sim
posture that trips it is a posture the hardware could not have reached, whatever
the planner says.

This replays every rollout through the ACTUAL check. `h12_safety_layer.core.
config.load_config` builds the limit arrays and `check_estop_limits` reads
`q`, `dq` and `tau_est` off `lowstate`; the three sim analogues are the joint
position, the joint velocity, and `data.actuator_force` (the motor torque
MuJoCo applies, which is what `tau_est` measures). The config is loaded rather
than transcribed, so a limit change on the robot shows up here.

WHAT THE CHECK CAN AND CANNOT SAY. It is evaluated on the sim's OWN trajectory,
so it answers "was this rollout inside the envelope", not "what would the robot
have done instead". A trip means the maneuver as planned is unavailable on
hardware; it does not predict the fallback the robot would have found.

usage:
  estop_replay.py --run DIR [--json analysis.json] [--config relax_safety_split.yaml]
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

import simple_lean as S
import brace_vs_stand as BVS
import bvs_plots as BP

SAFETY = "/home/humanoid/Programs/Humanoid_Simulation/core_ws/src/h12_safety_layer"
# What `h1_real_controller.launch.py` actually passes to `safety_node`. The
# other six configs in that directory are tighter; `relax` is the permissive end,
# so anything that trips here trips on every real configuration.
DEFAULT_CFG = "relax_safety_split.yaml"
STAGES = ("realpose", "maxreach")
ARMS = ("stand", "brace")


def limits(cfg):
    sys.path.insert(0, SAFETY)
    from h12_safety_layer.core.config import load_config
    from h12_safety_layer.core.joint_limits import JOINT_NAMES
    c = load_config(os.path.join(SAFETY, "config", cfg))["limits"]
    return JOINT_NAMES, c


def index(m, names):
    """(qpos adr, dof adr, actuator id) per safety-layer motor, in ITS order.

    The safety layer indexes by motor 0-26 and MuJoCo by its own joint order;
    the two agree here only because the MJCF was built from the same URDF. The
    lookup is by NAME so that a model reorder is a KeyError rather than a
    silently permuted torque check."""
    qa, va, ai = [], [], []
    for n in names:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if j < 0 or a < 0:
            raise SystemExit("model has no joint/actuator %r" % n)
        qa.append(m.jnt_qposadr[j]); va.append(m.jnt_dofadr[j]); ai.append(a)
    return np.array(qa), np.array(va), np.array(ai)


def replay(m, d, path, names, lim, qa, va, ai, hand, qtol):
    col, rows, meta = S.load_traj(path)
    qi = [col["qpos%d" % i] for i in range(m.nq)]
    vi = [col["qvel%d" % i] for i in range(m.nv)]
    ui = [col["ctrl%d" % i] for i in range(m.nu)]
    t = rows[:, col["time"]]
    qlo, qhi = lim["q_estop_limits"][:, 0], lim["q_estop_limits"][:, 1]
    dqm, taum = lim["dq_estop_limits"], lim["tau_estop_limits"]

    fr = m.actuator_forcerange[ai][:, 1]
    # Duty, not just peaks. A joint that touches its stop for one sample and a
    # joint pinned there for the whole episode look identical in a peak table,
    # and only the second one is a standing posture the hardware has to hold.
    n_at_stop = np.zeros(len(names))
    n_sat = np.zeros(len(names))
    trip = None
    strict = None
    # `q` is carried in RADIANS OF OVERSHOOT, not as a fraction: the knee band is
    # [-0.12, 2.19] and dividing by a 0.12 rad endpoint turns an 8 mrad solver
    # overshoot into "1.07x the limit", which reads as a violent violation and is
    # not one. The other two channels are fractions of their limit, where the
    # denominator is a real scale.
    worst = {"q": (0.0, None), "dq": (0.0, None), "tau": (0.0, None),
             "sat": (0.0, None)}
    tipx = np.zeros(len(rows))
    for k in range(len(rows)):
        d.qpos[:] = rows[k, qi]; d.qvel[:] = rows[k, vi]; d.ctrl[:] = rows[k, ui]
        mujoco.mj_forward(m, d)
        tipx[k] = d.site_xpos[hand][0]
        q, dq = d.qpos[qa], d.qvel[va]
        tau = d.actuator_force[ai]
        # Fractions of the way to each limit, so the three channels are
        # comparable and "how close did it get" survives a clean run.
        over = np.maximum(qlo - q, q - qhi)          # [rad], <=0 means inside
        fdq, ftau, fsat = np.abs(dq) / dqm, np.abs(tau) / taum, np.abs(tau) / fr
        n_at_stop += over > -0.005                   # within 5 mrad of a stop
        n_sat += fsat > 0.99
        for key, f in (("q", over), ("dq", fdq), ("tau", ftau), ("sat", fsat)):
            i = int(np.argmax(f))
            if f[i] > worst[key][0]:
                worst[key] = (float(f[i]), names[i])
        # Two verdicts from one pass: `strict` is the config as deployed
        # (0.0001 rad of slack), `trip` additionally requires the position
        # overshoot to exceed qtol so solver softness does not masquerade as a
        # safety event. dq and tau are identical under both.
        for tgt, thr in ((0, 0.0), (1, qtol)):
            if (strict, trip)[tgt] is not None:
                continue
            for key, f, cut, bad in (("q", over, thr, q),
                                     ("dq", fdq, 1.0, dq),
                                     ("tau", ftau, 1.0, tau)):
                i = int(np.argmax(f))
                if f[i] > cut:
                    hit = (float(t[k]), key, names[i], float(bad[i]), k)
                    if tgt == 0:
                        strict = hit
                    else:
                        trip = hit
                    break
    duty = dict(stop=n_at_stop / len(rows), sat=n_sat / len(rows))
    return t, tipx, trip, strict, worst, duty, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--json", default=None)
    ap.add_argument("--config", default=DEFAULT_CFG)
    ap.add_argument("--out", default=None)
    # MuJoCo joint limits are SOFT constraints, and the estop band is the URDF
    # range minus 0.0001 rad -- i.e. zero tolerance. So the solver's own
    # overshoot at a hard stop registers as a limit violation. Anything under
    # this is reported as solver softness rather than a posture the robot
    # actually adopted; 0.01 rad = 0.57 deg is well under joint-encoder
    # resolution arguments and far under anything visible.
    ap.add_argument("--q-tol", type=float, default=0.010,
                    help="[rad] overshoot below this is solver softness")
    a = ap.parse_args()
    jp = a.json or os.path.join(a.run, "analysis.json")
    res = json.load(open(jp))

    names, lim = limits(a.config)
    m, d = S.load()
    qa, va, ai = index(m, names)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, BVS.HAND_SITE)

    # Structural check first: MuJoCo clamps actuator force to `forcerange` and
    # joints to `jnt_range`, so two of the three channels may be unreachable in
    # sim BY CONSTRUCTION. Saying which is a stronger statement than a count of
    # zero trips, because it holds for rollouts nobody has run yet.
    caps = []
    for i, n in enumerate(names):
        fr = float(m.actuator_forcerange[ai[i]][1])
        caps.append(fr / lim["tau_estop_limits"][i])
    print("MODEL vs SAFETY LAYER (%s)" % a.config)
    print("  torque: MJCF forcerange / estop limit = %.2f-%.2f  -> %s"
          % (min(caps), max(caps),
             "torque trip IMPOSSIBLE in sim" if max(caps) <= 1.0
             else "reachable on %d joints" % sum(c > 1.0 for c in caps)))
    ql = lim["q_estop_limits"]
    same = sum(1 for i, n in enumerate(names)
               if abs(m.jnt_range[mujoco.mj_name2id(
                   m, mujoco.mjtObj.mjOBJ_JOINT, n)][1] - ql[i, 1]) < 2e-3)
    print("  position: %d/%d MJCF joint ranges equal the estop range "
          "(MuJoCo enforces them, so only solver overshoot can trip)"
          % (same, len(names)))
    print("  velocity: MuJoCo does NOT bound joint velocity -- the live channel\n")

    out = []
    for stage in STAGES:
        for arm in ARMS:
            g = [r for r in res if r.get("stage") == stage and r.get("arm") == arm
                 and "error" not in r and not r.get("fell")]
            rowsout = []
            for r in sorted(g, key=lambda z: z["tag"]):
                p = os.path.join(a.run, r["tag"] + ".csv")
                if not os.path.exists(p):
                    continue
                t, tipx, trip, strict, worst, duty, _ = replay(
                    m, d, p, names, lim, qa, va, ai, hand, a.q_tol)
                settled = float(tipx[t >= 0.6 * t[-1]].mean())
                rowsout.append(dict(tag=r["tag"], trip=trip, strict=strict,
                                    worst=worst, duty=duty,
                                    settled=settled,
                                    at_trip=(float(tipx[trip[4]]) if trip else None),
                                    cls=BP.contact_class(r)))
            if not rowsout:
                continue
            n = len(rowsout)
            tr = [z for z in rowsout if z["trip"]]
            st = [z for z in rowsout if z["strict"]]
            print("%-9s %-6s  n=%2d   tripped %2d/%2d = %3.0f%%   "
                  "(config as deployed, 0.0001 rad slack: %d/%d = %.0f%%)"
                  % (stage, arm, n, len(tr), n, 100.0 * len(tr) / n,
                     len(st), n, 100.0 * len(st) / n))
            for key, unit in (("dq", "x estop limit"), ("tau", "x estop limit"),
                              ("sat", "x MJCF forcerange"),
                              ("q", "rad past the joint stop")):
                w = max(z["worst"][key][0] for z in rowsout)
                who = [z["worst"][key][1] for z in rowsout
                       if z["worst"][key][0] == w][0]
                print("      peak %-4s %6.3f %-24s (%s)" % (key, w, unit, who))
            for key, lab in (("stop", "at a joint stop (<5 mrad)"),
                             ("sat", "at MJCF torque saturation")):
                mu = np.mean([z["duty"][key] for z in rowsout], axis=0)
                top = np.argsort(mu)[::-1][:3]
                print("      %% of episode %-26s %s" % (lab, ", ".join(
                    "%s %.0f%%" % (names[i], 100 * mu[i]) for i in top
                    if mu[i] > 0.01) or "none above 1%"))
            rf = np.array([z["settled"] for z in rowsout]) * 100
            print("      settled tip x  %6.1f cm   SD %4.1f   range %.1f-%.1f"
                  % (rf.mean(), rf.std(ddof=1), rf.min(), rf.max()))
            if tr:
                t0 = np.array([z["trip"][0] for z in tr])
                print("      first trip  %.2f-%.2f s (median %.2f)   channels %s"
                      % (t0.min(), t0.max(), float(np.median(t0)),
                         dict((k, sum(1 for z in tr if z["trip"][1] == k))
                              for k in ("q", "dq", "tau"))))
                j = {}
                for z in tr:
                    j[z["trip"][2]] = j.get(z["trip"][2], 0) + 1
                print("      joints      %s" % j)
                rs = np.array([z["at_trip"] for z in tr]) * 100
                rf = np.array([z["settled"] for z in tr]) * 100
                print("      tip x at trip %5.1f cm  vs settled %5.1f cm "
                      "(%+.1f cm of reach never happens)"
                      % (rs.mean(), rf.mean(), rs.mean() - rf.mean()))
            out.append(dict(stage=stage, arm=arm, n=n, tripped=len(tr),
                            rows=[dict(tag=z["tag"], trip=z["trip"],
                                       worst={k: list(v) for k, v in
                                              z["worst"].items()},
                                       settled=z["settled"],
                                       at_trip=z["at_trip"], cls=z["cls"])
                                  for z in rowsout]))
            print()
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
