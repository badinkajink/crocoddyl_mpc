#!/usr/bin/env python3
"""The braced lean, run against a plant on the far side of Unitree DDS.

This is `croco_replay.py --ctrl mpc` with ONE thing changed: the plant. Same
plan, same OCP, same MPC, same gains. What is different is everything the
in-process replay could not be wrong about --

    the controller no longer owns the clock          (DDSPlant.OWNS_CLOCK=False)
    the state is a MESSAGE and can be old            (State.age, the watchdog)
    the command is a MESSAGE and can be dropped      (lowcmd timeout -> collapse)
    the 20 ms period is real, and a 12 ms solve eats most of it

-- which is the whole point. A replay proves the plan survives the physics; this
proves the controller survives the deployment. They are different claims and the
first has never implied the second.

WHAT IT DOES NOT PROVE. The plant here is `lean_twin`, i.e. the SAME MJCF the
plan was solved against, served over the wire. Scene parity with
`h1_robocasa`/`h1_mujoco` (which carry a kitchen, not the lean table) is the
next problem and is deliberately not mixed into this one: a failure here is a
deployment failure and cannot be blamed on the scene.

BASE POSE. The OCP needs one and `rt/lowstate` does not carry one -- no robot
has it. For this stage it is read from the twin's `--publish-truth` channel,
which is GROUND TRUTH and is why `--base truth` has to be typed. The real
estimator (h12_deploy_mjpc's estimator_node, FAST-LIO, the tag anchor) plugs
into the same `base_source` hook with nothing else changing, and the gap between
those two numbers is the next thing worth measuring.

usage:
  # terminal 1
  python -m croco.twin.lean_twin --model $LEAN_TASK_DIR/Lean_H12_Magpie.xml \
      --key stand --publish-truth
  # terminal 2
  studies/croco_twin.py --dir runs/.../grid/<cell> --tag elbow_palm --base truth
"""
import argparse
import ctypes
import json
import os
import sys
import threading
import time

sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

import numpy as np                                              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "croco_ext"))
sys.path.insert(0, os.path.join(HERE, ".."))

import croco_bridge as cb                                       # noqa: E402
import contact_select as cs                                     # noqa: E402
import croco_replay as cr                                       # noqa: E402

from croco.control.mpc import MPC                               # noqa: E402
from croco.plant.dds_plant import (DDSPlant, PollingReceiver,   # noqa: E402
                                   assert_joint_order)
from croco.runtime.loop import ControlLoop, LoopConfig          # noqa: E402


# --------------------------------------------------------------- base --- #
class TruthBase:
    """Subscribe to the twin's ground-truth base pose (`rt/sim_state`).

    NAMED `truth` ON THE COMMAND LINE ON PURPOSE. This is the one privileged
    input in the loop, it exists so the deployment plumbing can be tested with
    the estimator held at perfect, and every result taken with it has to say so.
    Swapping in a real estimator means replacing this class and nothing else.

    POLLED, for the reason in dds_plant.py: a Python callback cannot run while
    crocoddyl holds the GIL. This channel was the worse of the two offenders --
    it carries JSON, so the callback path spent a `json.loads` per sample at the
    twin's 500 Hz, all of it contending for the same GIL the solver is sitting
    on. Polling parses ONE document per control period, and parses the newest.
    """

    def __init__(self, recv="poll"):
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        self._v = None
        self._lock = threading.Lock()
        self._recv = None
        if recv == "poll":
            self._recv = PollingReceiver("rt/sim_state", String_)
        else:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            self._sub = ChannelSubscriber("rt/sim_state", String_)
            self._sub.Init(self._on, 10)

    def _on(self, msg):
        self._decode(msg)

    def _decode(self, msg):
        try:
            d = json.loads(msg.data)
        except Exception:                                       # noqa: BLE001
            return
        with self._lock:
            self._v = (np.array(d["base_pos"]), np.array(d["base_quat"]),
                       np.array(d["base_linvel"]), np.array(d["base_angvel"]),
                       time.monotonic())

    def __call__(self):
        msg = self._recv.latest() if self._recv is not None else None
        if msg is not None:
            self._decode(msg)
        with self._lock:
            v = self._v
        if v is None:
            return None
        p, q, lv, av, stamp = v
        return p, q, lv, av, max(0.0, time.monotonic() - stamp)

    def wait(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self() is not None:
                return True
            time.sleep(0.01)
        raise TimeoutError(
            "no rt/sim_state in %.1f s -- start lean_twin with --publish-truth, "
            "or the OCP has no base pose to plan from." % timeout)


# ---------------------------------------------------------------- run --- #
def build(args):
    """The MPC and the reference plan, exactly as croco_replay builds them."""
    plan = json.load(open(os.path.join(args.dir, "plan_%s.json" % args.tag)))
    ocp, _ = cr.build_ocp(plan, args.dir)
    problem = ocp.build(dt=plan["dt"], n_approach=plan["n_approach"],
                        n_braced=plan["n_braced"],
                        n_return=plan.get("n_return", 0),
                        dwell=plan.get("dwell", 0), cones=plan["cones"])
    xs = np.load(os.path.join(args.dir, "xs_%s.npy" % args.tag))
    us = np.load(os.path.join(args.dir, "us_%s.npy" % args.tag))
    mpc = MPC(ocp, list(problem.runningModels), problem.terminalModel,
              horizon=args.horizon, iters=args.iters, xs_plan=xs, us_plan=us,
              n_alphas=args.alphas, nthreads=args.threads)
    return plan, mpc, xs, us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="grid cell / run directory")
    ap.add_argument("--tag", default="elbow_palm")
    ap.add_argument("--horizon", type=int, default=35)
    ap.add_argument("--iters", type=int, default=1)
    ap.add_argument("--alphas", type=int, default=0)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--domain", type=int, default=1)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--plant", choices=["dds", "mujoco"], default="dds",
                    help="'mujoco' runs the SAME ControlLoop and the SAME "
                         "policy against in-process physics, i.e. with zero "
                         "latency and no wire. It is the control for this "
                         "experiment: if the maneuver survives there and dies "
                         "over DDS, the deployment is what broke it; if it dies "
                         "in both, the bug is in this file and not on the wire.")
    ap.add_argument("--base", choices=["truth", "none"], default="none",
                    help="'truth' reads the TWIN'S GROUND TRUTH base pose. "
                         "There is no estimator in this loop yet, so 'none' "
                         "cannot run the MPC -- it is here to make the "
                         "dependency explicit rather than implicit.")
    ap.add_argument("--bringup", action="store_true",
                    help="run the warmup/ramp/hold/blend phases before the "
                         "maneuver, as the MJPC deploy node does on hardware")
    ap.add_argument("--stale-ms", type=float, default=50.0,
                    help="watchdog threshold. The MJPC deploy node's 50 ms was "
                         "chosen for a 200 Hz loop; this one runs at 50 Hz with "
                         "a ~16 ms solve, so the two are not obviously the same "
                         "setting. Raise it to test whether a fall is the "
                         "watchdog or the latency -- not to make it go away.")
    ap.add_argument("--recv", default="poll", choices=("poll", "callback"),
                    help="how lowstate/sim_state are received. `poll` takes the "
                         "newest sample in the control thread; `callback` is "
                         "unitree_sdk2py's listener+queue threads, which starve "
                         "while the solver holds the GIL. Kept only for the A/B.")
    ap.add_argument("--gui", nargs="?", type=int, const=8770, default=None,
                    metavar="PORT",
                    help="serve the live panel (default port 8770): solve time "
                         "against the period, cost per term, state age, and "
                         "sliders for every cost weight. Weight edits are "
                         "applied BETWEEN periods and recorded in --out, "
                         "because a retuned run that does not say so is not "
                         "reproducible.")
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--emit-qpos0", default=None,
                    help="write the plan's start qpos to this file and exit. "
                         "Feed it to `lean_twin --qpos0`: the twin must begin "
                         "where the plan begins, and no keyframe is that pose.")
    args = ap.parse_args()

    if args.emit_qpos0:
        plan = json.load(open(os.path.join(args.dir, "plan_%s.json" % args.tag)))
        xs = np.load(os.path.join(args.dir, "xs_%s.npy" % args.tag))
        m, _ = cs.load(ik_margin=0.0)
        q = cb.pin_to_mj(xs[0][:cb.NQ_ROBOT], cs.start_qpos(m, plan["start"]))
        os.makedirs(os.path.dirname(os.path.abspath(args.emit_qpos0)) or ".",
                    exist_ok=True)
        np.savetxt(args.emit_qpos0, q)
        print("[croco_twin] wrote %s (%d) -- pass it to lean_twin --qpos0"
              % (args.emit_qpos0, q.size))
        return 0

    if args.plant == "mujoco":
        args.base = "truth"          # in-process physics IS the truth
    if args.base == "none":
        raise SystemExit(
            "--base none: the lean OCP needs a floating-base pose and "
            "rt/lowstate does not carry one. Pass --base truth to use the "
            "twin's ground truth (and say so in whatever you report), or plug "
            "an estimator into TruthBase's place.")

    plan, mpc, xs, us = build(args)
    dt_plan = plan["dt"]
    nq = cb.NQ_ROBOT

    # Gains and limits come off the SAME model the plan was solved against.
    m, _d = cs.load(ik_margin=0.0)
    assert_joint_order(m, nu=27)
    kp, kd = cr.servo_gains(m)
    tau_lim = cs.torque_limits(m)

    if args.plant == "mujoco":
        from croco.plant.mujoco_plant import MuJoCoPlant
        m2, d2 = cs.load(ik_margin=0.0)
        q0 = cb.pin_to_mj(xs[0][:nq], cs.start_qpos(m2, plan["start"]))
        d2.qpos[:] = q0
        d2.qvel[:] = 0.0
        import mujoco as _mj
        _mj.mj_forward(m2, d2)
        plant = MuJoCoPlant(m2, d2, sense=None, tau_limit=tau_lim, nu=27)
        base = None
    else:
        pass
    # ORDER MATTERS: DDSPlant is what calls ChannelFactoryInitialize, and a
    # subscriber built before the participant exists fails inside cyclonedds as
    # "'NoneType' object has no attribute '_ref'", which names neither the
    # participant nor the ordering.
    if args.plant == "dds":
        plant = DDSPlant(network_interface=args.iface, domain_id=args.domain,
                         twin_dt=None, base_source=None, tau_limit=tau_lim,
                         q_range=(m.jnt_range[1:28, 0].copy(),
                                  m.jnt_range[1:28, 1].copy()),
                         recv=args.recv)
        base = TruthBase(recv=args.recv)
        plant.base_source = base
        print("[croco_twin] waiting for the twin ...")
        plant.wait_for_state(timeout=15.0)
        base.wait(timeout=15.0)
        print("[croco_twin] twin is up: lowstate + rt/sim_state (GROUND TRUTH "
              "base)")

    stats = dict(steps=0, mpc_none=0)

    first = {}

    def policy(t, st):
        """(q_des, v_des, tau_ff) for the plant's joints, from the MPC.

        `k` is derived from the plant's clock rather than counted, so a missed
        period advances the plan by a period instead of replaying it -- the
        maneuver is a function of time, not of how many times we managed to
        solve.
        """
        k = int(round(t / dt_plan))
        if k >= len(us):
            return None
        if not first:
            # WHERE IS THE ROBOT WHEN THE MANEUVER STARTS? The plan assumes x0.
            # The twin has been holding a pose for as long as this process took
            # to build its OCP -- tens of seconds -- and a hold is not a freeze:
            # the floating base is not held by anything. If the robot has crept,
            # the maneuver begins from somewhere it was never planned from, and
            # that is an initial-condition failure wearing a controller's
            # clothes.
            q0p = cb.pin_to_mj(xs[0][:nq], cs.start_qpos(m, plan["start"]))
            first["dq_max_rad"] = float(np.max(np.abs(st.q - q0p[7:34])))
            first["dq_rms_rad"] = float(np.sqrt(np.mean((st.q - q0p[7:34]) ** 2)))
            first["dbase_mm"] = float(1e3 * np.linalg.norm(st.base_pos - q0p[0:3]))
            first["dquat"] = float(np.linalg.norm(st.base_quat - q0p[3:7]))
            first["v_max"] = float(np.max(np.abs(st.v)))
            print("[croco_twin] at first command: dq_max %.4f rad  dq_rms %.4f  "
                  "base %.1f mm  dquat %.4f  |v|max %.3f rad/s"
                  % (first["dq_max_rad"], first["dq_rms_rad"], first["dbase_mm"],
                     first["dquat"], first["v_max"]))
        qpos = np.concatenate([st.base_pos, st.base_quat, st.q])
        R = _quat_to_mat(st.base_quat)
        qvel = np.concatenate([st.base_linvel, st.base_angvel, st.v])
        x_meas = np.concatenate([cb.mj_to_pin(qpos), cb.mj_to_pin_v(qvel, R)])
        u0, xs1 = mpc(min(k, len(us) - 1), x_meas)
        stats["steps"] += 1
        if u0 is None:
            stats["mpc_none"] += 1
            return xs[k][7:nq], np.zeros(27), us[k]
        return xs1[:nq][7:], xs1[nq:][6:], np.clip(u0, -tau_lim, tau_lim)

    cfg = LoopConfig(ctrl_hz=1.0 / dt_plan, stale_s=args.stale_ms * 1e-3)
    stance = cs.start_qpos(m, plan["start"])[7:] if args.bringup else None
    panel = None
    if args.gui:
        from croco.gui import Panel
        panel = Panel(mpc, port=args.gui, period_ms=1e3 * dt_plan)
        print("[croco_twin] panel on %s -- open it before the maneuver starts, "
              "the run is only %.1f s long" % (panel.url, len(us) * dt_plan))
    loop = ControlLoop(plant, policy, stance=stance, cfg=cfg,
                       on_step=None if panel is None else panel.on_step)
    print("[croco_twin] %.0f Hz, horizon %d, %d iter(s), %d thread(s), "
          "%s bring-up" % (cfg.ctrl_hz, args.horizon, args.iters, args.threads,
                           "with" if args.bringup else "no"))
    t0 = time.monotonic()
    try:
        log = loop.run(kp, kd, max_seconds=args.max_seconds)
    except KeyboardInterrupt:
        log = loop.log
    finally:
        plant.close()

    if args.plant == "mujoco":
        print("[croco_twin] in-process outcome: pelvis z %.4f m  %s"
              % (plant.d.qpos[2], "FELL" if plant.d.qpos[2] < 0.55 else "upright"))
    solves = [r["solve_ms"] for r in log if "solve_ms" in r]
    ages = [1e3 * r["age"] for r in log if "age" in r]
    out = dict(
        wall_s=time.monotonic() - t0,
        periods=len(log), mpc_steps=stats["steps"],
        overruns=getattr(loop, "overruns", None),
        worst_overrun_ms=1e3 * getattr(loop, "worst_overrun_s", 0.0),
        watchdog_trips=getattr(loop, "watchdog_trips", None),
        safe_periods=sum(1 for r in log if r.get("phase") == "safe"),
        tau_saturated=sum(r.get("tau_sat", 0) for r in log),
        q_clipped=sum(r.get("q_clip", 0) for r in log),
        solve_ms_mean=float(np.mean(solves)) if solves else None,
        solve_ms_p95=float(np.percentile(solves, 95)) if solves else None,
        age_ms_p50=float(np.percentile(ages, 50)) if ages else None,
        age_ms_p95=float(np.percentile(ages, 95)) if ages else None,
        age_ms_max=float(np.max(ages)) if ages else None,
        stale_ms=args.stale_ms,
        nthreads_effective=int(mpc.problem.nthreads),
        recv=args.recv,
        **({} if panel is None else panel.summary()),
        recv_samples_per_poll=(
            None if getattr(plant, "recv_polls", 0) == 0
            else round(plant.recv_samples / plant.recv_polls, 2)),
        recv_empty_polls=getattr(plant, "recv_empty", None),
        base_source="GROUND TRUTH (rt/sim_state)")
    if panel is not None and panel.dirty:
        print("[croco_twin] WEIGHTS WERE CHANGED LIVE (%d edits). This run is "
              "NOT the plan's cost function; see gui_weight_changes in --out."
              % len(panel.changes))
    print("[croco_twin] " + json.dumps(out, indent=1))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(dict(summary=out, log=log), open(args.out, "w"), indent=1)
        print("[croco_twin] wrote %s" % args.out)
    if panel is not None:
        # HOLD THE PANEL OPEN. The maneuver is four seconds long and the
        # process would otherwise exit before a browser could finish loading
        # the page, which is how the first run of this was measured as
        # "HTTP 000". The run is over; the numbers are what you came to look at.
        print("[croco_twin] panel still serving at %s -- Ctrl-C to exit"
              % panel.url)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        panel.close()
    return 0


def _quat_to_mat(q):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


if __name__ == "__main__":
    raise SystemExit(main())
