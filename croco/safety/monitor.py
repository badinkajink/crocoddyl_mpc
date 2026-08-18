"""What the H1-2 safety layer WOULD have done to this run.

WHY A MONITOR AND NOT THE LAYER ITSELF. `h12_safety_layer` is a DDS relay: it
takes `rt/safety/lowcmd_in`, clips it, watches `rt/lowstate`, and republishes
to `rt/lowcmd`. Putting it in the path is one topic name (see
`--via-safety`) and is the real thing. But its e-stop LATCHES -- `_estop` is
set once and `_publisher_loop` then emits `make_estop_cmd` (mode 0, kp = kd =
tau = 0) on every subsequent tick, with no clear path short of restarting the
process. On the twin that is a robot that goes limp mid-maneuver and stays
limp. That is correct behaviour for hardware and ruinous for a study run,
which is why the layer is opt-in and this is the thing you turn on first.

So: same limits, same predicates, same YAML, no authority. The monitor reads
the state the control loop already read and answers one question per period --
"would the layer have tripped here, and why" -- and the answer lands on the
loop's log row and in the panel as a red mark on the period plot. You find out
that the brace's ankle torque crosses the estop line 1.2 s in WITHOUT having
discovered it by watching the robot collapse.

WHERE THE NUMBERS COME FROM. Not from here. The limit tables
(`URDF_POSITION_LIMITS`, `URDF_VELOCITY_LIMITS`, `URDF_TORQUE_LIMITS`) and the
ratio arithmetic that turns a YAML into bounds both live in the safety layer
package and are IMPORTED, never copied -- a vendored copy of 27 torque limits
is a copy that goes stale silently and then lies to you about the margin. The
package is found the same way CL_Assets is: an env var, else the sibling
checkout in the superproject.

WHAT IS CHECKED, AND WHAT THAT MISSES.

  * e-stop, from the STATE: q outside the estop range, |dq| over, |tau| over,
    or any non-finite. This is `check_estop_limits` transcribed onto arrays,
    because our State is arrays and its is a `LowState_`.
  * clipping, from the COMMAND: how many of q/dq/tau/kp/kd the layer would
    have altered before forwarding. Clipping does not stop anything, but a
    command that is being clipped is a command the robot is not receiving, and
    a run where that count is nonzero is not the run the plan describes.
  * NOT checked: the estop hardware line (`h12/estop_status_raw`, a physical
    button), the upper-body staleness watchdog (split mode only), and CRC. All
    three are properties of the deployment, not of the trajectory, and
    pretending to predict them here would be theatre.
"""
from __future__ import annotations

import os
import sys

import numpy as np

#: Searched in order after $H12_SAFETY_LAYER, relative to this repo's root.
PKG_CANDIDATES = (
    "../core_ws/src/h12_safety_layer",
    "../../core_ws/src/h12_safety_layer",
)

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def find_safety_layer():
    """The h12_safety_layer checkout, or None. Never raises."""
    explicit = os.environ.get("H12_SAFETY_LAYER")
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for rel in PKG_CANDIDATES:
        cand = os.path.normpath(os.path.join(_ROOT, rel))
        if os.path.isdir(os.path.join(cand, "h12_safety_layer", "core")):
            return cand
    return None


def _import_safety_layer():
    """Import the package's core modules, adding its checkout to sys.path.

    Imported lazily and by path rather than declared as a dependency: this
    repo does not otherwise need ROS, a colcon workspace, or the H1-2 robot,
    and making the safety layer a hard requirement would mean the study could
    not be run by anyone who has only the simulator.

    THE SOURCE CHECKOUT WINS OVER AN AMBIENT IMPORT. A sourced ROS workspace
    puts an INSTALLED copy on PYTHONPATH (`ws_ctrl/install/h12_safety_layer/
    lib/python3.10/site-packages`) which is a build of some past state of the
    source, under a different Python minor version, and silently preferring it
    would mean the limits being checked are not the limits in the tree
    everyone is editing. So the explicit path and the superproject sibling are
    put in FRONT, and the ambient copy is the fallback that keeps this working
    on a robot PC where the source is not checked out.
    """
    root = find_safety_layer()
    if root and root not in sys.path:
        sys.path.insert(0, root)
    try:
        from h12_safety_layer.core import config as cfg
        from h12_safety_layer.core import joint_limits as jl
    except ImportError as exc:
        if "yaml" in str(exc):
            raise ImportError(
                "h12_safety_layer needs PyYAML to read its config, and this "
                "interpreter has none: pip install pyyaml. (--safety reads "
                "that package's YAML rather than carrying its own copy of 27 "
                "joint limits.)")
        if root is None:
            raise ImportError(
                "h12_safety_layer not found (%s). It is a ROS package in the "
                "GOLEM superproject (core_ws/src/h12_safety_layer); this repo "
                "reads its limit tables rather than copying them. Set "
                "H12_SAFETY_LAYER=<path to that package>, or drop --safety."
                % exc)
        raise
    return cfg, jl


def resolve_config(name):
    """Accept a path, or a bare config name from the layer's own config/ dir."""
    if os.path.isfile(name):
        return name
    cand = name if name.endswith((".yaml", ".yml")) else name + ".yaml"
    if os.path.isfile(cand):
        return cand
    root = find_safety_layer()
    if root:
        p = os.path.join(root, "config", os.path.basename(cand))
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "no safety config %r. Give a path, or a bare name from "
        "h12_safety_layer/config (e.g. default_safety_full, "
        "sim_safety_split, tight_safety_full)." % name)


class SafetyMonitor:
    """Evaluate the safety layer's limits on every control period.

    Use as a `ControlLoop` on_step hook. It annotates the row in place, which
    is what carries the verdict to both the run artifact and the panel without
    either of them having to know this class exists.
    """

    #: The joint order both sides must agree on. The DDS plant asserts the
    #: same thing for the wire; this asserts it for the limits, because a
    #: monitor that checks the ankle's limit against the elbow's torque is
    #: worse than no monitor.
    def __init__(self, config, joint_names=None):
        cfg, jl = _import_safety_layer()
        self.config_path = resolve_config(config)
        conf = cfg.load_config(self.config_path)
        self.limits = conf["limits"]
        self.mode = conf.get("mode")
        self.topics = conf.get("topics", {})
        self.estop_enabled = bool(conf.get("estop", {}).get("enabled", False))
        self.pkg_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(cfg.__file__)))
        self.nu = int(jl.MOTOR_COUNT)
        self.names = list(jl.JOINT_NAMES)
        if joint_names is not None and list(joint_names) != self.names:
            raise ValueError(
                "joint order disagrees with h12_safety_layer.JOINT_NAMES -- "
                "the limits would be applied to the wrong joints. First "
                "mismatch: %s" % next(
                    (("%d: %s != %s" % (i, a, b))
                     for i, (a, b) in enumerate(zip(joint_names, self.names))
                     if a != b), "length %d vs %d"
                    % (len(joint_names), len(self.names))))
        # Counters over the session, not the episode: the question "did this
        # configuration ever cross the line" outlives any one run.
        self.trips = 0
        self.first_trip = None        # (row t, reason) -- the one that matters
        self.clipped_periods = 0
        self.clip_kinds = {}          # kind -> entries clipped, over the session
        self.worst_clip = None        # (t, description) of the first tau clip
        self.no_tau = 0               # periods where the plant had no torque

    # -- the predicates ----------------------------------------------------
    def estop_reason(self, st):
        """Why the layer would trip on this state, or None. Order = the layer's."""
        L, n = self.limits, self.nu
        q, v = np.asarray(st.q, float), np.asarray(st.v, float)
        tau = None if st.tau is None else np.asarray(st.tau, float)
        if q.size < n or v.size < n:
            return None               # not a full-body state; nothing to judge
        if not (np.all(np.isfinite(q[:n])) and np.all(np.isfinite(v[:n]))):
            return "state contains non-finite values"
        lo, hi = L["q_estop_limits"][:, 0], L["q_estop_limits"][:, 1]
        bad = np.where((q[:n] < lo) | (q[:n] > hi))[0]
        if bad.size:
            i = int(bad[0])
            return ("motor %d (%s) q %.4f outside estop range [%.4f, %.4f]"
                    % (i, self.names[i], q[i], lo[i], hi[i]))
        bad = np.where(np.abs(v[:n]) > L["dq_estop_limits"])[0]
        if bad.size:
            i = int(bad[0])
            return ("motor %d (%s) dq %.3f over estop limit %.3f"
                    % (i, self.names[i], v[i], L["dq_estop_limits"][i]))
        if tau is None or tau.size < n:
            self.no_tau += 1
            return None
        if not np.all(np.isfinite(tau[:n])):
            return "measured torque contains non-finite values"
        bad = np.where(np.abs(tau[:n]) > L["tau_estop_limits"])[0]
        if bad.size:
            i = int(bad[0])
            return ("motor %d (%s) tau %.1f over estop limit %.1f"
                    % (i, self.names[i], tau[i], L["tau_estop_limits"][i]))
        return None

    #: Clip categories, in the order clip_low_cmd applies them.
    CLIP_KINDS = ("q", "dq", "tau", "kp", "kd")

    def clip_count(self, cmd):
        """How many command entries the layer would alter, BY KIND.

        Per kind rather than as one number, because the two kinds that fire
        mean different things. MEASURED on the certified brace+reach against
        `default_safety_full`, in process, 198 periods: 16-18 periods clipped,
        20-23 `tau` entries and 25-26 `dq`; q, kp and kd never. The spread is
        real and not a flaw in the count -- the offending torques sit within
        10% of their band, and latency compensation makes the command a
        function of the MEASURED solve time, so which side of the line a
        borderline period lands on moves with machine load. Neither kind is
        noise.

          tau: the layer's `torque_ratio: 0.60` DERATES the URDF limits (knee
            270 -> 180 N.m, hip roll 180 -> 120, shoulder yaw 18 -> 10.8)
            while the MPC clamps its own output to the MODEL's limits. So the
            plan is free to ask for a torque the relay will not pass on. First
            one here is the shoulder yaw at 11.7 against a 10.8 band.
          dq: `velocity_ratio: 0.10` puts the commanded-velocity band as low
            as 0.9 rad/s, and the maneuver's own joint velocities exceed that
            -- the plan's v_des is a feedforward, not a request, but the layer
            does not know the difference and flattens it.

        Neither stops anything; clipping is silent by design. But a command
        that is being clipped is not the command the robot receives, so a run
        with a nonzero count is not the run the plan describes.
        """
        L, n = self.limits, self.nu
        out = dict.fromkeys(self.CLIP_KINDS, 0)
        if cmd is None or cmd.nu < n:
            return out, None
        lo, hi = L["q_clip_limits"][:, 0], L["q_clip_limits"][:, 1]
        over = {
            "q": (cmd.q_des[:n] < lo) | (cmd.q_des[:n] > hi),
            "dq": np.abs(cmd.v_des[:n]) > L["dq_clip_limits"],
            "tau": np.abs(cmd.tau_ff[:n]) > L["tau_clip_limits"],
            "kp": cmd.kp[:n] > L["kp_clip_max"],
            "kd": cmd.kd[:n] > L["kd_clip_max"],
        }
        # The worst OFFENDER, by how far past its own band it is -- not by
        # magnitude. A knee 5 N.m over a 180 band is a rounding error; a wrist
        # 5 over 10.8 is the plan asking for twice what it may have.
        val = {"q": cmd.q_des, "dq": cmd.v_des, "tau": cmd.tau_ff,
               "kp": cmd.kp, "kd": cmd.kd}
        band = {"dq": L["dq_clip_limits"], "tau": L["tau_clip_limits"],
                "kp": L["kp_clip_max"], "kd": L["kd_clip_max"]}
        unit = {"tau": " N.m", "dq": " rad/s"}
        worst, worst_by = None, 0.0
        for kind, mask in over.items():
            out[kind] = int(np.count_nonzero(mask))
            if not out[kind] or kind not in band:
                continue
            excess = (np.abs(val[kind][:n]) - band[kind]) / np.maximum(
                band[kind], 1e-9)
            i = int(np.argmax(np.where(mask, excess, -np.inf)))
            if excess[i] > worst_by:
                worst_by = float(excess[i])
                worst = ("%s %s %.1f > %.1f%s (%.0f%% over)"
                         % (self.names[i], kind, val[kind][i], band[kind][i],
                            unit.get(kind, ""), 100.0 * excess[i]))
        return out, worst

    # -- the hook ----------------------------------------------------------
    def __call__(self, row, st, cmd):
        """`ControlLoop` on_step. Must be cheap and must not raise."""
        try:
            why = self.estop_reason(st)
            per_kind, worst = self.clip_count(cmd)
        except Exception as exc:                                 # noqa: BLE001
            row["safety_error"] = str(exc)
            return
        clipped = sum(per_kind.values())
        row["estop"] = bool(why)
        row["estop_why"] = why
        row["safety_clip"] = clipped
        if clipped:
            self.clipped_periods += 1
            for k, v in per_kind.items():
                if v:
                    self.clip_kinds[k] = self.clip_kinds.get(k, 0) + v
            if worst and self.worst_clip is None:
                self.worst_clip = (row.get("t"), worst)
        if why:
            self.trips += 1
            if self.first_trip is None:
                self.first_trip = (row.get("t"), why)

    def reset_episode(self):
        """Nothing: the counters are per session on purpose. Here for symmetry."""

    def describe(self):
        return dict(config=self.config_path, package=self.pkg_dir,
                    mode=self.mode,
                    estop_enabled_in_config=self.estop_enabled,
                    hardware_estop_topic=self.topics.get("estop_topic"),
                    acting=False)

    def summary(self):
        """What the run artifact carries. `first_trip` is the load-bearing one.

        The layer's e-stop latches, so a run with three trips did not survive
        two of them -- it would have gone limp at the first and reported the
        rest against a robot that was already falling.
        """
        t, why = self.first_trip or (None, None)
        return dict(safety_config=self.config_path,
                    safety_package=self.pkg_dir,
                    safety_acting=False,
                    safety_estop_periods=self.trips,
                    safety_first_estop_t=t,
                    safety_first_estop_why=why,
                    safety_clipped_periods=self.clipped_periods,
                    safety_clip_kinds=(self.clip_kinds or None),
                    safety_first_clip=(self.worst_clip or (None, None))[1],
                    safety_periods_without_torque=self.no_tau)
