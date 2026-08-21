"""Watch the planner live, retune it live, and keep the run reproducible.

THE MVP IS THREE THINGS, and only one of them was ever in doubt:

  watch      `ControlLoop` already calls `on_step(row, state, cmd)` once per
             control period with everything a display needs. The panel is a
             CONSUMER of a hook that exists, not a change to the loop.
  plot       rolling traces in the browser, a few dozen lines of canvas.
  retune     the one that could have been hard, and is not: crocoddyl's
             `CostModelSum` exposes its items, and
             `costs.costs["reg"].weight = 7.5` mutates a BUILT model in place.
             No rebuild, so a slider is a slider and not a restart button.

WHAT IT DELIBERATELY IS NOT. No task editing, no scene tree, no saved
configurations, no profiler. That is the line MJPC's GUI crossed, after which
it was something to maintain rather than something to use.

REPRODUCIBILITY IS NOT OPTIONAL HERE. A panel that retunes weights live makes
it trivial to produce a number nobody can reproduce, so every change is
timestamped into `self.changes`, `dirty` goes true the first time one lands,
and `summary()` carries both. A run whose weights differ from its plan's has to
say so, for the same reason `--base truth` is something you have to type.

WEIGHT CHANGES ARE APPLIED BETWEEN PERIODS, NEVER MID-SOLVE. The browser thread
only enqueues; `drain()` runs at the top of `on_step`, which is the control
thread, outside `solver.solve`. Writing a weight into a model that the solver is
reading is a data race whose symptom is a bad step, not a crash.
"""
from __future__ import annotations

import collections
import os
import threading
import time

from .ws import Server

PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "page.html")


def _cost_sum(model):
    """The CostModelSum inside an IntegratedActionModel, or None.

    Integrated -> differential -> costs, with the differential absent on some
    model types. Written defensively because the panel must never be the reason
    a controller stops.
    """
    diff = getattr(model, "differential", None)
    return None if diff is None else getattr(diff, "costs", None)


class Panel:
    """Telemetry out, weights back, for one `ControlLoop` + `MPC`.

    Pass `panel.on_step` to `ControlLoop(..., on_step=...)`. Everything else
    happens on the server's threads.
    """

    def __init__(self, mpc=None, port=8770, host="127.0.0.1", every=1,
                 period_ms=None, history=4000, on_command=None, config=None):
        """`mpc` MAY BE None AT CONSTRUCTION, and usually should be.

        Building the OCP takes ~20 s, and a panel that only exists afterwards
        cannot show you that it is happening -- which is how "the page is
        blank" and "the solver is still building" became indistinguishable.
        The session constructs the panel first and calls `set_mpc` when a task
        finishes building; switching task calls it again, so the weight
        sliders always belong to the OCP that is actually running.
        """
        self.mpc = mpc
        self.period_ms = period_ms      # drawn as the red line the solve must stay under
        self.every = max(1, int(every))          # send every Nth period
        self.n = 0
        self.changes = []                        # [(t, name, old, new)]
        self.dirty = False
        self._q = []                             # pending weight edits
        self._qlock = threading.Lock()
        self._t0 = time.monotonic()
        # EVERY STEP MESSAGE, KEPT. A browser that attaches after the maneuver
        # is replayed the whole run instead of being shown an empty chart --
        # see Server.__init__ for why that was the normal case rather than the
        # unlucky one. 4000 periods is 80 s at 50 Hz, i.e. every run this study
        # produces, at a few hundred bytes each.
        self.hist = collections.deque(maxlen=int(history))
        self._video = None              # (content_type, bytes), set after the run
        self.on_command = on_command    # (name, payload) from the browser
        self.config = dict(config or {})   # read-only facts shown in the panel
        self.session = None             # last session state, replayed on connect
        self.server = Server(PAGE, on_message=self._on_message,
                             host=host, port=port,
                             on_connect=self._backlog,
                             routes={"/video.mp4": self._serve_video})

    # -- backlog / video ---------------------------------------------------

    def set_mpc(self, mpc):
        """Point the weight sliders at a different OCP (task switch, rebuild)."""
        self.mpc = mpc
        self.server.broadcast(dict(type="weights", weights=self.weights()))

    def set_session(self, state):
        """Remember the session's state so a late browser sees it too."""
        self.session = state

    def _backlog(self):
        """What a newly-connected browser is sent, in order.

        The `replay` marker lets the page say "192 recorded periods" rather
        than "live": a panel that cannot tell the difference invites reading a
        finished run as a running one.
        """
        out = [dict(type="config", config=self.config),
               dict(type="replay", n=len(self.hist))]
        out.extend(self.hist)
        if self.session is not None:
            out.append(self.session)
        if self.mpc is not None:
            out.append(dict(type="weights", weights=self.weights()))
        if self._video is not None:
            out.append(dict(type="video", url="/video.mp4"))
        return out

    def _serve_video(self):
        return self._video

    def set_video(self, path):
        """Publish a rendered mp4 to the page. Read into memory once, on
        purpose: the file is ~1 MB and the alternative is a socket handler
        that can be broken by deleting a file mid-session."""
        try:
            with open(path, "rb") as fh:
                self._video = ("video/mp4", fh.read())
        except OSError:
            return False
        self.server.broadcast(dict(type="video", url="/video.mp4"))
        return True

    @property
    def url(self):
        return self.server.url

    # -- weights -----------------------------------------------------------

    #: Cost families collapsed to ONE control. A keep-out set is per sampled
    #: POINT -- `ko_<geom>_<point>`, 89 of them on a keepout cell -- and they
    #: are one physical intent ("do not put the gripper through the table"),
    #: never tuned individually. Left expanded they are 89 of the panel's 104
    #: sliders and 89 of its cost traces, and the fifteen terms someone
    #: actually reasons about become unfindable: this is why `reachRot` looked
    #: absent on an S18 cell when it was present and third from the bottom.
    GROUPS = (("ko_", "keepout"),)

    def _group_of(self, name):
        for pre, label in self.GROUPS:
            if name.startswith(pre):
                return label
        return None

    def _members(self, label):
        """Raw cost names behind a group label, in the built models."""
        return sorted(n for n in self._raw_weights()
                      if self._group_of(n) == label)

    def _raw_weights(self):
        if self.mpc is None:
            return {}
        out = {}
        for mdl in list(self.mpc.models) + [self.mpc.terminal]:
            cs = _cost_sum(mdl)
            if cs is None:
                continue
            for name in cs.costs.todict():
                out.setdefault(name, float(cs.costs[name].weight))
        return out

    def weights(self):
        """{name: weight} over the running models, from the FIRST model that
        has each term. The horizon's models share cost structure by
        construction (the MPC slides over the plan's own models), so one is
        representative; a term that exists only in some phase still appears.

        Grouped families collapse to one entry named for the family, carrying
        the weight its members share. If they ever DISAGREE the group is not
        offered and the members are shown individually -- a single slider over
        a set with two different weights would silently flatten them.
        """
        raw = self._raw_weights()
        out, grouped = {}, {}
        for name, w in raw.items():
            label = self._group_of(name)
            if label is None:
                out[name] = w
            else:
                grouped.setdefault(label, []).append(w)
        for label, ws in grouped.items():
            if len(set(ws)) == 1:
                out[label] = ws[0]
            else:
                out.update({n: raw[n] for n in raw
                            if self._group_of(n) == label})
        return out

    def _apply(self, name, value):        # guarded by `weights()` returning {}
        """Set one weight on EVERY model that carries it, plus the terminal.

        All of them, because the horizon slides: changing only the models
        currently in the problem means the weight silently reverts as the
        window advances past them.
        """
        if self.mpc is None:
            return 0
        # A group label is not a cost. Expand it to its members and set them
        # all, which is what "one intent, one knob" has to mean.
        members = ({name} if not any(name == lbl for _, lbl in self.GROUPS)
                   else set(self._members(name)))
        old, n_models, n_terms = None, 0, 0
        for mdl in list(self.mpc.models) + [self.mpc.terminal]:
            cs = _cost_sum(mdl)
            if cs is None:
                continue
            have = cs.costs.todict()
            hit = [nm for nm in members if nm in have]
            for nm in hit:
                if old is None:
                    old = float(cs.costs[nm].weight)
                cs.costs[nm].weight = float(value)
            n_terms += len(hit)
            n_models += bool(hit)
        if n_models:
            # `name` is what the operator moved -- the group label when they
            # moved a group. Recording a member's name instead would make the
            # artifact describe an edit nobody made.
            self.changes.append(dict(t=time.monotonic() - self._t0, name=name,
                                     old=old, new=float(value),
                                     models=n_models, terms=n_terms))
            self.dirty = True
        return n_models

    def _on_message(self, msg):
        """Browser -> here. This runs on a SOCKET thread.

        Weight edits are enqueued and applied by `drain` on the control thread,
        because writing a weight into a model the solver is reading is a data
        race whose symptom is a bad step rather than a crash. Everything else
        is a SESSION command -- pause, reset, task, viewer -- and those are
        handed straight to the session, which owns its own locking and must be
        able to act on a pause while the control thread is blocked inside a
        solve.
        """
        cmd = msg.get("cmd")
        if cmd == "weight":
            with self._qlock:
                self._q.append((msg["name"], float(msg["value"])))
        elif cmd and self.on_command is not None:
            try:
                self.on_command(cmd, msg)
            except Exception:                                    # noqa: BLE001
                pass      # a panel is never a reason to take down a session

    def drain(self):
        """Apply pending edits. Control thread, between periods."""
        with self._qlock:
            pending, self._q = self._q, []
        for name, value in pending:
            self._apply(name, value)
        return len(pending)

    # -- telemetry ---------------------------------------------------------

    def _terms(self):
        """Per-term cost at the first running node of the current solve.

        This is the breakdown that is otherwise only visible by running
        `croco_speed.py terms` offline, and it is the plot that pays: a weight
        slider with no per-term readout is a knob with no dial.
        """
        try:
            data = self.mpc.problem.runningDatas[0]
            cs = getattr(getattr(data, "differential", None), "costs", None)
            if cs is None:
                return {}
            out = {}
            for k in cs.costs.todict():
                label = self._group_of(k)
                # SUM, not mean: the grouped trace has to be the contribution
                # of the family to the total cost, or the cost plot's series
                # no longer add up to the number above it.
                key = label or k
                out[key] = out.get(key, 0.0) + float(cs.costs[k].cost)
            return out
        except Exception:                                        # noqa: BLE001
            return {}

    def on_step(self, row, state, cmd):
        """`ControlLoop`'s hook. Control thread; must be cheap and must not raise."""
        try:
            self.drain()
            self.n += 1
            if self.n % self.every:
                return
            msg = dict(
                type="step", k=self.n, t=row.get("t"), phase=row.get("phase"),
                period_ms=self.period_ms,
                solve_ms=row.get("solve_ms"), age_ms=1e3 * (row.get("age") or 0.0),
                period_ms_actual=row.get("period_ms"),
                deadline_ms=row.get("deadline_ms"),
                latency_ms=row.get("latency_ms"),
                tau_sat=row.get("tau_sat"), q_clip=row.get("q_clip"),
                # Present only when --safety is on. The panel draws a red mark
                # per tripping period and calls out the FIRST one, because the
                # layer's e-stop latches: everything after it would have been
                # measured against a robot already going limp.
                estop=row.get("estop"), estop_why=row.get("estop_why"),
                safety_clip=row.get("safety_clip"),
                step_length=(self.mpc.step_lengths[-1]
                             if getattr(self.mpc, "step_lengths", None) else None),
                terms=self._terms(), weights=self.weights(),
                # PHYSICS, not cost. Put on the row by the session before the
                # hooks run (see croco_twin's on_step) because the numbers come
                # from the plant's mjData, which the panel has no handle on and
                # should not grow one -- a panel that reaches into the plant is
                # a panel that can crash a control period.
                telem=row.get("telem"),
                dirty=self.dirty)
            self.hist.append(msg)
            self.server.broadcast(msg)
        except Exception:                                        # noqa: BLE001
            pass          # a panel is never a reason for a control period to fail

    def summary(self):
        """What the run artifact has to carry so a retuned run stays honest."""
        return dict(gui_weight_changes=self.changes,
                    gui_weights_modified=self.dirty,
                    gui_final_weights=self.weights() if self.dirty else None)

    def close(self):
        self.server.close()
