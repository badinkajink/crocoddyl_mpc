"""Receding-horizon crocoddyl, lifted out of the replay script.

WHY IT MOVED. This class is the CONTROLLER -- the thing that would run on the
robot -- and it lived at line 434 of a 1268-line file whose other 1100 lines are
replay, scoring, ghost markers and video. Nothing about it was reusable while it
was in there, and "can this ship" was unanswerable by looking at the imports.
It is unchanged apart from the crocoddyl import, which now goes through this
module rather than through the study's bridge.

The solver contract is the one the study measured: warm-started BoxFDDP over the
SAME action models the offline plan was built from, one `circularAppend` per
control period, a shrinking-horizon tail, and an optional truncated line-search
ladder. See the docstrings below -- every one of those choices was a measurement.

Deployment note: `__call__` returns (first control, first predicted state), which
is exactly what a `croco.control.Command` needs for tau_ff and q_des. The glue
lives in the runtime loop, not here, so this class stays a solver.
"""
import ctypes
import sys
import time

import numpy as np

_crocoddyl = None


def _import_crocoddyl():
    """Import crocoddyl with RTLD_GLOBAL, because a plain import SEGFAULTS.

    The crocoddyl 3.2.1 cmeel wheel ships the same symbols twice -- once in
    libcrocoddyl.so and again inlined into the pywrap -- and Python dlopens
    extension modules RTLD_LOCAL, so the two copies never reconcile. The
    contact-cone residuals identify their contact by dynamic_cast on the contact
    data; across two unreconciled copies that cast returns the wrong thing and
    the data constructor memmoves against a bogus size. It surfaces as a SIGSEGV
    inside ShootingProblem::allocateData with nothing in the traceback about
    contacts, or -- when the allocator notices first -- as a bare MemoryError,
    and which one you get varies between identical invocations.

    The flag has to cover pinocchio as well, so it is set before either loads and
    left set. Duplicated from the study's bridge on purpose: the runtime must not
    import the study to be correct.
    """
    global _crocoddyl
    if _crocoddyl is None:
        sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
        import crocoddyl
        _crocoddyl = crocoddyl
    return _crocoddyl


class MPC:
    """Receding-horizon crocoddyl about the plan's own models.

    The horizon slides over the SAME action models the plan was built from, so
    every cost the MPC sees -- landing spots, the reach target, the cones, the
    keep-out -- is the one the plan was solved against.  Nothing is re-authored
    for the online problem, which is the point: this measures what re-solving
    from the measured state buys, not what a different cost function buys.

    `iters` is small on purpose.  A warm-started DDP that takes one or two
    iterations per control step is the standard MPC construction (the previous
    solution is a very good guess when the state has moved 20 ms), and it is what
    makes the per-step solve time reportable rather than embarrassing.
    """

    COARSE = """A COARSER MPC GRID, and why it is decimation and not a
    non-uniform horizon.

    The obvious structural saving is a fine head at the control period and a
    coarse tail at a multiple of it: a 1 s preview in 30 nodes instead of 50.
    It is not implementable cheaply here, and the reason is the window
    management rather than the models.  crocoddyl gives exactly one cheap
    sliding operation, `circularAppend`, measured at 4.4 us -- and it ROTATES
    the whole list, which is only correct if every node is the same duration.
    A mixed fine/coarse window has to be re-pointed with `updateModel`, measured
    at 263 us per node because it re-creates that node's data (`createData`
    alone is 98 us).  Re-pointing a 15-node coarse tail every control period is
    3.9 ms against a ~10 ms step: the window management costs more than the
    fifteen nodes it removes.

    What IS cheap is a UNIFORM grid at n times the control period, because
    rotation stays valid -- the window simply advances one coarse node every n
    control periods instead of one fine node every period.  The controller
    still runs at the control period and still re-solves from the measured
    state every time; what changes is that its grid shifts on a coarser clock.
    So `dt_scale` decimates: model i of the coarse list is the plan's node i
    integrated for n*dt, and the window covers plan nodes anchored n apart.
    Same preview, n times fewer nodes, one `circularAppend` per n periods."""

    def __init__(self, ocp, models, terminal, horizon=40, iters=2,
                 xs_plan=None, us_plan=None, n_alphas=0, nthreads=0,
                 dt_scale=1):
        cro = _import_crocoddyl()
        self.n_alphas = n_alphas
        # 0 = leave crocoddyl's own default alone.  The knob only does anything
        # against a libcrocoddyl built with -DBUILD_WITH_MULTITHREADS=ON; the
        # stock conda-forge build prints a warning and pins it to 1, which is
        # why `croco_speed.py threads` reports the value it read BACK rather
        # than the value it asked for.
        self.nthreads = nthreads
        # DECIMATION.  Every index below is in COARSE units; `__call__` maps the
        # control step into them.  At dt_scale = 1 this is the identity and the
        # path is bit-for-bit the one S13-S16 measured.
        self.dt_scale = max(int(dt_scale), 1)
        n = self.dt_scale
        if n > 1:
            models = list(models[::n])
            xs_plan = None if xs_plan is None else xs_plan[::n]
            us_plan = None if us_plan is None else us_plan[::n]
        self.ocp, self.models, self.terminal = ocp, models, terminal
        self.H = min(horizon, len(models))
        self.iters = iters
        self.xs_plan, self.us_plan = xs_plan, us_plan
        self.solve_times = []
        # Line-search diagnostics.  FDDP's forward pass is a full nonlinear
        # rollout PER TRIAL STEP, and crocoddyl's default alpha ladder has 10
        # rungs, so a step that walks to the bottom costs ten rollouts -- as much
        # again as the backward pass it followed.  Whether that happens is not
        # inferable from the mean solve time, so it is recorded.
        self.step_lengths = []
        # ONE problem, rotated with circularAppend, not a fresh ShootingProblem
        # per control step.  Rebuilding costs an allocateData over the whole
        # horizon every 20 ms, and with the box keep-out that is 25 nodes x 86
        # Python activation datas -- measured at 336 ms per step, seventeen times
        # the control period, which makes the "is this real-time" question
        # unanswerable for reasons that have nothing to do with the solve.
        self.datas = [m.createData() for m in models]
        self.problem = cro.ShootingProblem(
            ocp.x0, list(models[:self.H]), terminal)
        self.solver = self._make_solver(cro, self.problem)
        self.head = self.H            # index of the next model to append
        self.xs = self.us = None

    def _make_solver(self, cro, problem):
        """A BoxFDDP with, optionally, a TRUNCATED line-search ladder.

        FDDP's forward pass is a full nonlinear rollout per trial step, and
        crocoddyl's default ladder is ten rungs down to alpha = 2^-9.  Measured
        here the median step is 0.125-0.1875, i.e. ~4 rollouts -- but the tail of
        the distribution is what decides whether a control period is met, and the
        p95 is 40% above the mean for exactly this reason.  Capping the ladder at
        `n_alphas` rungs BOUNDS the rollouts per iteration, which is the shape a
        real-time budget wants: a step that would have needed a tenth of a rung
        is not taken at all, the regularisation goes up, and the controller
        re-applies its shifted previous solution for that period.  Whether that
        costs anything is the `alphas` column of croco_speed.py sweep.
        """
        if self.nthreads:
            problem.nthreads = self.nthreads
        solver = cro.SolverBoxFDDP(problem)
        if self.n_alphas:
            solver.alphas = [2.0 ** -i for i in range(self.n_alphas)]
        return solver

    def __call__(self, k, x_meas):
        """Returns (first control, first predicted state) or (None, None)."""
        k = k // self.dt_scale
        if k >= len(self.models):
            return None, None
        if k + self.H <= len(self.models):
            # Window still fits: slide it by rotating the existing problem.
            while self.head < k + self.H:
                self.problem.circularAppend(self.models[self.head],
                                            self.datas[self.head])
                self.head += 1
        else:
            # TAIL: the window would run past the end, and circularAppend can
            # only rotate, never shrink.  Left alone the window simply stops
            # sliding, so the MPC keeps solving a stale set of models while the
            # robot advances past them -- and for H equal to the full horizon it
            # never slides at all, which is not "MPC with a long horizon" but
            # "restart the whole maneuver every 20 ms".  Measured: that
            # configuration stands still and misses the target by 856 mm.  So the
            # tail rebuilds a shrinking-horizon problem instead; it costs an
            # allocateData per step over the last H steps only.
            cro = _import_crocoddyl()
            self.problem = cro.ShootingProblem(
                x_meas, list(self.models[k:]), self.terminal)
            self.solver = self._make_solver(cro, self.problem)
            self.head = len(self.models)
        self.problem.x0 = x_meas
        H = self.problem.T
        if self.xs is None:
            # First call: warm-start from the OFFLINE PLAN, not from a constant.
            # The plan is the best guess that exists for this window, and one
            # DDP iteration from a constant guess is not an MPC, it is noise.
            xs = [x_meas] + [np.array(x) for x in self.xs_plan[k + 1:k + H + 1]]
            us = [np.array(u) for u in self.us_plan[k:k + H]]
        else:                                   # shift the previous solution
            xs = [x_meas] + list(self.xs[2:]) + [self.xs[-1]]
            us = list(self.us[1:]) + [self.us[-1]]
        while len(xs) < H + 1:
            xs.append(xs[-1])
        while len(us) < H:
            us.append(us[-1])
        xs, us = xs[:H + 1], us[:H]
        t0 = time.time()
        self.solver.solve(xs, us, self.iters, False, 1e-9)
        self.solve_times.append(time.time() - t0)
        self.step_lengths.append(float(self.solver.stepLength))
        self.xs, self.us = list(self.solver.xs), list(self.solver.us)
        return np.array(self.solver.us[0]), np.array(self.solver.xs[1])
