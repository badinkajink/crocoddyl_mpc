"""Which interpreter this study runs under, and why it is not a free choice.

THE FAILURE THIS FILE EXISTS TO PREVENT. Typing the obvious thing --

    studies/croco_twin.py --dir <cell> --tag elbow_palm --plant mujoco --gui
    [croco_twin] panel on http://127.0.0.1:8770/ ...
    Segmentation fault (core dumped)

-- from a conda `(base)` shell is a SIGSEGV with no diagnostic, no traceback
and no hint that the interpreter is the problem. It is reproducible: measured
here 3/3 on `~/miniconda3/bin/python3`, twice as SIGSEGV and once as a
`MemoryError`, always inside `crocoddyl.ShootingProblem(...)` at
croco_plan.py's `build`. Both interpreters on that machine report the same
versions -- crocoddyl 3.2.1, pinocchio 4.1.0 -- so a version check would have
cleared it. The difference is the BUILD: base has the pip/cmeel wheels
(`.../site-packages/cmeel.prefix/...`) and the working env has conda-forge's,
and a crocoddyl linked against one pinocchio's C++ ABI faults when handed
objects built by another's.

WHY RE-EXEC RATHER THAN REFUSE. `run_session.sh` has always resolved the
interpreter (`_croco_py`) and selected the OpenMP libcrocoddyl per process, so
the shell entry points were already immune -- it was only the direct
`studies/*.py` invocation, the one every README example shows, that inherited
whatever shell you were in. Refusing would be honest but would leave the
documented command broken, so the entry points hand themselves to the pinned
interpreter and say so in one line. `CROCO_NO_REEXEC=1` turns it off,
`CROCO_PY=<path>` chooses the target.

THE OPENMP LIBRARY IS A SEPARATE, OPTIONAL PIN. conda-forge's crocoddyl is
built without OpenMP, which does not fault -- it silently pins `nthreads` to 1,
so `--threads 4` quietly does nothing. The rebuilt `libcrocoddyl.so.3.2.1` is
selected with LD_PRELOAD because the bindings carry `RPATH $ORIGIN/../../..`
and RPATH beats LD_LIBRARY_PATH. It is set only on the process being exec'd
and never exported into the shell: an exported LD_PRELOAD is inherited by
`git`, `ssh` and everything else you type next, which breaks them.
"""
from __future__ import annotations

import importlib.util
import os
import sys

#: Set in the child so a re-exec can never loop.
_GUARD = "_CROCO_REEXEC"

#: Once per process. `ensure_runtime` is called from the two modules that own
#: native setup (croco_bridge, contact_select), so a script importing both
#: would otherwise print the diagnosis twice.
_CALLED = False

#: Where an OpenMP-enabled libcrocoddyl is looked for, after $CROCO_CROCODDYL_LIB.
OMP_LIB_CANDIDATES = (
    "~/opt/crocoddyl-omp/lib/libcrocoddyl.so.3.2.1",
)

#: Conda roots searched for an `envs/croco`, after $CROCO_PY. Same list, same
#: order as run_session.sh's `_croco_py`, so the shell and the Python entry
#: points cannot disagree about what the pinned interpreter is.
CONDA_ROOTS = ("$CONDA_EXE", "~/miniconda3", "~/anaconda3", "~/miniforge3",
               "/opt/conda")


def _expand(p):
    return os.path.expanduser(os.path.expandvars(p))


def find_python():
    """The interpreter this study is pinned to, or None if there is no such env."""
    explicit = os.environ.get("CROCO_PY")
    if explicit:
        return explicit if os.access(explicit, os.X_OK) else None
    roots = []
    conda_exe = os.environ.get("CONDA_EXE", "")
    if conda_exe.endswith("/bin/conda"):
        roots.append(conda_exe[: -len("/bin/conda")])
    roots += [_expand(r) for r in CONDA_ROOTS if not r.startswith("$")]
    for root in roots:
        cand = os.path.join(root, "envs", "croco", "bin", "python")
        if os.access(cand, os.X_OK):
            return cand
    return None


def find_omp_crocoddyl():
    """The OpenMP libcrocoddyl to LD_PRELOAD, or None. Optional by design."""
    explicit = os.environ.get("CROCO_CROCODDYL_LIB")
    if explicit:
        return explicit if os.path.exists(explicit) else None
    for cand in OMP_LIB_CANDIDATES:
        cand = _expand(cand)
        if os.path.exists(cand):
            return cand
    return None


def crocoddyl_origin():
    """Where crocoddyl WOULD be imported from, without importing it.

    `find_spec` on a top-level package reads the finder's answer and stops; it
    does not execute the module. That matters because the whole point is to
    decide whether to hand this process to another interpreter BEFORE paying
    the ~1 s import that the exec would throw away.
    """
    try:
        spec = importlib.util.find_spec("crocoddyl")
    except Exception:                                            # noqa: BLE001
        return None
    return None if spec is None else (spec.origin or None)


def diagnose(stream=sys.stderr):
    """Say something useful when we are stuck on an interpreter we distrust.

    Only the cmeel case is named, because it is the only one that has been
    MEASURED to fault. Anything else gets the neutral message: an interpreter
    that merely is not the pinned one may well be fine, and crying wolf about
    it would train people to ignore this.
    """
    origin = crocoddyl_origin()
    if origin is None:
        print("[croco] crocoddyl is not importable under %s -- this will fail "
              "at the first solve. Install it, or point CROCO_PY at an "
              "interpreter that has it." % sys.executable, file=stream)
    elif "cmeel.prefix" in origin:
        print("[croco] WARNING: crocoddyl here is the pip/cmeel wheel\n"
              "          %s\n"
              "        which has been measured to SIGSEGV inside "
              "ShootingProblem on this stack (the wheel's pinocchio ABI is "
              "not the one crocoddyl was linked against). Versions match, so "
              "this is invisible to a version check. Use the conda-forge "
              "build: CROCO_PY=<env>/bin/python, or `conda activate croco`."
              % origin, file=stream)


def ensure_runtime(need_omp=False, stream=sys.stderr):
    """Re-exec into the pinned interpreter if this is not already it.

    Call FIRST, before importing crocoddyl/pinocchio/mujoco -- the point is to
    replace the process before it has paid for anything. Returns normally
    (having possibly warned) when there is nothing to switch to; never returns
    when it execs.
    """
    global _CALLED
    if _CALLED:
        return
    _CALLED = True
    if os.environ.get(_GUARD) or os.environ.get("CROCO_NO_REEXEC"):
        diagnose(stream)
        return
    target = find_python()
    if target is None:
        diagnose(stream)
        return
    try:
        same = os.path.samefile(target, sys.executable)
    except OSError:
        same = False
    lib = find_omp_crocoddyl()
    want_preload = bool(lib) and lib not in os.environ.get("LD_PRELOAD", "")
    if same and not (need_omp and want_preload):
        return
    env = dict(os.environ, **{_GUARD: "1"})
    why = []
    if not same:
        why.append("pinned interpreter")
    if want_preload:
        pre = os.environ.get("LD_PRELOAD", "")
        env["LD_PRELOAD"] = lib + (":" + pre if pre else "")
        why.append("OpenMP libcrocoddyl")
    print("[croco] re-exec into %s (%s). CROCO_NO_REEXEC=1 to disable, "
          "CROCO_PY=<python> to choose." % (target, ", ".join(why)),
          file=stream)
    try:
        os.execve(target, [target, "-u"] + sys.argv, env)
    except OSError as exc:                                       # noqa: BLE001
        print("[croco] re-exec failed (%s); continuing under %s"
              % (exc, sys.executable), file=stream)
        diagnose(stream)
