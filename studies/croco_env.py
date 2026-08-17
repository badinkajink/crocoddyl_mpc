#!/usr/bin/env python3
"""Does THIS interpreter actually work, and is it pointed at the right assets?

WHY THIS EXISTS.  On 2026-08-16 the whole grid failed with SIGSEGV in every
contact mode, on both the old and the new model, with no Python traceback -- and
the cause was none of those things.  It was the interpreter: `base` has a
crocoddyl/pinocchio/coal wheel set whose CONTACT dynamics segfault, while the
`croco` env's works.  Nothing in the study said which interpreter it needed, so
the failure looked like an asset regression for as long as it took to bisect it.

The check is three lines of crocoddyl and takes ~1 s, and it fails with the
answer instead of a core dump:

    contact-free quasiStatic   works in both -- so "crocoddyl imports fine" and
                               even "crocoddyl runs" prove nothing
    contact    quasiStatic     segfaults in the bad one

Because the bad case is a SEGFAULT and not an exception, it cannot be caught
in-process: `check()` runs it in a SUBPROCESS and reads the exit code. A crash
therefore reports a diagnosis rather than taking the caller down with it.

usage:  python3 croco_env.py            # report, exit 1 if unusable
        from croco_env import require   # call at the top of a long run
"""
import os
import subprocess
import sys

# The probe. Deliberately tiny and study-independent: it must fail for exactly
# the reason the grid failed, and not for any reason of our own making.
PROBE = r"""
import numpy as np, pinocchio as pin, crocoddyl as cro
rm = pin.buildSampleModelHumanoid()
state = cro.StateMultibody(rm)
act = cro.ActuationModelFloatingBase(state)
nu = act.nu
x0 = np.concatenate([pin.neutral(rm), np.zeros(rm.nv)])
cts = cro.ContactModelMultiple(state, nu)
fid = rm.getFrameId(rm.frames[-1].name)
cts.addContact("c", cro.ContactModel3D(state, fid, np.zeros(3),
                                       pin.LOCAL_WORLD_ALIGNED, nu, np.zeros(2)))
dam = cro.DifferentialActionModelContactFwdDynamics(
    state, act, cts, cro.CostModelSum(state, nu), 0., True)
iam = cro.IntegratedActionModelEuler(dam, 0.02)
p = cro.ShootingProblem(x0, [iam] * 3, iam)
p.quasiStatic([x0] * 3)
print("OK")
"""

HINT = (
    "This interpreter cannot run crocoddyl CONTACT dynamics -- it segfaults.\n"
    "  interpreter: %s\n"
    "Use the environment the study is built for:\n"
    "  conda run -n croco python ...      (or: conda activate croco)\n"
    "Do NOT 'fix' this by changing the study: a contact-free crocoddyl problem\n"
    "solves fine in the broken environment, so partial success proves nothing."
)


def check(python=None):
    """(ok, detail). Runs the probe in a subprocess so a SIGSEGV is survivable."""
    exe = python or sys.executable
    try:
        p = subprocess.run([exe, "-c", PROBE], capture_output=True, text=True,
                           timeout=180)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, "probe did not run: %s" % exc
    if p.returncode == 0 and "OK" in p.stdout:
        return True, "contact dynamics OK"
    if p.returncode < 0:
        return False, "contact dynamics crashed with signal %d" % -p.returncode
    return False, "probe failed (rc=%d): %s" % (
        p.returncode, (p.stderr or "").strip().splitlines()[-1:] or "")


def assets():
    """The two paths every croco_* script resolves against, and whether they exist."""
    import croco_bridge as cb          # sets RTLD_GLOBAL; safe, no contacts
    import contact_select as cs
    return [("LEAN_TASK_DIR (MJCF)", cs.MODEL, os.path.exists(cs.MODEL)),
            ("CL_ASSETS_DIR (URDF)", cb.URDF, os.path.exists(cb.URDF))]


EXT_PROBE = r"""
import os, json
os.environ.setdefault("CROCO_KEEPOUT", "fused")
import croco_geom as cg
import croco_plan as cp
out = {}
out["keepout_mode"] = cg._MODE
out["keepout_cpp"] = cg._CPP is not None
out["passive_cpp"] = cp._cpp_actuation is not None and __import__(
    "importlib").util.find_spec("croco_passive") is not None
print(json.dumps(out))
"""


def extensions(python=None):
    """Which keep-out / actuation implementation will actually be used.

    THIS IS A 6x KNOB AND IT FAILS SILENTLY. croco_geom falls back to the Python
    activation with a one-line note if `croco_ext/*.so` is missing, so a fresh
    checkout runs correctly and six times too slowly. Measured on this machine,
    same plan, identical trajectory (pelvis z 0.956 m in every case):

        keepout=fused  passive=cpp      13.6 ms mean / 19.7 ms p95
        keepout=cpp    passive=cpp      28.7 ms        / 37.5 ms
        keepout=python passive=cpp      83.3 ms        / 136.1 ms
        keepout=python passive=python   85.2 ms        / 141.3 ms

    against a 20 ms control period. The unbuilt state is not "a bit slower", it
    is 4x outside the loop -- and nothing in a replay's output says so except
    the solve time, which is easy to read as "this problem is expensive".

    Fix: croco_ext/build.sh keepout passive   (or run_session.sh deps)
    """
    exe = python or sys.executable
    try:
        p = subprocess.run([exe, "-c", EXT_PROBE], capture_output=True,
                           text=True, timeout=180,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, "probe did not run: %s" % exc
    if p.returncode != 0:
        return None, (p.stderr or "").strip().splitlines()[-1:] or "probe failed"
    import json
    try:
        return json.loads(p.stdout.strip().splitlines()[-1]), "ok"
    except Exception:                                           # noqa: BLE001
        return None, "unparseable probe output"


def require():
    """Fail fast, with the fix, before a long run starts."""
    ok, detail = check()
    if not ok:
        raise SystemExit("croco_env: %s\n%s" % (detail, HINT % sys.executable))
    ext, _ = extensions()
    if ext and not ext.get("keepout_cpp"):
        raise SystemExit(
            "croco_env: the native keep-out extension is NOT built, so the MPC "
            "will run ~6x slower than the control period allows (85 ms against "
            "a 20 ms budget).\n"
            "  build it:  croco_ext/build.sh keepout passive\n"
            "If you genuinely want the slow path, set CROCO_KEEPOUT=python "
            "explicitly so it is a choice and not an accident.")
    bad = [(k, p) for k, p, e in assets() if not e]
    if bad:
        raise SystemExit("croco_env: missing assets:\n" + "\n".join(
            "  %s: %s" % (k, p) for k, p in bad)
            + "\nRun studies/stage_assets.sh and export the paths it prints.")


def main():
    ok, detail = check()
    print("interpreter : %s" % sys.executable)
    print("contacts    : %s" % detail)
    ext, edetail = extensions()
    if ext is None:
        print("extensions  : could not probe (%s)" % edetail)
    else:
        print("extensions  : keepout=%s (native %s)  passive native %s"
              % (ext["keepout_mode"],
                 "YES" if ext["keepout_cpp"] else "NO -- ~6x SLOWER",
                 "YES" if ext["passive_cpp"] else "no"))
    try:
        for k, p, e in assets():
            print("%-12s: %s %s" % (k.split()[0], "OK " if e else "MISSING", p))
    except SystemExit:
        raise
    except Exception as exc:                      # asset probe needs a working env
        print("assets      : could not probe (%s)" % exc)
    if not ok:
        print("\n" + HINT % sys.executable)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
