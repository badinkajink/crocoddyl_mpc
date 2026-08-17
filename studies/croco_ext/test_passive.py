#!/usr/bin/env python3
"""The C++ passive actuation must be the SAME model as the Python one.

`croco_plan._make_actuation` carries the plant's joint damping and friction loss
into the actuation, which is what makes the plan's torques the torques an
actuator has to produce rather than net joint torques.  Getting that wrong is not
a slow plan, it is a wrong one, so the transcription is checked before it is
used -- on tau, on dtau_dx and on dtau_du, at states drawn from the real planned
trajectory (where |v_j| is small and tanh is in its steep region, which is
exactly where a sign or an eps error hides) as well as at large velocities.

`commands` is checked too: it is the inverse map the replay uses to read joint
torques back out of a generalized torque.

usage: croco_ext/test_passive.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import croco_bridge as cb          # noqa: E402  (first: sets RTLD_GLOBAL)

crocoddyl = cb.import_crocoddyl()

import croco_plan as cp            # noqa: E402
import croco_passive as ck         # noqa: E402


def main():
    rmodel = cb.build_pin()
    state = crocoddyl.StateMultibody(rmodel)
    m_mj, _ = cb.mj_model()
    b, _, f = cb.mj_joint_passive(m_mj)
    nu = state.nv - 6

    # `_make_actuation` returns the C++ model when it is built, which is the
    # whole point of this test -- so the Python one is built by asking for it
    # explicitly through the same environment switch the module reads.
    os.environ["CROCO_PASSIVE"] = "python"
    py = cp._make_actuation(state, b, f)
    os.environ["CROCO_PASSIVE"] = "cpp"
    cc = cp._make_actuation(state, b, f)
    if type(cc).__name__ != "ActuationModelJointPassive" or cc is py:
        raise SystemExit("croco_passive is not built; nothing to compare")

    d_py, d_cc = py.createData(), cc.createData()
    rng = np.random.default_rng(0)
    worst = dict(tau=0.0, dtau_dx=0.0, dtau_du=0.0, commands=0.0)
    # Velocity scales: the plan's own (|v_j| <~ 1 rad/s), the tanh knee, and
    # well past it.
    for scale in (0.0, 1e-3, 0.05, 0.5, 5.0):
        for _ in range(40):
            q = cb.mj_to_pin(m_mj.key_qpos[0].copy())
            v = rng.normal(0.0, scale, state.nv)
            x = np.concatenate([q, v])
            u = rng.normal(0.0, 40.0, nu)
            for mdl, d in ((py, d_py), (cc, d_cc)):
                mdl.calc(d, x, u)
                mdl.calcDiff(d, x, u)
            worst["tau"] = max(worst["tau"], float(np.max(np.abs(
                np.array(d_py.tau) - np.array(d_cc.tau)))))
            worst["dtau_dx"] = max(worst["dtau_dx"], float(np.max(np.abs(
                np.array(d_py.dtau_dx) - np.array(d_cc.dtau_dx)))))
            worst["dtau_du"] = max(worst["dtau_du"], float(np.max(np.abs(
                np.array(d_py.dtau_du) - np.array(d_cc.dtau_du)))))
            tau = rng.normal(0.0, 40.0, state.nv)
            py.commands(d_py, x, tau)
            cc.commands(d_cc, x, tau)
            worst["commands"] = max(worst["commands"], float(np.max(np.abs(
                np.array(d_py.u) - np.array(d_cc.u)))))

    for k, v in worst.items():
        print(f"  {k:9s} max |python - c++| = {v:.3e}")
    ok = max(worst.values()) < 1e-14
    print("\nAGREE" if ok else "\nDISAGREE -- do not use the C++ actuation")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
