"""The crocoddyl braced-lean controller, as a library rather than a script pile.

WHAT LIVES HERE vs in `studies/` (was `studies/`): this package is the
RUNTIME -- the things that have to run on a robot. Everything that measures,
sweeps, scores or renders stays a study. The split is the deploy boundary, so
"can this ship" is answerable by looking at the import graph.

    croco.control   the command tuple, and the receding-horizon solver
    croco.plant     what is being driven: MuJoCo in-process, or DDS
    croco.model     the Pinocchio<->MuJoCo bridge and the table geometry

The deploy story in one line: the controller emits (q_des, kp, kd, tau_ff) and
the plant decides what that means, so twin and robot differ by one constructor.
"""
