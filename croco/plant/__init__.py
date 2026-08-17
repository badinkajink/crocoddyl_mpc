"""Plants: the swappable other half of the control loop.

    MuJoCoPlant   in-process physics, for replay and stress testing
    DDSPlant      rt/lowstate -> rt/lowcmd, i.e. the twin AND the real robot

Import DDSPlant lazily: it needs unitree_sdk2py, which a study machine may not
have, and a missing DDS stack must not stop anyone from running a replay.
"""
from .base import Plant, State                    # noqa: F401
from .mujoco_plant import MuJoCoPlant, default_sense   # noqa: F401

__all__ = ["Plant", "State", "MuJoCoPlant", "default_sense", "DDSPlant"]


def __getattr__(name):
    if name == "DDSPlant":
        from .dds_plant import DDSPlant
        return DDSPlant
    raise AttributeError(name)
