#!/usr/bin/env python3
"""Filmstrip PNGs from a replay video, for looking at rather than playing.

Metrics say a controller "stayed upright with 6.5 mm of reach error".  They do
not say whether the arm swings through the table on the way in, whether the
brace lands flat or on an edge, or whether the robot ends up in a posture a human
would call a braced lean.  Six evenly spaced stills answer that in one glance,
survive being pasted anywhere, and are how this session actually checked its own
results before writing them up.

usage: croco_strip.py runs/.../replay_s13_mpc.mp4 [more.mp4 ...]
"""

import os
import sys

import imageio.v2 as imageio
import numpy as np


def strip(path, n=6, scale=2):
    frames = imageio.mimread(path, memtest=False)
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    tiles = [np.asarray(frames[i])[::scale, ::scale, :3] for i in idx]
    h = min(t.shape[0] for t in tiles)
    out = np.concatenate([t[:h] for t in tiles], axis=1)
    dst = os.path.splitext(path)[0] + "_strip.png"
    imageio.imwrite(dst, out)
    return dst, len(frames)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        dst, n = strip(p)
        print(f"{dst}  ({n} frames)")
