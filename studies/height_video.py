#!/usr/bin/env python3
"""Render a height_dynamic episode against the slab it was actually flown on.

`simple_video.py` cannot do this: it resolves its model through
`simple_lean.load()`, which reads the compiled MJCF, so a 1.085 m episode would
be drawn over a 0.985 m table -- the trajectory would look like a robot bracing
on air, which is the exact claim under test and the exact way to fake it.  This
loads through `contact_select` with `TABLE_H` set, so the slab in the picture is
the slab in the physics.

usage: TABLE_H is taken from --face.
  height_video.py --face 1.085 runs/.../h1085_x1060_brace_r0.csv --out media/
"""
import argparse, csv, os, sys

_FACE = None
for _i, _a in enumerate(sys.argv):
    if _a == "--face" and _i + 1 < len(sys.argv):
        _FACE = float(sys.argv[_i + 1])
if _FACE is not None:
    os.environ["TABLE_H"] = "%.6f" % _FACE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import mujoco
import imageio.v2 as imageio
import contact_select as cs

W, H = 960, 540


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--face", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--azimuth", type=float, default=140.0)
    a = ap.parse_args()

    m, d = cs.load(ik_margin=0.0)
    # The MJCF's offscreen framebuffer is smaller than the frame we want and
    # MuJoCo will not grow it at runtime; raising it on the loaded model is the
    # supported way (the alternative is editing a shared task XML for a video).
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
    face = float(cs.table_top_z(m, d))
    assert abs(face - a.face) < 1e-6, "slab is at %.4f, wanted %.4f" % (face, a.face)
    os.makedirs(a.out, exist_ok=True)
    ren = mujoco.Renderer(m, H, W)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation, cam.distance = a.azimuth, -12.0, 3.4
    cam.lookat[:] = [0.55, -0.10, 0.95]

    for path in a.csv:
        rows = [r for r in csv.reader(open(path)) if r and not r[0].startswith("#")]
        head, rows = rows[0], rows[1:]
        qi = [i for i, c in enumerate(head) if c.startswith("qpos")]
        ti = head.index("time")
        step = max(1, int(round(a.speed / (a.fps * (float(rows[1][ti]) - float(rows[0][ti]))))))
        out = os.path.join(a.out, os.path.basename(path)[:-4] + ".mp4")
        with imageio.get_writer(out, fps=a.fps, macro_block_size=1) as w:
            for r in rows[::step]:
                d.qpos[:len(qi)] = [float(r[i]) for i in qi]
                mujoco.mj_forward(m, d)
                ren.update_scene(d, cam)
                w.append_data(ren.render())
        print("wrote %s  (face %.3f, %d frames)" % (out, face, len(rows[::step])))


if __name__ == "__main__":
    main()
