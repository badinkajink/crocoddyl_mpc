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
import argparse, csv, hashlib, json, os, sys

_FACE = None
for _i, _a in enumerate(sys.argv):
    if _a == "--face" and _i + 1 < len(sys.argv):
        _FACE = float(sys.argv[_i + 1])
if _FACE is not None:
    os.environ["TABLE_H"] = "%.6f" % _FACE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

W, H = 960, 540


def frame_samples(times, fps, speed):
    """Nearest logged state at each output timestamp; no accumulated stride error."""
    times = np.asarray(times, dtype=float)
    if fps <= 0 or not np.isfinite(speed) or speed <= 0:
        raise ValueError("fps and speed must be positive")
    if len(times) < 2 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("At least two strictly increasing finite timestamps required")
    count = int(np.floor((times[-1] - times[0]) * fps / speed + 1e-9)) + 1
    requested = times[0] + np.arange(count) * speed / fps
    right = np.clip(np.searchsorted(times, requested), 1, len(times) - 1)
    left = right - 1
    indices = np.where(requested - times[left] <= times[right] - requested, left, right)
    return requested, indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--face", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    # The September height clips compressed 25 s to 8.37 s without a label.
    # Default to real time and label every frame. Integer strides also give
    # 0.9x at requested 1x for 100 Hz logs / 30 fps; sample timestamps instead.
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Simulation seconds per video second (default: 1)")
    ap.add_argument("--prefix", default="", help="Output filename prefix")
    ap.add_argument("--azimuth", type=float, default=140.0)
    a = ap.parse_args()
    if a.fps <= 0 or not np.isfinite(a.speed) or a.speed <= 0:
        ap.error("fps and speed must be positive")
    os.environ["TABLE_H"] = "%.6f" % a.face
    import mujoco
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    import contact_select as cs

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

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    for path in a.csv:
        with open(path) as f:
            rows = [r for r in csv.reader(f) if r and not r[0].startswith("#")]
        head, rows = rows[0], rows[1:]
        qi = [i for i, c in enumerate(head) if c.startswith("qpos")]
        ti = head.index("time")
        times = np.asarray([float(r[ti]) for r in rows])
        requested, indices = frame_samples(times, a.fps, a.speed)
        out = os.path.join(a.out, a.prefix + os.path.basename(path)[:-4] + ".mp4")
        with imageio.get_writer(out, fps=a.fps, macro_block_size=1,
                               ffmpeg_params=["-threads", "1"]) as w:
            for k in indices:
                r = rows[k]
                d.qpos[:len(qi)] = [float(r[i]) for i in qi]
                mujoco.mj_forward(m, d)
                ren.update_scene(d, cam)
                frame = Image.fromarray(ren.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, W, 38), fill=(15, 20, 27))
                label = "Table %.3f m   |   Simulation t = %.2f s   |   Playback %gx" % (
                    face, times[k], a.speed)
                draw.text((12, 6), label, fill=(240, 240, 240), font=font)
                w.append_data(np.asarray(frame))
        metadata = dict(source_csv=os.path.abspath(path),
                        source_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(),
                        model=os.path.abspath(cs.MODEL), table_face_m=face,
                        fps=a.fps, playback_speed=a.speed, frames=len(indices),
                        source_start_s=float(times[0]), source_end_s=float(times[-1]),
                        video_duration_s=len(indices) / a.fps,
                        max_sample_time_error_s=float(np.max(np.abs(times[indices] - requested))),
                        sampling="Nearest logged state at each output timestamp",
                        video_sha256=hashlib.sha256(open(out, "rb").read()).hexdigest())
        with open(out[:-4] + ".json", "w") as f:
            json.dump(metadata, f, indent=2)
            f.write("\n")
        print("wrote %s  (face %.3f, %d frames, %gx)" % (out, face, len(indices), a.speed), flush=True)
    ren.close()


if __name__ == "__main__":
    main()
