#!/usr/bin/env python3
"""Render a Lean Simple rollout as an annotated video (and/or a contact sheet).

A rollout that shows, in the corner, which links are ACTUALLY in contact and how
far each candidate link is from the slab is hard to misread. The S11 page wrote
up a run that spent its whole duration in keyframe 0 as a brace attempt, and the
S12 page's headline "holds a genuine brace for 12 s" seed turns out to have an
arm on the table 3-15% of the time. Both were watchable mistakes.

Contact dots are drawn where MuJoCo's narrowphase reports the contact, not at a
nominal brace site: the two differ by centimetres and drawing the site makes a
hovering forearm look seated.

usage:
  simple_video.py CSV [CSV ...] [--out DIR] [--fps 30] [--speed 4] [--sheet]
"""
import argparse
import os
import subprocess

import numpy as np
import mujoco
from PIL import Image, ImageDraw

import simple_lean as sl

FFMPEG = os.path.join(sl.ROOT, "build/tools/imageio_ffmpeg/binaries/"
                               "ffmpeg-linux-x86_64-v7.0.2")
W, H = 900, 700
DOT = {"elbow": (40, 120, 215), "forearm": (235, 105, 50),
       "palm": (28, 176, 122), "trunk": (232, 125, 166)}


def make_renderer(m):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    return mujoco.Renderer(m, H, W, max_geom=m.ngeom + 64)


def cam(az, el=-12, dist=2.9, look=(0.80, 0.0, 0.92)):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = look
    c.distance, c.azimuth, c.elevation = dist, az, el
    return c


def add_marker(scn, pos, rgba, size=0.022):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([size, 0, 0]), np.asarray(pos, dtype=float),
                        np.eye(3).flatten(), np.asarray(rgba, dtype=np.float32))
    scn.ngeom += 1


def annotate(img, text_lines, dots):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    dr.rectangle([0, 0, 330, 22 + 17 * len(text_lines)], fill=(0, 0, 0, 140))
    for i, t in enumerate(text_lines):
        dr.text((10, 8 + 17 * i), t, fill=(255, 255, 255, 255))
    # contact legend: filled = touching, hollow = not
    x = 12
    for name, on in dots:
        c = DOT[name]
        dr.ellipse([x, H - 26, x + 14, H - 12],
                   fill=c + (255,) if on else None, outline=c + (255,), width=2)
        dr.text((x + 19, H - 26), name, fill=(255, 255, 255, 255))
        x += 26 + 8 * len(name)
    return np.asarray(im)


def render(path, out_dir, fps, speed, sheet, cams):
    col, rows, meta = sl.load_traj(path)
    m, d = sl.load()
    r = make_renderer(m)
    qi = [col["qpos%d" % i] for i in range(m.nq)]
    vi = [col["qvel%d" % i] for i in range(m.nv)]
    ui = [col["ctrl%d" % i] for i in range(m.nu)]
    dt = rows[1, col["time"]] - rows[0, col["time"]]
    stride = max(1, int(round(speed / (fps * dt))))

    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    geoms = {k: sl.body_geoms(m, b) for k, b in sl.LINKS.items()}
    geoms["trunk"] = sl.body_geoms(m, sl.TRUNK)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    n = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    target = sl.target_from_meta(
        meta, m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 3])

    base = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(out_dir, exist_ok=True)
    mp4 = os.path.join(out_dir, base + ".mp4")
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", "%dx%d" % (W * len(cams), H), "-r", str(fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "26",
         "-movflags", "+faststart", mp4],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)

    sheet_frames, sheet_at = [], set()
    if sheet:
        sheet_at = set(np.linspace(0, len(rows) - 1, 6).astype(int) //
                       stride * stride)

    for k in range(0, len(rows), stride):
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = rows[k, vi]
        d.ctrl[:] = rows[k, ui]
        mujoco.mj_forward(m, d)
        s = sl.slab(m, d)

        gaps = {name: sl.seat_gap(m, d, s, sl.SEAT[name]) for name in sl.LINKS}
        touch = {name: sl.touching(m, d, gs, table) for name, gs in geoms.items()}
        err = float(np.linalg.norm(d.site_xpos[hand] - target))

        tiles = []
        for az in cams:
            r.update_scene(d, camera=cam(az))
            scn = r.scene
            add_marker(scn, target, (0.15, 0.85, 0.30, 0.85), 0.035)
            for i in range(d.ncon):
                c = d.contact[i]
                if c.dist > 0 or table not in (c.geom1, c.geom2):
                    continue
                for name, gs in geoms.items():
                    if c.geom1 in gs or c.geom2 in gs:
                        add_marker(scn, c.pos,
                                   tuple(v / 255 for v in DOT[name]) + (0.95,))
            tiles.append(r.render())
        frame = np.concatenate(tiles, axis=1) if len(tiles) > 1 else tiles[0]

        lines = ["t = %5.2f s   reach err %5.3f m" % (rows[k, col["time"]], err),
                 "gap  elbow %+6.1f  forearm %+6.1f  palm %+6.1f mm"
                 % (gaps["elbow"] * 1e3, gaps["forearm"] * 1e3,
                    gaps["palm"] * 1e3)]
        frame = annotate(frame, lines,
                         [(nm, touch[nm]) for nm in ("elbow", "forearm",
                                                     "palm", "trunk")])
        proc.stdin.write(frame.astype(np.uint8).tobytes())
        if k in sheet_at:
            sheet_frames.append(frame)

    proc.stdin.close()
    proc.wait()
    print("wrote", mp4)

    if sheet and sheet_frames:
        cols = 3
        rowsn = int(np.ceil(len(sheet_frames) / cols))
        fh, fw = sheet_frames[0].shape[:2]
        canvas = np.full((rowsn * fh, cols * fw, 3), 255, np.uint8)
        for i, f in enumerate(sheet_frames):
            rr, cc = divmod(i, cols)
            canvas[rr * fh:(rr + 1) * fh, cc * fw:(cc + 1) * fw] = f
        p = os.path.join(out_dir, base + "_sheet.png")
        Image.fromarray(canvas).resize((cols * fw // 2, rowsn * fh // 2)).save(p)
        print("wrote", p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default=os.path.join(sl.ROOT, "docs/lean/media"))
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--cams", default="120,180")
    a = ap.parse_args()
    cams = [float(x) for x in a.cams.split(",")]
    for p in a.csv:
        render(p, a.out, a.fps, a.speed, a.sheet, cams)


if __name__ == "__main__":
    main()
