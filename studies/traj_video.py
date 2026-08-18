#!/usr/bin/env python3
"""Replay a dumped MJPC rollout as video, annotated with the phase it is in.

The S11 page had no clips of the MJPC side, which is how a run that spent all
12 s in keyframe 0 got written up as a brace attempt.  A rollout that shows its
phase name, its live table contacts and its pelvis height in the corner is hard
to misread that way.

Contact markers sit where MuJoCo's narrowphase reports the contact, not at the
nominal brace site -- the two differ by 3-13 cm because the site is on the link
axis and the contact is on the capsule surface, and drawing the site makes a
resting forearm look like it is hovering.

usage: traj_video.py CSV [CSV ...] [--out DIR] [--fps 30] [--speed 4]
"""
import argparse
import csv
import os

import numpy as np
import mujoco
import imageio.v2 as imageio

import contact_select as cs

W, H = 960, 720
PHASE_NAMES = ["stand_up", "brace_lean", "brace_reach", "brace_release",
               "standback_r1", "standback_r2", "standback_r3", "stand_up (end)"]
BRACE_BODIES = {"elbow": "%s_shoulder_yaw_link" % cs.BRACE_ARM,
                "forearm": "%s_elbow_link" % cs.BRACE_ARM,
                "palm": "%s_magpie_gripper" % cs.BRACE_ARM,
                "wrist": "%s_wrist_yaw_link" % cs.BRACE_ARM,
                "hip": "torso_link"}
MARK_RGBA = {"elbow": (0.16, 0.47, 0.84, 1), "forearm": (0.92, 0.41, 0.20, 1),
             "palm": (0.11, 0.69, 0.48, 1), "wrist": (0.93, 0.63, 0.00, 1),
             "hip": (0.91, 0.49, 0.65, 1)}


def load_traj(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    r = csv.reader(lines)
    hdr = next(r)
    rows = np.array([[float(v) for v in row] for row in r])
    return {n: i for i, n in enumerate(hdr)}, rows


def make_renderer(m):
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    return mujoco.Renderer(m, H, W, max_geom=m.ngeom + 64)


def cam(azim, elev=-14, dist=3.0, lookat=(0.75, 0.0, 0.90)):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, azim, elev
    return c


def render_traj(path, out_dir, fps, speed, target):
    col, rows = load_traj(path)
    m, d = cs.load(ik_margin=0)
    r = make_renderer(m)
    nq = m.nq
    qi = [col["qpos%d" % i] for i in range(nq)]
    dt = rows[1, col["time"]] - rows[0, col["time"]]
    stride = max(1, int(round(speed / (fps * dt))))

    frames, overlays = [], []
    for k in range(0, len(rows), stride):
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)

        r.update_scene(d, camera=cam(120))
        scn = r.scene
        if target is not None and scn.ngeom < scn.maxgeom:
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.045] * 3),
                                np.asarray(target, dtype=np.float64),
                                np.eye(3).flatten(),
                                np.array([1., 1., 1., .9], dtype=np.float32))
            scn.ngeom += 1

        tbl = cs.bid(m, "table")
        live = {}
        for c in range(d.ncon):
            con = d.contact[c]
            b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
            other = b2 if b1 == tbl else (b1 if b2 == tbl else None)
            if other is not None and other not in live:
                live[other] = con.pos.copy()
        names = []
        for s, body in BRACE_BODIES.items():
            b = cs.bid(m, body)
            if b not in live or scn.ngeom >= scn.maxgeom:
                continue
            names.append(s)
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.035] * 3), live[b],
                                np.eye(3).flatten(),
                                np.array(MARK_RGBA[s], dtype=np.float32))
            scn.ngeom += 1

        frames.append(r.render())
        ph = int(rows[k, col["phase"]])
        overlays.append((float(rows[k, col["time"]]),
                         PHASE_NAMES[ph] if 0 <= ph < len(PHASE_NAMES) else "auto",
                         "+".join(sorted(names)) or "none",
                         float(d.xpos[cs.bid(m, "pelvis")][2])))

    frames = burn_in(frames, overlays)
    tag = os.path.splitext(os.path.basename(path))[0]
    dst = os.path.join(out_dir, "mjpc_%s.mp4" % tag)
    imageio.mimsave(dst, frames, fps=fps, quality=8, macro_block_size=1)
    print("wrote %s  (%d frames, %.0fx speed)" % (dst, len(frames), speed))
    return dst


def burn_in(frames, overlays):
    """Draw the phase banner with PIL if available, else leave frames alone --
    a clip without a caption is still evidence; a crashed render is not."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  (PIL missing: no burned-in captions)")
        return frames
    out = []
    for img, (t, phase, contacts, pz) in zip(frames, overlays):
        im = Image.fromarray(img)
        dr = ImageDraw.Draw(im, "RGBA")
        dr.rectangle([0, 0, W, 62], fill=(12, 12, 12, 190))
        dr.text((16, 8), "t %5.1f s    phase: %s" % (t, phase),
                fill=(255, 255, 255, 255))
        dr.text((16, 34), "table contacts: %-28s pelvis z %.3f m"
                % (contacts, pz), fill=(205, 205, 195, 255))
        out.append(np.asarray(im))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out", default="/home/humanoid/Programs/mjpc_icra2026/docs/lean/media")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--speed", type=float, default=4.0,
                    help="playback speed multiple; 4 keeps a 66 s chain under 30 s")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    m0, _ = cs.load(ik_margin=0)
    nid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    target = [float(v) for v in
              m0.numeric_data[m0.numeric_adr[nid]:m0.numeric_adr[nid] + 3]]
    for path in a.csv:
        render_traj(path, a.out, a.fps, a.speed, target)


if __name__ == "__main__":
    main()
