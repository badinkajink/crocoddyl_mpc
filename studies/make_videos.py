#!/usr/bin/env python3
"""Render proof-of-contact videos for the lean contact-selection study.

Three clips:
  A. approach_<set>.mp4  -- the IK marching from the stand_up seed into the braced
     pose, so the contacts are seen being MADE, not just held. Contact points that
     are actually touching the table are marked live.
  B. orbit_<set>.mp4     -- camera orbit around the settled pose.
  C. sweep_targets.mp4   -- march across the reach grid, showing the selected
     contact set change with the target.
"""
import os, sys, itertools
import numpy as np
import mujoco
import imageio.v2 as imageio

import contact_select as cs

W, H, FPS = 960, 720, 30
OUT = "/home/humanoid/Programs/mjpc_icra2026/docs/lean/media"
SITE_RGBA = {"elbow": (0.16, 0.47, 0.84, 1), "forearm": (0.92, 0.41, 0.20, 1),
             "palm": (0.11, 0.69, 0.48, 1), "hip": (0.91, 0.49, 0.65, 1),
             "torso": (0.93, 0.63, 0.00, 1)}


def make_renderer(m):
    """The model declares a small <global offwidth>; raise it before constructing
    the renderer, and leave headroom for the contact-marker geoms we add."""
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, W)
    m.vis.global_.offheight = max(m.vis.global_.offheight, H)
    return mujoco.Renderer(m, H, W, max_geom=m.ngeom + 64)


def cam(azim, elev=-14, dist=2.9, lookat=(0.85, 0.0, 0.95)):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:] = lookat
    c.distance, c.azimuth, c.elevation = dist, azim, elev
    return c


def draw(renderer, m, d, camera, subset, target=None, label=""):
    """Render one frame, overlaying a marker at each ACTIVE contact site."""
    renderer.update_scene(d, camera=camera)
    scn = renderer.scene
    tbl = cs.bid(m, "table")
    touching = set()
    for c in range(d.ncon):
        con = d.contact[c]
        if con.dist > cs.IK_MARGIN * 0.5:
            continue
        b1, b2 = m.geom_bodyid[con.geom[0]], m.geom_bodyid[con.geom[1]]
        if b1 == tbl: touching.add(b2)
        elif b2 == tbl: touching.add(b1)
    # the reach target itself -- without this the clip is ambiguous, because the
    # green sphere already in the scene is the MODEL's own object marker, not ours
    if target is not None and scn.ngeom < scn.maxgeom:
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.045] * 3), np.asarray(target, dtype=np.float64),
                            np.eye(3).flatten(),
                            np.array([1.0, 1.0, 1.0, 0.9], dtype=np.float32))
        scn.ngeom += 1
    # Markers sit at the contact MuJoCo actually reports, not at the nominal site.
    # The two differ by 3-13 cm (the site is on the link axis, the contact is on
    # the capsule surface), and drawing the site made the clip look like the arm
    # was hovering above the table when it was resting on it.
    resolved = {}
    for b, p, _ in cs.resolved_contacts(m, d, tuple(subset))[8:]:
        resolved.setdefault(b, p)
    for s in subset:
        body, off = cs.SITES[s]
        b = cs.bid(m, body)
        live = b in touching
        p = resolved.get(b, cs.point_world(m, d, body, off))
        if scn.ngeom >= scn.maxgeom:
            break
        g = scn.geoms[scn.ngeom]
        rgba = np.array(SITE_RGBA[s], dtype=np.float32)
        if not live:
            rgba[3] = 0.30                      # ghosted until contact is made
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([0.035 if live else 0.025] * 3),
                            np.asarray(p, dtype=np.float64), np.eye(3).flatten(), rgba)
        scn.ngeom += 1
    return renderer.render()


def approach(target, subset, tag):  # target drawn as a white sphere
    """Animate the IK converging: contacts are seen being established."""
    m, d = cs.load()
    r = make_renderer(m)
    frames = []
    # step the IK in small increments, capturing as we go
    for k in range(60):
        cs.solve_ik(m, d, np.array(target), subset, iters=4)
        az = 118 + 22 * np.sin(k / 60 * np.pi)
        frames.append(draw(r, m, d, cam(az), subset, target))
    for _ in range(25):                          # hold the settled brace
        frames.append(frames[-1])
    path = f"{OUT}/approach_{tag}.mp4"
    imageio.mimsave(path, frames, fps=FPS, quality=8, macro_block_size=1)
    print("wrote", path, len(frames), "frames")
    return m, d


def orbit(m, d, subset, tag, target=None):
    r = make_renderer(m)
    frames = [draw(r, m, d, cam(a), subset, target) for a in np.linspace(0, 360, 120)]
    path = f"{OUT}/orbit_{tag}.mp4"
    imageio.mimsave(path, frames, fps=FPS, quality=8, macro_block_size=1)
    print("wrote", path)


if __name__ == "__main__":
    OUT = os.environ.get("MEDIA_OUT", OUT)
    os.makedirs(OUT, exist_ok=True)
    # S11 roster. Each job carries its OWN stance offset, because the point of
    # three of these clips is what the stance does: the centred-stance clip is
    # there to show the bracing arm hanging off the table's side edge, which is
    # the thing the decentred stance fixes. Targets are each set's own
    # strength-limited envelope from reach_headtohead.py, so every clip shows the
    # pose at the limit rather than at a comfortable interior point.
    Y, Z = -0.2348, 1.0982
    jobs = [
        # (target x, subset, tag, stance dy)
        (1.10, (),                                  "s11_legs",        -0.25),
        (1.35, ("elbow", "forearm"),                "s11_EF",          -0.25),
        (1.40, ("elbow", "forearm", "palm"),        "s11_EFP",         -0.25),
        (1.30, ("elbow", "forearm", "palm"),        "s11_EFP_centred",  0.00),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for x, sub, tag, dy in jobs:
        if only and only != tag:
            continue
        cs.STANCE_DY = dy
        tgt = [x, Y, Z]
        print("== %s  target %s  stance dy %.2f" % (tag, tgt, dy), flush=True)
        m, d = approach(tgt, sub, tag)
        orbit(m, d, sub, tag, tgt)
