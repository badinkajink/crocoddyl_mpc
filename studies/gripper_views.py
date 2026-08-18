#!/usr/bin/env python3
"""Show the magpie gripper: the CAD it now is, and the proxy it used to be.

The 2026-08-16 asset fix is a geometry claim, and a table of half-extents is a
poor way to check a geometry claim. These renders put the two models side by
side and let the difference be seen:

  orbit       the corrected gripper at the braced pose, camera orbiting, visual
              mesh and collision proxy overlaid. What the robot now carries.
  roll        the wrist rolled through +-100 deg. This is the money shot for the
              transposition: the certified brace rolls the wrist ~89 deg to put
              the jaws LATERAL, and against the corrected gripper that same roll
              puts them somewhere else entirely.
  compare     old proxy vs new, same camera, same pose, as a two-panel video.
  jaws        the jaws opening and closing through the 4-bar's real range, which
              the old single-box proxy could not express at all.

Everything is rendered from the STAGED model, i.e. the same file the grid plans
against -- not a special-cased scene -- so what you see is what the planner has.

The comparison model is optional: pass --old <path to a pre-fix
Lean_H12_Magpie.xml> and the `compare` view is produced, otherwise it is skipped
with a note. Nothing else depends on it.

usage: gripper_views.py --out runs/<session>/media [--views orbit roll compare jaws]
                        [--old /path/to/old/Lean_H12_Magpie.xml]
"""
import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import croco_bridge as cb          # first: sets RTLD_GLOBAL
import contact_select as cs
import mujoco

FPS = 30
W, H = 960, 720

# The wrist the study braces on (contact_select.BRACE_ARM is 'left' for Allen's
# handedness); rendering follows it rather than hard-coding a side.
def brace_side():
    return getattr(cs, "BRACE_ARM", "left")


def load(path=None):
    if path is None:
        m, d = cs.load(ik_margin=0.0)
    else:
        m = mujoco.MjModel.from_xml_path(path)
        d = mujoco.MjData(m)
    hide(m, "_keepaway")
    return m, d


def pose_at(m, d, key="forearm_brace_reach"):
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key)
    if kid < 0:
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    mujoco.mj_forward(m, d)


def gripper_focus(m, d, side):
    """World point to orbit: the gripper body's own origin."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_magpie_gripper" % side)
    if b < 0:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                              "%s_wrist_yaw_link" % side)
    return d.xpos[b].copy()


def hide(m, *substrings):
    """Make geoms invisible by NAME, which geomgroup cannot do.

    `*_gripper_keepaway` shares group 3 with the real collision proxies but is
    not hardware -- it is a planning-only repulsion channel on its own contact
    bit. Left visible it is a 90 x 130 x 140 mm translucent slab centred on the
    gripper, and it hides the very geometry these renders exist to show.
    """
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if any(sub in n for sub in substrings):
            m.geom_rgba[g][3] = 0.0


def scene_option(show_collision):
    """Visual meshes always; collision proxies overlaid when asked.

    MuJoCo's group convention in this model: group 1 = visual, group 3 =
    collision. Showing both at once is the point -- a proxy that disagrees with
    the mesh it stands for is exactly what this page is about.
    """
    o = mujoco.MjvOption()
    for g in range(len(o.geomgroup)):
        o.geomgroup[g] = 0
    o.geomgroup[1] = 1
    if show_collision:
        o.geomgroup[3] = 1
    return o


def renderer(m):
    """A renderer at W x H, enlarging the model's offscreen framebuffer to match.

    The task XML declares <global offwidth="640" offheight="480"> and MuJoCo
    refuses any request larger than that rather than silently downscaling. The
    buffer is a VISUAL property, so raising it changes nothing the planner sees.
    """
    if m.vis.global_.offwidth < W:
        m.vis.global_.offwidth = W
    if m.vis.global_.offheight < H:
        m.vis.global_.offheight = H
    return mujoco.Renderer(m, H, W)


def cam_at(focus, azimuth, distance=0.55, elevation=-12):
    c = mujoco.MjvCamera()
    c.lookat[:] = focus
    c.distance = distance
    c.azimuth = azimuth
    c.elevation = elevation
    return c


def save(frames, path, fps=FPS):
    import imageio.v2 as imageio
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=1)
    print("  wrote %s  (%d frames)" % (path, len(frames)))


# --------------------------------------------------------------------- views

def view_orbit(out, n=120):
    """Camera orbits the corrected gripper; collision proxy overlaid."""
    m, d = load()
    pose_at(m, d)
    side = brace_side()
    focus = gripper_focus(m, d, side)
    r, opt = renderer(m), scene_option(True)
    frames = []
    for i in range(n):
        r.update_scene(d, camera=cam_at(focus, 360.0 * i / n), scene_option=opt)
        frames.append(r.render())
    save(frames, os.path.join(out, "s18_gripper_orbit.mp4"))


def view_roll(out, n=120, span=100.0):
    """Sweep the brace wrist's roll; this is where the 90 deg error showed."""
    m, d = load()
    pose_at(m, d)
    side = brace_side()
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT,
                            "%s_wrist_roll_joint" % side)
    if jid < 0:
        print("  no %s_wrist_roll_joint; skipping roll" % side)
        return
    adr = m.jnt_qposadr[jid]
    lo, hi = m.jnt_range[jid]
    q0 = d.qpos[adr]
    r, opt = renderer(m), scene_option(True)
    frames = []
    for i in range(n):
        # there and back, so the video loops cleanly
        f = np.sin(2 * np.pi * i / n)
        d.qpos[adr] = float(np.clip(q0 + np.deg2rad(span) * f, lo, hi))
        mujoco.mj_forward(m, d)
        focus = gripper_focus(m, d, side)
        r.update_scene(d, camera=cam_at(focus, 90, distance=0.5, elevation=-8),
                       scene_option=opt)
        frames.append(r.render())
    d.qpos[adr] = q0
    save(frames, os.path.join(out, "s18_gripper_wrist_roll.mp4"))


def view_jaws(out, n=90):
    """The jaws are welded, so this re-stamps the model at a ladder of openings.

    The gripper carries no joints in the planner model on purpose (see
    _gen_magpie_gripper.py), so a jaw animation cannot be produced by driving a
    qpos -- it needs the generator, once per opening.

    IT WORKS ON A COPY, NEVER ON THE STAGED TREE. Rewriting the staged model in
    place would be visible to anything else reading it, and a long grid run is
    exactly the kind of thing that is reading it -- a `plan` stage that picked up
    a half-written model would fail somewhere far away and for no visible reason.
    The whole task tree is copied first (symlinks preserved: the mesh dirs are
    relative links) and cs.MODEL is repointed at the copy for the duration.
    """
    import shutil
    import subprocess
    import tempfile

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gen = os.path.join(root,
                       "mjpc/tasks/humanoid_bench/h1_2_base/_gen_magpie_gripper.py")
    cl = os.environ.get("CL_ASSETS_DIR")
    if not (os.path.exists(gen) and cl):
        print("  need CL_ASSETS_DIR and the generator; skipping jaws")
        return

    lean_dir = os.path.dirname(os.path.abspath(cs.MODEL))
    hb_src = os.path.dirname(lean_dir)                     # .../humanoid_bench
    saved_model = cs.MODEL
    tmp = tempfile.mkdtemp(prefix="gripper_jaws_")
    frames = []
    try:
        hb = os.path.join(tmp, "humanoid_bench")
        shutil.copytree(hb_src, hb, symlinks=True)
        magpie = os.path.join(hb, "h1_2", "h1_2_modified_magpie.xml")
        model = os.path.join(hb, "lean", os.path.basename(saved_model))
        pristine = open(magpie).read()
        for i in range(12):
            f = 0.5 - 0.5 * np.cos(2 * np.pi * i / 12)      # 0 -> 1 -> 0
            jaw = 0.02 + f * (2.05 - 0.02)
            open(magpie, "w").write(pristine)
            rc = subprocess.run(
                [sys.executable, gen,
                 os.path.join(cl, "mujoco_assets/h1_2_magpie.xml"),
                 magpie, magpie, "--jaw", "%.4f" % jaw, "--collision", "cad",
                 "--verify-with", model],
                capture_output=True, text=True)
            if rc.returncode != 0:
                print("  jaw %.2f rad failed: %s"
                      % (jaw, (rc.stderr or "").strip()[-160:]))
                continue
            cs.MODEL = model
            m, d = load()
            pose_at(m, d)
            side = brace_side()
            focus = gripper_focus(m, d, side)
            r, opt = renderer(m), scene_option(True)
            r.update_scene(d, camera=cam_at(focus, 150, distance=0.42),
                           scene_option=opt)
            frames.extend([r.render()] * max(1, n // 12))
    finally:
        cs.MODEL = saved_model
        shutil.rmtree(tmp, ignore_errors=True)
    if frames:
        save(frames, os.path.join(out, "s18_gripper_jaws.mp4"))


def view_compare(out, old_path, n=120):
    """Old proxy vs corrected CAD, same pose, same camera, side by side."""
    if not old_path or not os.path.exists(old_path):
        print("  no --old model given; skipping compare")
        return
    side = brace_side()
    panels = []
    for path in (old_path, None):
        m, d = load(path)
        pose_at(m, d)
        focus = gripper_focus(m, d, side)
        r, opt = renderer(m), scene_option(True)
        fr = []
        for i in range(n):
            r.update_scene(d, camera=cam_at(focus, 360.0 * i / n, distance=0.45),
                           scene_option=opt)
            fr.append(r.render())
        panels.append(fr)
    bar = np.full((H, 4, 3), 255, np.uint8)
    save([np.hstack([a, bar, b]) for a, b in zip(*panels)],
         os.path.join(out, "s18_gripper_old_vs_new.mp4"))


VIEWS = {"orbit": view_orbit, "roll": view_roll,
         "jaws": view_jaws, "compare": view_compare}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", nargs="+", default=["orbit", "roll", "compare"],
                    choices=sorted(VIEWS))
    ap.add_argument("--old", default=None,
                    help="a pre-fix Lean_H12_Magpie.xml, for the compare view")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("model: %s" % cs.MODEL)
    for v in a.views:
        print("view %s" % v)
        if v == "compare":
            VIEWS[v](a.out, a.old)
        else:
            VIEWS[v](a.out)


if __name__ == "__main__":
    main()
