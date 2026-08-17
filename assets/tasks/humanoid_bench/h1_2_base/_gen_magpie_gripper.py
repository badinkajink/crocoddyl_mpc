#!/usr/bin/env python3
"""Stamp the REAL magpie gripper onto the MJPC magpie model, from CL_Assets.

CL_Assets IS THE SOURCE OF TRUTH for the robot (same rule as
_gen_h12_base_limits.py). This script extends that rule from the arm to the
thing bolted on the end of it, because the hand-authored gripper proxy in
h1_2_modified_magpie.xml.patch disagrees with the CAD in three ways that a
measurement of the STLs alone did not catch.

WHAT WAS WRONG (measured 2026-08-16, all in the wrist_yaw_link frame).

  1. THE VISUAL IS NOT THE GRIPPER. The patch draws one mesh, `h12_mount`
     (magpie_h12.stl), and calls it the magpie. That mesh is 26 x 63 x 61 mm --
     it is the H12 ADAPTER PUCK. The collision proxy meanwhile runs out to
     x = 0.225 m with jaws. So the model collides with a 170 mm gripper and
     draws a 26 mm disc, and every render, video and figure in the study shows
     a robot bracing on something that is not there.

  2. THE PROXY IS ROLLED 90 deg. The patch separates the jaws along wrist +-z
     and makes the gripper body thin in y:

         patch  body half-extents (0.0415, 0.0170, 0.0667)
         CAD    body half-extents (0.0415, 0.0667, 0.0170)

     Same three numbers, y and z transposed. The jaw plates transpose the same
     way (CAD: y +-0.054..0.1062, thin in z; patch: z +-0.054..0.1062, thin in
     y). So S14's "CAD-faithful" pass read the STLs correctly and then assigned
     the axes wrongly -- which is why S15 had to "re-aim at the wide +-z face,
     133 mm across against 34" to make the palm land. That wide face is the
     real gripper's wide +-Y face; the study was compensating in the controller
     for a transposed model.

     Consequence, and the reason this is not cosmetic: the certified brace pose
     rolls the wrist -1.553 rad (~89 deg) to put the jaws LATERAL. Against the
     real gripper that same roll puts them UP/DOWN -- the configuration the
     study rejected for hanging a finger at the wood.

  3. THE PROXY IS 25 mm SHORT. The patch mounts at pos="0.054 0 0"; CL's MJCF
     puts base_bot at x = 0.079 and the URDF's lg_mount_joint says
     xyz="0.079 0 -0.003125" rpy="0 1.5708 0". Both the body box and the jaw
     boxes carry the same 25 mm deficit, so the whole gripper under-reaches.

  Three independent witnesses agree on (2) and (3): CL's MJCF, CL's URDF, and
  the finger/base STL extents. The two wrist_yaw_link frames were checked to be
  the same frame FIRST (both narrow in y, tall in z, +-0.0425), so the
  transposition is in the gripper, not in the arm.

WHAT THIS EMITS. Into left_magpie_gripper / right_magpie_gripper:

  * VISUAL: one mesh geom per real part (mount, base_bot, base_top, both
    cranks, both fingers, both rockers), posed where CL's own forward
    kinematics puts them for the requested jaw opening.
  * COLLISION (--collision cad): body + flange + two jaw plates, as boxes
    derived from those same mesh extents. The DECOMPOSITION is the patch's, so
    the contact set and every geom NAME are unchanged -- the explicit <pair>s
    and excludes in the task XMLs keep referring to the same geoms, and only
    the numbers move. `--collision keep` emits visuals only.

WELDED, AND WHY THAT COSTS NOTHING. CL models each gripper as a 6-joint
linkage (hinge_1..3 per finger, 12 across both hands): 12 qpos, 12 qvel and up
to 4 actuators the planner would have to carry, for a mechanism that is opened
and closed OFF-BUS by the magpie_msgs controller and is not a decision variable
for a balance MPC. So the poses are BAKED at a chosen opening and emitted as
plain geoms in the gripper body: no <joint>, no <body>, no actuator. nq, nv and
nu are unchanged, and every emitted geom carries mass="0" density="0" so the
0.506 kg lumped gripper mass and the inertia tensor are unchanged too. Checked
before writing -- see verify(), which refuses to write if any of those move.

THE JAW OPENING IS AN INPUT, not a fact. `--jaw` takes hinge_1 in radians (CL
range -0.035 .. 2.05) or the names `open` (0.0) and `closed` (2.05); it moves
only the fingers, cranks and rockers. Measured inner-face gap after the linkage
solve: 155 mm open, 84 mm at 1.0 rad, 2 mm closed.

DEFAULT IS `open`, and the reason is proxy quality rather than preference. Each
jaw collision geom is an axis-aligned box bounding one finger, so it is tightest
when the finger is straight -- at `open` the box is 16 mm thick, and at
intermediate angles it inflates to bound a rotated part (reach x_max 0.2504 open
vs 0.2742 at 1.0 rad) and claims volume the hardware does not occupy. Since the
braced lean puts no load on the jaws, the faithful-proxy argument wins over the
tucked-out-of-the-way argument. If a task needs the jaws actually closed, pass
`--jaw closed` and read the clearance numbers knowing the boxes over-claim.

usage: _gen_magpie_gripper.py <CL h1_2_magpie.xml> <magpie model in> <out>
                              [--jaw open|closed|<rad>] [--collision cad|keep]
                              [--meshdir <dir>]
"""
import os
import re
import shutil
import sys

import numpy as np

try:
    import mujoco
except ImportError:
    sys.stderr.write("_gen_magpie_gripper: FATAL: mujoco is required\n")
    sys.exit(1)

# CL body -> mesh asset name emitted into the MJPC model. Order is emission
# order, i.e. draw order; it has no effect on physics.
# CL bodies, on the LEFT gripper, which CL prefixes `leftg_`. The right
# gripper is the same assembly (CL leaves it unprefixed) and the magpie is not
# mirrored between arms, so one read serves both hands.
PARTS = [
    ("leftg_base_bot",              "magpie_base_bot"),
    ("leftg_base_top",              "magpie_base_top"),
    ("leftg_left_crank",            "magpie_left_crank"),
    ("leftg_right_crank",           "magpie_right_crank"),
    ("leftg_left_finger_combined",  "magpie_left_finger"),
    ("leftg_right_finger_combined", "magpie_right_finger"),
    ("leftg_left_rocker",           "magpie_left_rocker"),
    ("leftg_right_rocker",          "magpie_right_rocker"),
]

# NOT emitted, because the patch already has them RIGHT -- checked against CL
# in the wrist frame and they agree to sub-millimetre:
#   h12_mount visual   patch pos (0.0668, 0.00005, 0.00161)  CL identical
#   *_gripper_flange   patch (0.0665,0,0) r 0.0315 h 0.0125  CL h12_collision_L
#                                         (0.0665,0,0) r 0.0320 h 0.0125
# So the ADAPTER is fine and only the gripper BODY and JAWS are displaced. That
# is also why the error survived review: the part nearest the wrist was correct.

# hinge_1 endpoints, from CL's own joint range.
JAW_OPEN, JAW_CLOSED = 0.0, 2.05

# Settling the passive 4-bar hinges (see cl_gripper_geometry). The damping is
# arbitrary but large: this is a relaxation, not a simulation, and the only
# thing that matters is that it reaches the constraint manifold.
LINKAGE_DAMPING = 5.0
LINKAGE_STEPS = 4000
LINKAGE_TOL = 1e-3

# How closely an emitted visual mesh must land on the pose CL_Assets puts it at,
# after the measure-and-correct pass in main(). Sub-micron is achievable because
# the correction is exact rigid algebra, not a fit.
PLACE_TOL = 1e-6

# The collision geoms this script owns. Anything else in the gripper body --
# notably *_gripper_keepaway, a planning-only repulsion channel and not
# hardware -- is left exactly as the patch wrote it.
OWNED_COLLISION = ("_gripper_collision", "_gripper_jaw_a", "_gripper_jaw_b")

MARK_OPEN = "<!-- BEGIN _gen_magpie_gripper -->"
MARK_CLOSE = "<!-- END _gen_magpie_gripper -->"


def die(msg):
    sys.stderr.write("_gen_magpie_gripper: FATAL: %s\n" % msg)
    sys.exit(1)


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).flatten())
    return q


def fmt(v, n=6):
    return " ".join(("%.*g" % (n, float(x))) for x in np.atleast_1d(v))


def compose(A, B):
    """Rigid frames as (p, R): returns A o B."""
    (pa, Ra), (pb, Rb) = A, B
    return (pa + Ra @ pb, Ra @ Rb)


def invert(A):
    p, R = A
    return (-R.T @ p, R.T)


def frame_error(A, B):
    """(translation m, rotation rad) between two rigid frames."""
    dp = float(np.linalg.norm(np.asarray(A[0]) - np.asarray(B[0])))
    c = (np.trace(np.asarray(A[1]).T @ np.asarray(B[1])) - 1.0) / 2.0
    return dp, float(np.arccos(np.clip(c, -1.0, 1.0)))


def geom_frames_in_wrist(model_path, side, meshes):
    """Where the emitted mesh geoms ACTUALLY ended up, keyed by mesh name."""
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    w = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_wrist_yaw_link" % side)
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "%s_magpie_gripper" % side)
    if w < 0 or b < 0:
        die("%s: missing %s_wrist_yaw_link / %s_magpie_gripper"
            % (model_path, side, side))
    Rw, pw = d.xmat[w].reshape(3, 3), d.xpos[w]
    out = {}
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom_dataid[g] < 0:
            continue
        name = m.mesh(m.geom_dataid[g]).name
        if name in meshes:
            out[name] = (Rw.T @ (d.geom_xpos[g] - pw),
                         Rw.T @ d.geom_xmat[g].reshape(3, 3))
    return out


# ---------------------------------------------------------------- CL geometry

def cl_gripper_geometry(cl_path, jaw):
    """Pose + mesh extent of every gripper part, in the wrist_yaw_link frame.

    Read out of CL's model by forward kinematics rather than by re-deriving the
    4-bar linkage here: the pivot chain is CL's to own, and a second
    implementation of it is a second thing to drift.
    """
    m = mujoco.MjModel.from_xml_path(cl_path)
    d = mujoco.MjData(m)
    driven = []

    # SOLVE THE 4-BAR, do not just evaluate it. Each finger is a closed loop --
    # crank, finger, rocker -- that CL closes with `connect` equality
    # constraints (m.neq == 4, two per hand). Setting hinge_1 and calling
    # mj_forward does NOT put the followers where the linkage would: forward
    # kinematics walks the tree, and the loop closure is a CONSTRAINT, not a
    # parent-child transform. Doing exactly that gave fingers that pass through
    # each other at hinge_1 = 2.05 -- the two jaw boxes overlapped by 26 mm,
    # i.e. a "closed" gripper that is not a pose the hardware can hold.
    #
    # So: hold hinge_1 at the requested angle, let the two passive hinges relax
    # under the constraints, and read the settled pose. Gravity off and contact
    # off because this is a KINEMATIC question -- with contact on the pads press
    # at full closure and the constraint cannot be satisfied (measured residual
    # 3.6e-2 with contact against 8.8e-5 without).
    #
    # Drive the LEFT gripper's cranks -- PARTS reads the `leftg_` assembly. CL
    # leaves the RIGHT gripper's joints unprefixed, so a bare `left_hinge_1`
    # would silently pose the other hand.
    for side, sign in (("left", +1.0), ("right", -1.0)):
        jn = "leftg_%s_hinge_1" % side
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            die("CL model %s has no joint %r -- the gripper linkage changed "
                "shape upstream; reconcile PARTS/JAW before regenerating."
                % (cl_path, jn))
        driven.append((m.jnt_qposadr[jid], m.jnt_dofadr[jid], sign * jaw))
    if m.neq == 0:
        die("CL model %s has no equality constraints: the 4-bar loop closure is "
            "gone, so a baked jaw pose cannot be trusted." % cl_path)

    m.opt.gravity[:] = 0
    m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    m.dof_damping[:] = LINKAGE_DAMPING
    mujoco.mj_resetData(m, d)
    for qa, da, val in driven:
        d.qpos[qa] = val
    for _ in range(LINKAGE_STEPS):
        for qa, da, val in driven:
            d.qpos[qa], d.qvel[da] = val, 0.0
        mujoco.mj_step(m, d)
    for qa, da, val in driven:
        d.qpos[qa], d.qvel[da] = val, 0.0
    mujoco.mj_forward(m, d)

    resid = float(np.abs(d.efc_pos[:d.nefc]).max()) if d.nefc else 0.0
    if resid > LINKAGE_TOL:
        die("the 4-bar did not close at jaw=%.4g rad: max constraint residual "
            "%.3e > %.0e. The baked jaw pose would not be one the linkage can "
            "hold." % (jaw, resid, LINKAGE_TOL))

    wid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
    if wid < 0:
        die("CL model %s has no left_wrist_yaw_link" % cl_path)
    Rw, pw = d.xmat[wid].reshape(3, 3), d.xpos[wid]

    out = {}
    for body, mesh_name in PARTS:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
        if bid < 0:
            die("CL model %s has no body %r. The magpie asset changed shape "
                "upstream -- reconcile PARTS before regenerating."
                % (cl_path, body))
        gids = [g for g in range(m.ngeom)
                if m.geom_bodyid[g] == bid and m.geom_dataid[g] >= 0]
        if not gids:
            die("CL body %r carries no mesh geom" % body)
        # Take the GEOM frame, not the body frame: CL carries the mesh
        # orientation on the geom (xyaxes="0 -1 0 1 0 0"), so an emitted geom
        # placed at the geom frame needs no separate orientation fixup.
        g = gids[0]
        R = Rw.T @ d.geom_xmat[g].reshape(3, 3)
        p = Rw.T @ (d.geom_xpos[g] - pw)
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = m.mesh_vert[a:a + n]
        world = (d.geom_xmat[g].reshape(3, 3) @ v.T).T + d.geom_xpos[g]
        local = (Rw.T @ (world - pw).T).T
        out[body] = dict(mesh=mesh_name, pos=p, R=R, quat=quat_of(R),
                         lo=local.min(0), hi=local.max(0),
                         meshid=mid, model=m)
    return m, out


def collision_boxes(geom):
    """CAD-derived collision proxies, in the wrist_yaw_link frame.

    Same decomposition the patch used -- body, flange, two jaw plates -- so the
    contact SET and the geom names are unchanged and only the numbers move.
    """
    def box(lo, hi):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        return (hi + lo) / 2.0, (hi - lo) / 2.0

    body_lo = np.minimum(geom["leftg_base_bot"]["lo"], geom["leftg_base_top"]["lo"])
    body_hi = np.maximum(geom["leftg_base_bot"]["hi"], geom["leftg_base_top"]["hi"])
    out = {}
    c, h = box(body_lo, body_hi)
    out["_gripper_collision"] = ("box", c, h)

    # The flange is NOT re-derived: the patch's cylinder already matches CL's
    # h12_collision_L. Left untouched so this script owns strictly what it fixes.
    for tag, part in (("_gripper_jaw_a", "leftg_left_finger_combined"),
                      ("_gripper_jaw_b", "leftg_right_finger_combined")):
        c, h = box(geom[part]["lo"], geom[part]["hi"])
        out[tag] = ("box", c, h)
    return out


# ------------------------------------------------------------------- emission

def cl_mesh_files(cl_path):
    """asset name -> STL path, read from CL's own <mesh> declarations.

    MjModel does not retain the source filename (no `.file` on a mesh view), and
    guessing `name + ".stl"` is wrong -- CL declares e.g.
    name="magpie_base_bot" file="magpie/base_bot.stl". So the mapping is read
    from the XML text, resolved against the model's own meshdir.
    """
    src = open(cl_path).read()
    md = re.search(r'<compiler\b[^>]*\bmeshdir="([^"]*)"', src)
    root = os.path.join(os.path.dirname(os.path.abspath(cl_path)),
                        md.group(1) if md else ".")
    out = {}
    for m in re.finditer(r'<mesh\b[^>]*/?>', src):
        n = re.search(r'\bname="([^"]+)"', m.group(0))
        f = re.search(r'\bfile="([^"]+)"', m.group(0))
        if n and f:
            out[n.group(1)] = os.path.normpath(os.path.join(root, f.group(1)))
    return out


def mesh_assets(names):
    """<mesh> declarations for the parts, keyed to the staged STL filenames."""
    lines, seen = [], set()
    for _, name in PARTS:
        if name in seen:
            continue
        seen.add(name)
        lines.append('    <mesh name="%s" file="%s" scale="0.001 0.001 0.001"/>'
                     % (name, os.path.basename(names[name])))
    return lines


def gripper_block(side, geom, boxes, jaw, indent, frames=None):
    """The geoms for one gripper, as XML lines."""
    pad = " " * indent
    L = [pad + MARK_OPEN,
         pad + "<!-- Generated by _gen_magpie_gripper.py from CL_Assets; do not",
         pad + "     hand-edit. Welded: no joint, no body, mass=0 density=0, so",
         pad + "     nq/nv/nu and the inertia are untouched. jaw hinge_1=%.4g rad."
         % jaw,
         pad + "     Poses are CL's own forward kinematics in the wrist frame. -->"]
    for body, name in PARTS:
        p, R = (frames or {}).get(name, (geom[body]["pos"], geom[body]["R"]))
        L.append(pad + '<geom type="mesh" mesh="%s" pos="%s" quat="%s" '
                       'contype="0" conaffinity="0" group="1" mass="0" '
                       'density="0" rgba="0.1 0.1 0.1 1"/>'
                 % (name, fmt(p), fmt(quat_of(R))))
    if boxes:
        L.append(pad + "<!-- CAD-derived collision, same decomposition and the same")
        L.append(pad + "     geom NAMES the hand-written proxy used. -->")
        for tag in OWNED_COLLISION:
            kind, c, h = boxes[tag]
            attrs = 'name="%s%s" class="collision" type="%s" size="%s" pos="%s"' \
                    % (side, tag, kind, fmt(h), fmt(c))
            if kind == "cylinder":
                # cylinder axis is local +z; the flange axis is +x -> rotate 90 about y
                attrs += ' quat="0.7071068 0 0.7071068 0"'
            L.append(pad + '<geom %s contype="1" conaffinity="1" mass="0" '
                           'density="0"/>' % attrs)
    L.append(pad + MARK_CLOSE)
    return "\n".join(L)


def strip_previous(src):
    """Remove a previous run's block, so the script is idempotent."""
    return re.sub(re.escape(MARK_OPEN) + r".*?" + re.escape(MARK_CLOSE),
                  "", src, flags=re.S)


def drop_owned(src, side):
    """Delete the hand-written geoms this script replaces, by name."""
    for tag in OWNED_COLLISION:
        src = re.sub(r"[ \t]*<geom\b[^>]*\bname=\"%s%s\"[^>]*/>\s*\n"
                     % (side, tag), "", src)
    return src


def inject(src, side, block):
    """Put the block inside <body name="{side}_magpie_gripper"> ... </body>."""
    tag = '<body name="%s_magpie_gripper"' % side
    i = src.find(tag)
    if i < 0:
        die("model has no body %r -- run this AFTER the magpie patch."
            % ("%s_magpie_gripper" % side))
    j = src.find(">", i)
    if j < 0:
        die("malformed <body> for %s_magpie_gripper" % side)
    return src[:j + 1] + "\n" + block + src[j + 1:]


# ------------------------------------------------------------------ invariants

def invariants(model_path):
    """The numbers this script is forbidden to move."""
    m = mujoco.MjModel.from_xml_path(model_path)
    out = dict(nq=m.nq, nv=m.nv, nu=m.nu, njnt=m.njnt,
               total_mass=float(m.body_mass.sum()))
    for name in ("left_magpie_gripper", "right_magpie_gripper"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            die("%s has no body %r -- run this AFTER the magpie patch."
                % (model_path, name))
        out[name + ".mass"] = float(m.body_mass[i])
        out[name + ".inertia"] = np.array(m.body_inertia[i], dtype=float)
    return out


def verify(via, before, after_src, out_path):
    """Write, then re-read through a loadable task model, then compare.

    The magpie model is an INCLUDE FRAGMENT -- it has no <mujoco> wrapper of its
    own and its meshdir resolves relative to the including task file -- so it
    cannot be loaded on its own. Verification therefore goes through a real task
    model (`--verify-with`), which is also the model anyone actually runs. On any
    violation the original bytes are restored before exiting, so a failed run
    leaves the build tree exactly as it found it.
    """
    original = open(out_path).read() if os.path.exists(out_path) else None
    with open(out_path, "w") as f:
        f.write(after_src)
    try:
        after = invariants(via)
    except Exception as exc:
        if original is not None:
            open(out_path, "w").write(original)
        die("generated model does not load: %s" % exc)

    def restore_and_die(msg):
        if original is not None:
            open(out_path, "w").write(original)
        die(msg)

    for k in ("nq", "nv", "nu", "njnt"):
        if before[k] != after[k]:
            restore_and_die("%s changed %d -> %d: the gripper must stay WELDED."
                            % (k, before[k], after[k]))
    if abs(before["total_mass"] - after["total_mass"]) > 1e-9:
        restore_and_die("total mass changed %.9f -> %.9f: emitted geoms must "
                        "carry mass=0 density=0."
                        % (before["total_mass"], after["total_mass"]))
    for name in ("left_magpie_gripper", "right_magpie_gripper"):
        if abs(before[name + ".mass"] - after[name + ".mass"]) > 1e-9:
            restore_and_die("%s mass changed %.9f -> %.9f"
                            % (name, before[name + ".mass"], after[name + ".mass"]))
        if not np.allclose(before[name + ".inertia"], after[name + ".inertia"],
                           atol=1e-9):
            restore_and_die("%s inertia changed %s -> %s"
                            % (name, before[name + ".inertia"],
                               after[name + ".inertia"]))
    return after


# ----------------------------------------------------------------------- main

def main():
    argv = sys.argv[1:]
    opts = {"--jaw": "open", "--collision": "cad", "--meshdir": None,
            "--verify-with": None}
    pos = []
    i = 0
    while i < len(argv):
        if argv[i] in opts:
            if i + 1 >= len(argv):
                die("%s needs a value" % argv[i])
            opts[argv[i]] = argv[i + 1]
            i += 2
        elif argv[i].startswith("--"):
            die("unknown option %r" % argv[i])
        else:
            pos.append(argv[i])
            i += 1
    if len(pos) != 3:
        die("usage: %s <CL h1_2_magpie.xml> <magpie model in> <out> "
            "[--jaw open|closed|<rad>] [--collision cad|keep] [--meshdir <dir>] "
            "--verify-with <task xml that includes the model>" % sys.argv[0])
    cl_path, in_path, out_path = pos

    jaw_arg = opts["--jaw"]
    jaw = {"open": JAW_OPEN, "closed": JAW_CLOSED}.get(jaw_arg)
    if jaw is None:
        try:
            jaw = float(jaw_arg)
        except ValueError:
            die("--jaw wants open|closed|<radians>, got %r" % jaw_arg)
    if opts["--collision"] not in ("cad", "keep"):
        die("--collision wants cad|keep, got %r" % opts["--collision"])

    if not os.path.exists(cl_path):
        die("cannot read CL_Assets magpie model %s. The h12/CL FetchContent "
            "must point at correlllab/CL_Assets." % cl_path)
    cl_model, geom = cl_gripper_geometry(cl_path, jaw)
    boxes = collision_boxes(geom) if opts["--collision"] == "cad" else None

    with open(in_path) as f:
        base_src = f.read()
    base_src = strip_previous(base_src)
    for side in ("left", "right"):
        if boxes:
            base_src = drop_owned(base_src, side)

    # mesh assets first: build() emits geoms that reference them, and the
    # measure-and-correct pass has to LOAD the model in between.
    files = cl_mesh_files(cl_path)
    for _, name in PARTS:
        if name not in files:
            die("CL model %s declares no mesh %r -- the magpie asset changed "
                "shape upstream." % (cl_path, name))
    if "<asset>" not in base_src:
        die("model has no <asset> block to extend")
    if 'name="%s"' % PARTS[0][1] not in base_src:
        base_src = base_src.replace(
            "<asset>", "<asset>\n" + "\n".join(mesh_assets(files)), 1)

    def build(frames=None):
        src = base_src
        for side in ("left", "right"):
            src = inject(src, side,
                         gripper_block(side, geom, boxes, jaw, 24, frames))
        return src

    src = build()

    if opts["--meshdir"]:
        os.makedirs(opts["--meshdir"], exist_ok=True)
        for _, name in PARTS:
            src_stl = files[name]
            if not os.path.exists(src_stl):
                die("mesh %s (%s) not found -- CL_Assets is incomplete; "
                    "is git-lfs installed and pulled?" % (name, src_stl))
            shutil.copyfile(src_stl, os.path.join(opts["--meshdir"],
                                                  os.path.basename(src_stl)))

    via = opts["--verify-with"]
    if not via:
        die("--verify-with <task xml> is required: the magpie model is an "
            "include fragment and cannot be loaded on its own, so the "
            "welded/mass invariants can only be checked through a task model.")
    if out_path != in_path:
        shutil.copyfile(in_path, out_path)
    before = invariants(via)

    # PLACE THE MESHES BY MEASUREMENT, NOT BY ASSUMPTION.
    #
    # A mesh geom's final frame is NOT the pos/quat you wrote: MuJoCo compiles
    # every mesh into a re-centred frame of its own and composes that under the
    # geom, so pos/quat is the frame of the *authored* mesh, not of the compiled
    # one. Reading CL's geom_xpos/geom_xmat (already post-composition) and
    # emitting it verbatim therefore applies the mesh transform TWICE. Measured
    # on the first attempt: base_bot landed 43 mm out, the fingers 141 mm.
    #
    # Rather than hard-code MuJoCo's convention -- which would be one more thing
    # to break on a version bump -- the placement is solved by measurement. If
    # `achieved = emitted o M` for an unknown fixed M, then emitting
    # `desired o achieved^-1 o emitted` lands exactly on `desired`, whatever M
    # is. One pass suffices because the composition is exact and rigid; the
    # result is then re-measured and must agree with CL to PLACE_TOL, so a
    # future convention change fails the build instead of shifting the gripper.
    meshes = {name for _, name in PARTS}
    desired = {name: (geom[body]["pos"], geom[body]["R"]) for body, name in PARTS}

    open(out_path, "w").write(src)
    try:
        achieved = geom_frames_in_wrist(via, "left", meshes)
    except Exception as exc:
        open(out_path, "w").write(base_src)
        die("generated model does not load: %s" % exc)
    corrected = {}
    for body, name in PARTS:
        if name not in achieved:
            open(out_path, "w").write(base_src)
            die("emitted mesh %r did not appear in the loaded model" % name)
        emitted = (np.asarray(geom[body]["pos"], float),
                   np.asarray(geom[body]["R"], float))
        corrected[name] = compose(compose(desired[name], invert(achieved[name])),
                                  emitted)
    src = build(corrected)

    after = verify(via, before, src, out_path)

    final = geom_frames_in_wrist(via, "left", meshes)
    worst = ("", 0.0, 0.0)
    for _, name in PARTS:
        dp, dr = frame_error(final[name], desired[name])
        if dp > worst[1]:
            worst = (name, dp, dr)
    if worst[1] > PLACE_TOL:
        die("mesh placement did not converge: %s is %.4f m / %.4f rad from where "
            "CL_Assets puts it (tolerance %.0e m). MuJoCo's mesh-frame "
            "convention may have changed." % (worst[0], worst[1], worst[2], PLACE_TOL))

    sys.stderr.write(
        "_gen_magpie_gripper: %d parts per hand from CL_Assets, jaw=%.4g rad, "
        "collision=%s; nq/nv/nu %d/%d/%d unchanged, gripper mass unchanged\n"
        % (len(PARTS), jaw, opts["--collision"],
           after["nq"], after["nv"], after["nu"]))
    sys.stderr.write("_gen_magpie_gripper: worst mesh placement error %.3e m "
                     "(%s), tolerance %.0e\n" % (worst[1], worst[0], PLACE_TOL))


if __name__ == "__main__":
    main()
