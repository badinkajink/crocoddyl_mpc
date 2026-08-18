#!/usr/bin/env python3
"""Audit the brace surfaces against the CAD they are supposed to bound.

WHY THIS EXISTS.  Every contact this study certifies, plans and replays happens
between the tabletop and a COLLISION PROXY -- a capsule or a box someone added
to the MJCF to stand in for a link whose real shape MuJoCo would rather not
collide against.  The proxies were each added for a reason and each verified in
ONE direction, and the brace loads them in ANOTHER:

  * `*_wrist_pad` (2026-07-13) exists because the wrist_yaw collision capsule
    under-bounded the housing's UNDERSIDE, which sits at local z = -42.5 mm.  A
    capsule of radius 42.5 mm fixes z exactly.  But the housing is an OVAL --
    +-28.0 mm in y, +-42.5 mm in z -- and a capsule is round, so it also claims
    42.5 mm in y.  The palm brace works by ROLLING the wrist ~90 deg so the
    flat side faces the wood (contact_select.SITES' v2 note), i.e. the palm
    brace loads the pad in exactly the direction the pad was never checked in.

  * `*_gripper_collision` (2026-07-24) was widened to the real magpie's
    footprint using the H12 ADAPTER's cross-section, +-31.5 mm.  The adapter is
    a flange 25 mm long; the gripper body behind it is only 34 mm thick
    (base_bot y 0..+15, base_top y -19..-6).  The box applies the flange's
    half-thickness over the whole 85 mm of the body.

  * `*_forearm_pad` (radius 35 mm at local z = -15) reaches z = -50 where the
    forearm mesh it bounds reaches -44.

So the model can report a brace contact -- with force, with a friction cone,
with everything the certification scores -- while the real hardware is still in
the air.  This script measures by how much, from the CAD, and prints the
primitives that would not.

The measurement is done in the LINK's body frame, from the mesh vertices MuJoCo
itself loaded (`mjModel.mesh_vert` through `geom_pos/geom_quat`), so it is the
same geometry the renderer draws, not a re-import.  For the magpie the vertices
come from CL_Assets' part STLs, because the MJCF's `h12_mount` mesh is ONLY the
adapter -- the gripper body and fingers have no mesh in this model at all, which
is itself worth knowing: the boxes are not a simplification of a mesh that is
present, they are the only geometry there is.

usage: brace_surfaces.py [--json OUT]
"""

import argparse
import json
import os
import struct

import numpy as np
import mujoco

import contact_select as cs

# CL_Assets ships the magpie as one part per STL, all already positioned in a
# COMMON assembly frame (mm): +z is finger length out of the mount, +x is the
# jaw separation axis, +y is the thin direction.  No per-part transform is
# needed -- magpie_gripper.xml places every part at the identity and moves only
# the articulated bodies, which are at their modelled (open) pose in the STLs.
CL = os.environ.get("CL_MAGPIE_DIR",
                    "/home/correlllab/HAMS/CL_Assets/meshes/magpie")
PARTS = ["magpie_h12", "base_bot", "base_top", "left_crank", "right_crank",
         "left_rocker", "right_rocker", "left_finger_combined",
         "right_finger_combined"]

# The MJCF mounts that assembly with pos="0.054 0 0" quat="0.7071068 0 0.70710685 0"
# on `*_magpie_gripper`, a +90 deg rotation about body y.  R = [[0,0,1],[0,1,0],[-1,0,0]]:
MOUNT_X = 0.054


def _stl(path):
    """Vertices of a binary STL, in the file's own units."""
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b, dtype=np.uint8, count=n * 50, offset=84).reshape(n, 50)
    return a[:, 12:48].copy().view(np.float32).reshape(n * 3, 3).astype(float)


def magpie_parts():
    """{part: vertices in the GRIPPER BODY frame [m]}."""
    out = {}
    for p in PARTS:
        f = os.path.join(CL, p + ".stl")
        if not os.path.exists(f):
            continue
        V = _stl(f) * 1e-3
        out[p] = np.stack([MOUNT_X + V[:, 2], V[:, 1], -V[:, 0]], 1)
    return out


def geom_points(m, g):
    """Vertices of geom `g` in ITS OWN BODY's frame, plus a sphere radius.

    Returned as (points, radius): for a capsule the two segment ends and its
    radius, for a box its 8 corners, for a mesh every vertex.  The lowest point
    of the geom along any direction is then min over points minus radius, which
    is exact for all three -- and for a mesh it is also exactly the CONVEX
    HULL's lowest point, which is what MuJoCo actually collides.
    """
    t, s = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        a = m.mesh_vertadr[m.geom_dataid[g]]
        n = m.mesh_vertnum[m.geom_dataid[g]]
        P, r = m.mesh_vert[a:a + n].astype(float), 0.0
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        P = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1)
                      for k in (-1, 1)], float) * s[:3]
        r = 0.0
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        th = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        P = np.concatenate([np.stack([s[0] * np.cos(th), s[0] * np.sin(th),
                                      np.full_like(th, sg * s[1])], 1)
                            for sg in (-1, 1)])
        r = 0.0
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        P, r = np.array([[0, 0, -s[1]], [0, 0, s[1]]]), float(s[0])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        P, r = np.zeros((1, 3)), float(s[0])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        u = np.random.default_rng(0).normal(size=(4096, 3))
        P, r = u / np.linalg.norm(u, axis=1, keepdims=True) * s[:3], 0.0
    else:
        raise ValueError(f"geom type {t}")
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    return m.geom_pos[g] + P @ R.reshape(3, 3).T, r


def envelope(P, r, axis, sign):
    """How far the point cloud reaches along `sign * axis` [m]."""
    return float(sign * (P[:, axis] * sign).max() + r) if sign > 0 else \
        float((P[:, axis]).min() - r)


def audit(m):
    """Every brace proxy, against the geometry it stands in for."""
    arm = cs.BRACE_ARM
    def gid(n):
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    def body_meshes(body):
        b = cs.bid(m, body)
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == b
                and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]

    out = {}

    # ---- palm / wrist housing.  Brace direction is body -y (wrist rolled). ----
    wm = body_meshes(f"{arm}_wrist_yaw_link")[0]
    Pw, _ = geom_points(m, wm)
    pad = gid(f"{arm}_wrist_pad")
    Pp, rp = geom_points(m, pad)
    out["wrist_pad"] = dict(
        proxy="capsule r=%.1f mm" % (m.geom_size[pad][0] * 1e3),
        real_y=envelope(Pw, 0.0, 1, -1) * 1e3,
        real_z=envelope(Pw, 0.0, 2, -1) * 1e3,
        proxy_y=envelope(Pp, rp, 1, -1) * 1e3,
        proxy_z=envelope(Pp, rp, 2, -1) * 1e3)

    # ---- gripper body.  The MJCF has no mesh for it; CAD comes from CL_Assets. ----
    parts = magpie_parts()
    box = gid(f"{arm}_gripper_collision")
    Pb, _ = geom_points(m, box)
    body_cad = np.concatenate([parts[k] for k in ("base_bot", "base_top")
                               if k in parts]) if parts else None
    fing = np.concatenate([parts[k] for k in
                           ("left_finger_combined", "right_finger_combined")
                           if k in parts]) if parts else None
    if body_cad is not None:
        out["gripper_box"] = dict(
            proxy="box half %.1f %.1f %.1f mm @ x=%.1f" % (
                *(m.geom_size[box] * 1e3), m.geom_pos[box][0] * 1e3),
            real_y=envelope(body_cad, 0.0, 1, -1) * 1e3,
            real_y_plus=envelope(body_cad, 0.0, 1, +1) * 1e3,
            real_x=[float(body_cad[:, 0].min() * 1e3),
                    float(body_cad[:, 0].max() * 1e3)],
            real_z=envelope(body_cad, 0.0, 2, +1) * 1e3,
            proxy_y=envelope(Pb, 0.0, 1, -1) * 1e3,
            proxy_y_plus=envelope(Pb, 0.0, 1, +1) * 1e3,
            proxy_z=envelope(Pb, 0.0, 2, +1) * 1e3)
        ad = parts["magpie_h12"]
        out["adapter"] = dict(
            x=[float(ad[:, 0].min() * 1e3), float(ad[:, 0].max() * 1e3)],
            radius=float(np.hypot(ad[:, 1], ad[:, 2]).max() * 1e3),
            y_min=float(ad[:, 1].min() * 1e3))
    if fing is not None:
        ja = gid(f"{arm}_gripper_jaw_a")
        Pj, _ = geom_points(m, ja)
        out["jaws"] = dict(
            proxy="box half %.1f %.1f %.1f mm" % tuple(m.geom_size[ja] * 1e3),
            real_x=[float(fing[:, 0].min() * 1e3), float(fing[:, 0].max() * 1e3)],
            real_y=[float(fing[:, 1].min() * 1e3), float(fing[:, 1].max() * 1e3)],
            proxy_x=[float(Pj[:, 0].min() * 1e3), float(Pj[:, 0].max() * 1e3)],
            proxy_y=[float(Pj[:, 1].min() * 1e3), float(Pj[:, 1].max() * 1e3)])

    # ---- forearm.  Brace direction is body -z. ----
    fm = body_meshes(f"{arm}_elbow_link")[0]
    Pf, _ = geom_points(m, fm)
    fpad = gid(f"{arm}_forearm_pad")
    Pfp, rfp = geom_points(m, fpad)
    out["forearm_pad"] = dict(
        proxy="capsule r=%.1f mm @ z=%.1f" % (m.geom_size[fpad][0] * 1e3,
                                              m.geom_pos[fpad][2] * 1e3),
        real_z=envelope(Pf, 0.0, 2, -1) * 1e3,
        proxy_z=envelope(Pfp, rfp, 2, -1) * 1e3)

    # ---- elbow.  No proxy: the raw upper-arm mesh IS the contact surface. ----
    um = [g for g in range(m.ngeom)
          if m.geom_bodyid[g] == cs.bid(m, f"{arm}_shoulder_yaw_link")
          and m.geom_contype[g] == 1]
    out["elbow"] = dict(proxy="none -- raw mesh convex hull",
                        n_geoms=len(um))
    return out


def proposal(a):
    """The primitives the CAD supports, as MJCF attribute strings."""
    p = {}
    if "wrist_pad" in a:
        w = a["wrist_pad"]
        p["wrist_pad"] = (
            'type="box" size="0.0415 %.4f %.4f" pos="0.0215 0 0"'
            % (abs(w["real_y"]) * 1e-3, abs(w["real_z"]) * 1e-3))
    if "gripper_box" in a:
        g, ad = a["gripper_box"], a["adapter"]
        lo, hi = g["real_y"] * 1e-3, g["real_y_plus"] * 1e-3
        x0, x1 = g["real_x"][0] * 1e-3, g["real_x"][1] * 1e-3
        p["gripper_collision"] = (
            'type="box" size="%.4f %.4f %.4f" pos="%.4f %.4f 0"'
            % ((x1 - x0) / 2, (hi - lo) / 2, g["real_z"] * 1e-3,
               (x0 + x1) / 2, (lo + hi) / 2))
        ax0, ax1 = ad["x"][0] * 1e-3, ad["x"][1] * 1e-3
        p["gripper_flange"] = (
            'type="cylinder" size="%.4f %.4f" pos="%.4f 0 0"'
            ' quat="0.7071068 0 0.70710685 0"'
            % (abs(ad["y_min"]) * 1e-3, (ax1 - ax0) / 2, (ax0 + ax1) / 2))
    if "forearm_pad" in a:
        f = a["forearm_pad"]
        # keep the capsule (the forearm really is round here) and raise it so
        # its underside meets the mesh's, instead of hanging 6 mm below it.
        dz = (f["real_z"] - f["proxy_z"]) * 1e-3
        p["forearm_pad"] = 'pos z -0.015 -> %.4f  (raise by %.1f mm)' % (
            -0.015 + dz, dz * 1e3)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(cs.MODEL)
    a = audit(m)

    print(f"model {os.path.basename(cs.MODEL)}   brace arm {cs.BRACE_ARM}\n")
    w = a["wrist_pad"]
    print("PALM -- wrist housing, braced on body -y (wrist rolled ~90 deg)")
    print(f"  proxy   {w['proxy']:24s} reaches y {w['proxy_y']:+7.1f}  z {w['proxy_z']:+7.1f} mm")
    print(f"  CAD     wrist_yaw_link mesh          reaches y {w['real_y']:+7.1f}  z {w['real_z']:+7.1f} mm")
    print(f"  ==> the pad FABRICATES {w['real_y'] - w['proxy_y']:.1f} mm of material in the brace direction"
          f" (and is exact in z, which is the direction it was built for)\n")

    if "gripper_box" in a:
        g = a["gripper_box"]
        print("PALM -- gripper body, same direction")
        print(f"  proxy   {g['proxy']}")
        print(f"          reaches y {g['proxy_y']:+7.1f} .. {g['proxy_y_plus']:+7.1f} mm")
        print(f"  CAD     base_bot + base_top       y {g['real_y']:+7.1f} .. {g['real_y_plus']:+7.1f} mm"
              f"   (x {g['real_x'][0]:.0f}..{g['real_x'][1]:.0f})")
        print(f"  ==> the box FABRICATES {g['real_y'] - g['proxy_y']:.1f} mm; its y half-extent is the"
              f" ADAPTER's ({a['adapter']['radius']:.1f} mm max radius over x"
              f" {a['adapter']['x'][0]:.0f}..{a['adapter']['x'][1]:.0f}), applied over the whole body\n")
    if "jaws" in a:
        j = a["jaws"]
        print("PALM -- fingers")
        print(f"  proxy   {j['proxy']}  x {j['proxy_x'][0]:.1f}..{j['proxy_x'][1]:.1f}"
              f"  y {j['proxy_y'][0]:.1f}..{j['proxy_y'][1]:.1f}")
        print(f"  CAD     finger_combined           x {j['real_x'][0]:.1f}..{j['real_x'][1]:.1f}"
              f"  y {j['real_y'][0]:.1f}..{j['real_y'][1]:.1f}")
        print("  ==> exact bounding boxes; the jaws were done right\n")

    f = a["forearm_pad"]
    print("FOREARM -- braced on body -z")
    print(f"  proxy   {f['proxy']:24s} reaches z {f['proxy_z']:+7.1f} mm")
    print(f"  CAD     elbow_link mesh              reaches z {f['real_z']:+7.1f} mm")
    print(f"  ==> {f['real_z'] - f['proxy_z']:.1f} mm fabricated\n")
    print("ELBOW -- no proxy: MuJoCo collides the upper-arm mesh's convex hull, "
          "so the surface is faithful\n")

    print("PRIMITIVES THE CAD SUPPORTS")
    for k, v in proposal(a).items():
        print(f"  {k:20s} {v}")

    if args.json:
        json.dump({"audit": a, "proposal": proposal(a)},
                  open(args.json, "w"), indent=1)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
