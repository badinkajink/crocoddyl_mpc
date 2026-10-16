#!/usr/bin/env python3
"""Measure brace-site geometry and render archived poses without advancing physics.

Run in the croco environment under resguard. LEAN_TASK_DIR selects the model.
The mesh distance is unsigned distance to rendered triangles, not the collision
convex hull. Contact distances are MuJoCo narrowphase results at the saved pose.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import contact_select as cs
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def rotation(q):
    out = np.empty(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def triangle_distance(p, triangles):
    """Exact point-to-triangle distance; bounded batches avoid large allocations."""
    best = float('inf')
    for first in range(0, len(triangles), 1024):
        t = triangles[first:first + 1024]
        a, b, c = t[:, 0], t[:, 1], t[:, 2]
        ab, ac = b - a, c - a
        normal = np.cross(ab, ac)
        nn = np.einsum('ij,ij->i', normal, normal)
        pa = p - a
        proj = p - normal * (np.einsum('ij,ij->i', pa, normal) /
                             np.maximum(nn, 1e-30))[:, None]
        ap = proj - a
        d00 = np.einsum('ij,ij->i', ab, ab)
        d01 = np.einsum('ij,ij->i', ab, ac)
        d11 = np.einsum('ij,ij->i', ac, ac)
        d20 = np.einsum('ij,ij->i', ap, ab)
        d21 = np.einsum('ij,ij->i', ap, ac)
        den = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / np.maximum(den, 1e-30)
        w = (d00 * d21 - d01 * d20) / np.maximum(den, 1e-30)
        inside = (nn > 1e-25) & (v >= 0) & (w >= 0) & (v + w <= 1)
        if inside.any():
            best = min(best, float(np.linalg.norm(p - proj[inside], axis=1).min()))
        for x, y in ((a, b), (b, c), (c, a)):
            e = y - x
            u = np.clip(np.einsum('ij,ij->i', p - x, e) /
                        np.maximum(np.einsum('ij,ij->i', e, e), 1e-30), 0, 1)
            closest = x + u[:, None] * e
            best = min(best, float(np.linalg.norm(p - closest, axis=1).min()))
    return best


def mesh_distance(m, body, point):
    records = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != body or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
        verts = m.mesh_vert[va:va + vn]
        faces = m.mesh_face[fa:fa + fn]
        local = rotation(m.geom_quat[g]).T @ (point - m.geom_pos[g])
        dist = triangle_distance(local, verts[faces])
        records.append({'geom': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                        'mesh': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MESH, mid),
                        'distance_mm': 1000 * dist})
    return min(records, key=lambda x: x['distance_mm']) if records else None


def provenance(m, model_path):
    """Pin XML dependencies and loaded geometry, including external mesh changes."""
    dependencies = {}

    def visit(path):
        path = Path(path).resolve()
        if str(path) in dependencies:
            return
        raw = path.read_bytes()
        dependencies[str(path)] = hashlib.sha256(raw).hexdigest()
        # Archived MJCF comments contain double hyphens accepted by MuJoCo.
        clean = re.sub(rb'<!--.*?-->', b'', raw, flags=re.DOTALL)
        for child in ET.fromstring(clean).iter('include'):
            visit(path.parent / child.attrib['file'])

    visit(model_path)
    fields = ('body_pos', 'body_quat', 'body_parentid', 'jnt_pos', 'jnt_axis',
              'geom_pos', 'geom_quat', 'geom_size', 'geom_type', 'geom_bodyid',
              'geom_dataid', 'geom_contype', 'geom_conaffinity', 'mesh_vert',
              'mesh_face', 'mesh_vertadr', 'mesh_vertnum', 'mesh_faceadr',
              'mesh_facenum', 'pair_geom1', 'pair_geom2')
    hashes = {name: hashlib.sha256(np.ascontiguousarray(getattr(m, name))).hexdigest()
              for name in fields}
    return {'xml_sha256': dependencies, 'loaded_model_array_sha256': hashes,
            'mujoco_version': mujoco.__version__}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cell', default='studies/runs/wristcell')
    ap.add_argument('--out', default='docs/lean/media/20261015-contact-audit')
    ap.add_argument('--close-azimuth', type=float, default=60)
    args = ap.parse_args()
    cell, out = Path(args.cell), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    m = mujoco.MjModel.from_xml_path(cs.MODEL)
    cs._resolve_palm(m)
    d = mujoco.MjData(m)
    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'table_top_collision')
    names = ('elbow', 'forearm', 'palm', 'wrist')
    colors = ((1, .2, .15, 1), (.1, .65, 1, 1), (1, .7, .05, 1), (.8, .2, 1, 1))
    body_ids = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, cs.SITES[n][0])
                for n in names}
    mesh = {n: mesh_distance(m, body_ids[n], cs.SITES[n][1]) for n in names}
    result = {'model': cs.MODEL,
              'model_sha256': hashlib.sha256(Path(cs.MODEL).read_bytes()).hexdigest(),
              'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'provenance': provenance(m, cs.MODEL),
              'cell_manifest': json.loads((cell / 'modes.json').read_text()),
              'mesh_distance_definition': 'Unsigned distance to triangles on the named body; mm.',
              'poses': []}
    m.vis.global_.offwidth = 900
    m.vis.global_.offheight = 700
    with mujoco.Renderer(m, height=650, width=850) as renderer:
        for mode in ('elbow+forearm', 'elbow+palm', 'forearm+wrist'):
            mujoco.mj_resetData(m, d)
            q = np.loadtxt(cell / ('q_' + mode + '.txt'))
            d.qpos[:len(q)] = q
            mujoco.mj_forward(m, d)
            ztop = d.geom_xpos[table, 2] + m.geom_size[table, 2]
            sites = {n: d.xpos[body_ids[n]] + d.xmat[body_ids[n]].reshape(3, 3) @ cs.SITES[n][1]
                     for n in names}
            contacts = []
            for i in range(d.ncon):
                c = d.contact[i]
                if table not in (c.geom1, c.geom2):
                    continue
                other = c.geom2 if c.geom1 == table else c.geom1
                body = int(m.geom_bodyid[other])
                if body not in body_ids.values():
                    continue
                contacts.append({'body': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, body),
                                 'geom': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other),
                                 'body_id': body, 'pos': c.pos.tolist(),
                                 'gap_mm': float(c.dist * 1000)})
            rows = []
            for n in names:
                same = [c for c in contacts if c['body_id'] == body_ids[n]]
                nearest = min(same, key=lambda c: np.linalg.norm(sites[n] - c['pos'])) if same else None
                rows.append({'site': n, 'selected': n in mode.split('+'),
                             'body': cs.SITES[n][0], 'body_offset': cs.SITES[n][1].tolist(),
                             'world_position': sites[n].tolist(),
                             'site_height_above_table_mm': float((sites[n][2] - ztop) * 1000),
                             'nearest_mesh': mesh[n],
                             'nearest_same_body_table_contact_mm': float(np.linalg.norm(sites[n] - nearest['pos']) * 1000) if nearest else None,
                             'same_body_contact_count': len(same)})
            result['poses'].append({'mode': mode, 'qpos': q.tolist(),
                                    'qpos_sha256': hashlib.sha256((cell / ('q_' + mode + '.txt')).read_bytes()).hexdigest(),
                                    'sites': rows, 'contacts': contacts})
            images = []
            for title, groups, distance, elev, az in (
                    ('Visual geometry', [1, 1, 1, 0, 0, 0], 2.6, -15, 125),
                    ('Visual geometry and sites', [1, 1, 1, 0, 0, 0], .85, -10, args.close_azimuth),
                    ('Collision geometry and sites', [1, 0, 0, 1, 1, 1], .85, -10, args.close_azimuth)):
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = groups
                cam = mujoco.MjvCamera()
                cam.lookat[:] = [.65, 0, .85] if distance > 1 else np.mean(list(sites.values()), axis=0)
                cam.distance, cam.elevation, cam.azimuth = distance, elev, az
                renderer.update_scene(d, camera=cam, scene_option=opt)
                if distance < 1:
                    for n, color in zip(names, colors):
                        g = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                          np.array([.006, .006, .006]), sites[n],
                                          np.eye(3).ravel(), np.asarray(color, dtype=np.float32))
                        g.label = n
                        renderer.scene.ngeom += 1
                    for c in contacts:
                        g = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                          np.array([.003, .003, .003]), np.array(c['pos']),
                                          np.eye(3).ravel(), np.array([.1, 1, .3, 1], dtype=np.float32))
                        renderer.scene.ngeom += 1
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 850, 30), fill='white')
                draw.text((10, 8), mode + ' | ' + title, fill='black')
                images.append(im)
            panel = Image.new('RGB', (2550, 650), 'white')
            for j, im in enumerate(images):
                panel.paste(im, (j * 850, 0))
            panel.save(out / ('20261015-' + mode.replace('+', '_') + '.png'))
    (out / '20261015-audit.json').write_text(json.dumps(result, indent=2) + '\n')
    for pose in result['poses']:
        print(pose['mode'])
        for x in pose['sites']:
            print(x['site'], 'selected', x['selected'], 'height_mm', round(x['site_height_above_table_mm'], 2),
                  'mesh_mm', round(x['nearest_mesh']['distance_mm'], 2) if x['nearest_mesh'] else None,
                  'contacts', x['same_body_contact_count'])
    print('Saved', out)


if __name__ == '__main__':
    main()
