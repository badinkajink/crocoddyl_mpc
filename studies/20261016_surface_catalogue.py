#!/usr/bin/env python3
"""Read-only geometry comparison and opt-in brace patches; run under resguard.

No physics steps, planner imports, model edits or cell writes. Coordinates are
metres in MuJoCo body frames. Allowed extents are connected CAD triangle unions;
the seed triangle also supplies a guaranteed inscribed disk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull, cKDTree

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/lean/media/20261015-contact-audit'
AUDIT = OUT / '20261015-audit.json'
MODELS = {
    'staged': ROOT / 'studies/runs/_stage/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml',
    'current_cl': ROOT.parent / 'CL_Assets/mujoco_assets/h1_2_magpie.xml',
    'lean_export': ROOT.parent / 'mujoco_mpc/build_cmake/mjpc/tasks/humanoid_bench/lean/Lean_H12_Magpie.xml',
}
BODIES = ['left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_yaw_link',
          'left_magpie_gripper']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a)).hexdigest()


def name(m, kind, i):
    return mujoco.mj_id2name(m, getattr(mujoco.mjtObj, 'mjOBJ_' + kind), int(i))


def idx(m, kind, n):
    return mujoco.mj_name2id(m, getattr(mujoco.mjtObj, 'mjOBJ_' + kind), n)


def rot(q):
    a = np.zeros(9)
    mujoco.mju_quat2Mat(a, q)
    return a.reshape(3, 3)


def xml_dependencies(path):
    out = {}
    def visit(p):
        p = p.resolve()
        if str(p) in out:
            return
        out[str(p)] = sha(p)
        tree = ET.fromstring(re.sub(rb'<!--.*?-->', b'', p.read_bytes(), flags=re.S))
        for elem in tree.iter('include'):
            visit(p.parent / elem.attrib['file'])
    visit(path)
    return out


def mesh_files(path, xmls):
    """Resolve referenced mesh bytes without assuming include-relative meshdir."""
    trees = [ET.fromstring(re.sub(rb'<!--.*?-->', b'', Path(p).read_bytes(), flags=re.S))
             for p in xmls]
    meshdirs = [e.get('meshdir', '') for t in trees for e in t.iter('compiler')]
    result = {}
    for t in trees:
        for e in t.iter('mesh'):
            if 'file' not in e.attrib:
                continue
            f = e.attrib['file']
            candidates = [path.parent / md / f for md in meshdirs]
            candidates += [Path(p).parent / f for p in xmls]
            candidates += [path.parent / f]
            existing = list(dict.fromkeys(str(p.resolve()) for p in candidates if p.is_file()))
            if len(existing) != 1:
                raise ValueError(('Mesh resolution must be unique', f, existing))
            p = Path(existing[0])
            result[e.get('name', p.stem)] = {'path': str(p), 'sha256': sha(p),
                                           'scale': e.get('scale', '1 1 1')}
    return result


def mesh_triangles(m, g):
    mid = int(m.geom_dataid[g])
    va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
    v = m.geom_pos[g] + m.mesh_vert[va:va+vn].astype(float) @ rot(m.geom_quat[g]).T
    return v, m.mesh_face[fa:fa+fn].copy()


def georec(m, g):
    r = {'id': g, 'name': name(m, 'GEOM', g), 'body': name(m, 'BODY', m.geom_bodyid[g]),
         'type': mujoco.mjtGeom(int(m.geom_type[g])).name, 'pos_m': m.geom_pos[g].tolist(),
         'quat_wxyz': m.geom_quat[g].tolist(), 'size_m': m.geom_size[g].tolist(),
         'group': int(m.geom_group[g]), 'contype': int(m.geom_contype[g]),
         'conaffinity': int(m.geom_conaffinity[g]),
         'explicit_pairs': [name(m, 'GEOM', b if a == g else a) for a, b in
                            zip(m.pair_geom1, m.pair_geom2) if g in (a, b)]}
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        v, f = mesh_triangles(m, g)
        r.update(mesh=name(m, 'MESH', m.geom_dataid[g]),
                 bounds_body_m=[v.min(0).tolist(), v.max(0).tolist()],
                 vertices=len(v), triangles=len(f),
                 body_vertices_sha256=arr_sha(v), faces_sha256=arr_sha(f))
    return r


def inventory(m, path):
    xmls = xml_dependencies(path)
    fields = ('body_pos', 'body_quat', 'body_parentid', 'jnt_pos', 'jnt_axis',
              'geom_pos', 'geom_quat', 'geom_size', 'geom_type', 'geom_bodyid',
              'geom_dataid', 'geom_contype', 'geom_conaffinity', 'mesh_vert',
              'mesh_face', 'mesh_vertadr', 'mesh_vertnum', 'mesh_faceadr',
              'mesh_facenum', 'pair_geom1', 'pair_geom2')
    bodies = {}
    for b in range(m.nbody):
        n = name(m, 'BODY', b)
        bodies[n] = {'id': b, 'parent': name(m, 'BODY', m.body_parentid[b]),
                     'pos_parent_m': m.body_pos[b].tolist(),
                     'quat_parent_wxyz': m.body_quat[b].tolist(),
                     'joints': [name(m, 'JOINT', j) for j in range(m.njnt) if m.jnt_bodyid[j] == b]}
    return {'path': str(path.resolve()), 'xml_sha256': xmls, 'mesh_files': mesh_files(path, xmls),
            'loaded_model_array_sha256': {k: arr_sha(getattr(m, k)) for k in fields},
            'nq': m.nq, 'nv': m.nv, 'nu': m.nu, 'nbody': m.nbody, 'ngeom': m.ngeom,
            'bodies': bodies, 'geoms': [georec(m, g) for g in range(m.ngeom)
                                      if (name(m, 'BODY', m.geom_bodyid[g]) or '').startswith(('left', 'gripper_attachment_L'))]}


def set_pose(m, d, source, q):
    d.qpos[:] = m.qpos0
    copied = []
    for j in range(source.njnt):
        n = name(source, 'JOINT', j)
        target = idx(m, 'JOINT', n) if n else -1
        if target < 0:
            continue
        assert source.jnt_type[j] == m.jnt_type[target]
        count = 7 if source.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        a, b = source.jnt_qposadr[j], m.jnt_qposadr[target]
        d.qpos[b:b+count] = q[a:a+count]
        copied.append(n)
    mujoco.mj_forward(m, d)
    return copied


def probe(out):
    start = time.monotonic()
    audit = json.loads(AUDIT.read_text())
    models = {k: mujoco.MjModel.from_xml_path(str(p)) for k, p in MODELS.items()}
    inv = {k: inventory(m, MODELS[k]) for k, m in models.items()}
    result = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'mujoco_version': mujoco.__version__, 'audit_path': str(AUDIT),
              'audit_sha256': sha(AUDIT), 'models': inv, 'poses': [], 'frame_mappings': []}
    result['audit_reproduction'] = {
        'xml_equal': inv['staged']['xml_sha256'] == audit['provenance']['xml_sha256'],
        'arrays_equal': inv['staged']['loaded_model_array_sha256'] == audit['provenance']['loaded_model_array_sha256']}
    s = models['staged']
    for po in audit['poses']:
        q = np.asarray(po['qpos'])
        ds = mujoco.MjData(s)
        ds.qpos[:] = q
        mujoco.mj_forward(s, ds)
        rec = {'mode': po['mode'], 'qpos': po['qpos'], 'qpos_sha256': po['qpos_sha256'],
               'table_z_m': float(ds.geom_xpos[idx(s, 'GEOM', 'table_top_collision'), 2] +
                                  s.geom_size[idx(s, 'GEOM', 'table_top_collision'), 2]), 'models': {}}
        for key, m in models.items():
            d = mujoco.MjData(m)
            copied = set_pose(m, d, s, q)
            frames = {}
            for bn in BODIES:
                mapped = 'leftg_base_bot' if key == 'current_cl' and bn == 'left_magpie_gripper' else bn
                b = idx(m, 'BODY', mapped)
                a = idx(s, 'BODY', bn)
                if b < 0:
                    continue
                rs, rt = ds.xmat[a].reshape(3, 3), d.xmat[b].reshape(3, 3)
                frames[bn] = {'target_body': mapped,
                              'target_origin_in_staged_body_m': (rs.T @ (d.xpos[b]-ds.xpos[a])).tolist(),
                              'target_axes_in_staged_body': (rs.T @ rt).tolist()}
            rec['models'][key] = {'mapped_joints': copied, 'frames': frames}
        result['poses'].append(rec)
    result['elapsed_s'] = time.monotonic() - start
    out.write_text(json.dumps(result, indent=2) + '\n')
    print('Audit reproduction:', result['audit_reproduction'])
    for k, v in inv.items():
        print(k, v['nq'], v['ngeom'], MODELS[k])
        for g in v['geoms']:
            if g['body'] in BODIES or g['body'] in ('leftg_base_bot', 'leftg_base_top'):
                print(' ', g['id'], g['body'], g['name'], g.get('mesh', g['type']),
                      'bounds', g.get('bounds_body_m', g['size_m']))
    return result


def plane_patch(m, geom, target, outward, exposed=False):
    """Select a CAD facet and its connected coplanar triangles, preserving holes."""
    v, f = mesh_triangles(m, geom)
    tri = v[f]
    cross = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    twice = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(twice[:, None], 1e-30)
    sides = np.stack([np.linalg.norm(tri[:, 1]-tri[:, 2], axis=1),
                      np.linalg.norm(tri[:, 0]-tri[:, 2], axis=1),
                      np.linalg.norm(tri[:, 0]-tri[:, 1], axis=1)], axis=1)
    centers = (tri * sides[:, :, None]).sum(1) / np.maximum(sides.sum(1)[:, None], 1e-30)
    radii = twice / np.maximum(sides.sum(1), 1e-30)
    eligible = (normals @ outward > .80) & (radii > .0002)
    score = np.linalg.norm(centers-target, axis=1) - .25*radii
    score[~eligible] = np.inf
    seed = int(score.argmin())
    if exposed:
        for candidate in np.argsort(score):
            if not np.isfinite(score[candidate]):
                raise ValueError('No exposed facet near requested direction')
            if np.max(v @ normals[candidate])-centers[candidate] @ normals[candidate] < .0005:
                seed = int(candidate)
                break
    assert np.isfinite(score[seed])
    p, n = centers[seed], normals[seed]
    coplanar = (normals @ n > 1-1e-8) & (np.max(np.abs((tri-p) @ n), axis=1) < 1e-6)
    edge_faces = {}
    for i in np.flatnonzero(coplanar):
        for a, b in zip(f[i], np.roll(f[i], -1)):
            edge_faces.setdefault(tuple(sorted((int(a), int(b)))), []).append(int(i))
    selected, stack = {seed}, [seed]
    while stack:
        i = stack.pop()
        for a, b in zip(f[i], np.roll(f[i], -1)):
            for j in edge_faces[tuple(sorted((int(a), int(b))))]:
                if j not in selected:
                    selected.add(j)
                    stack.append(j)
    chosen = sorted(selected)
    ids = sorted(set(f[chosen].ravel().tolist()))
    mapping = {j: i for i, j in enumerate(ids)}
    faces = [[mapping[int(j)] for j in row] for row in f[chosen]]
    u = tri[seed, 1]-tri[seed, 0]
    u /= np.linalg.norm(u)
    return {'local_surface_point_m': p.tolist(), 'outward_normal_body': n.tolist(),
            'tangent_u_body': u.tolist(), 'tangent_v_body': np.cross(n, u).tolist(),
            'surface_mesh': name(m, 'MESH', m.geom_dataid[geom]), 'surface_geom_id': int(geom),
            'seed_face_index': seed,
            'cad_support_plane_inset_mm': float(1000*(np.max(v @ n)-p @ n)),
            'allowed_extent': {'type': 'connected_coplanar_triangle_union',
                               'source_face_indices': chosen, 'vertices_body_m': v[ids].tolist(),
                               'faces': faces, 'area_mm2': float(twice[chosen].sum()*5e5),
                               'bounds_body_m': [v[ids].min(0).tolist(), v[ids].max(0).tolist()],
                               'nominal_inscribed_disk_radius_m': float(.9*radii[seed]),
                               'normal_distance_tolerance_m': .0005,
                               'normal_angle_tolerance_deg': 25,
                               'rule': 'Projected point must lie in the triangle union; holes and all other bodies are excluded.'}}


def proxy_offset(m, g, point, outward):
    """Signed separation from proxy boundary; negative means inside the proxy."""
    r = rot(m.geom_quat[g])
    p = r.T @ (point-m.geom_pos[g])
    t, size = m.geom_type[g], m.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        delta = np.abs(p)-size
        return float(np.linalg.norm(np.maximum(delta, 0)) + min(float(delta.max()), 0))
    if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        nearest = np.array([0, 0, np.clip(p[2], -size[1], size[1])])
        return float(np.linalg.norm(p-nearest)-size[0])
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        v, _ = mesh_triangles(m, g)
        # Plane distance to the convex collision hull; exact inside, a lower
        # bound on distance outside near a hull edge.
        hull = ConvexHull(v)
        return float(np.max(hull.equations[:, :3] @ point + hull.equations[:, 3]))
    raise ValueError(t)


def comparison_metrics(comparison, m, cl):
    output = {'shared_mesh_hashes': [], 'mesh_body_frame_errors': [], 'collision_changes': []}
    models = comparison['models']
    for mesh, record in models['staged']['mesh_files'].items():
        for key in ('current_cl', 'lean_export'):
            other = models[key]['mesh_files'].get(mesh)
            output['shared_mesh_hashes'].append({'mesh': mesh, 'other_model': key,
                                                 'exists': other is not None,
                                                 'bytes_equal': other is not None and other['sha256'] == record['sha256']})
    sd, cd = mujoco.MjData(m), mujoco.MjData(cl)
    mujoco.mj_forward(m, sd)
    mujoco.mj_forward(cl, cd)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or name(m, 'BODY', m.geom_bodyid[g]) not in BODIES:
            continue
        mesh = name(m, 'MESH', m.geom_dataid[g])
        candidates = [h for h in range(cl.ngeom) if cl.geom_type[h] == mujoco.mjtGeom.mjGEOM_MESH
                      and name(cl, 'MESH', cl.geom_dataid[h]) == mesh
                      and (name(cl, 'BODY', cl.geom_bodyid[h]) or '').startswith(('left', 'gripper_attachment_L'))]
        if not candidates:
            continue
        h = candidates[0]
        ref = 'left_wrist_yaw_link' if mesh.startswith('magpie_') or mesh == 'h12_mount' else name(m, 'BODY', m.geom_bodyid[g])
        vs, _ = mesh_triangles(m, g)
        vc, _ = mesh_triangles(cl, h)
        def into_reference(model, data, geom, points):
            b, r = int(model.geom_bodyid[geom]), idx(model, 'BODY', ref)
            world = data.xpos[b] + points @ data.xmat[b].reshape(3, 3).T
            return (world-data.xpos[r]) @ data.xmat[r].reshape(3, 3)
        vs, vc = into_reference(m, sd, g, vs), into_reference(cl, cd, h, vc)
        # Linear-memory nearest-vertex queries, both directions; no dense distance matrix.
        err = max(float(cKDTree(vs).query(vc, workers=1)[0].max()),
                  float(cKDTree(vc).query(vs, workers=1)[0].max()))
        output['mesh_body_frame_errors'].append({'mesh': mesh, 'staged_geom_id': g,
                                                 'cl_geom_id': h, 'reference_body': ref,
                                                 'symmetric_vertex_distance_max_mm': 1000*err,
                                                 'jaw_configuration': 'Each model qpos0; no linkage settling'})
    for gn in ('left_forearm_pad', 'left_wrist_pad', 'left_gripper_collision', 'left_gripper_jaw_a', 'left_gripper_jaw_b'):
        row = {'geom': gn}
        for key in ('staged', 'lean_export'):
            row[key] = next(g for g in models[key]['geoms'] if g['name'] == gn)
        output['collision_changes'].append(row)
    return output


def create_catalogue():
    start = time.monotonic()
    comparison = json.loads((OUT / '20261016-model-comparison.json').read_text())
    audit = json.loads(AUDIT.read_text())
    m = mujoco.MjModel.from_xml_path(str(MODELS['staged']))
    cl = mujoco.MjModel.from_xml_path(str(MODELS['current_cl']))
    result = {'schema_version': 1, 'enabled_by_default': False, 'certified': False,
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'units': {'positions': 'm', 'normals': 'unit vectors', 'area': 'mm^2', 'distances': 'mm'},
              'model_comparison': '20261016-model-comparison.json',
              'model_comparison_sha256': sha(OUT / '20261016-model-comparison.json'),
              'staged_model_path': str(MODELS['staged']),
              'required_staged_xml_sha256': comparison['models']['staged']['xml_sha256'],
              'required_staged_array_sha256': comparison['models']['staged']['loaded_model_array_sha256'],
              'activation': 'Standalone research data only. A future consumer must opt in, verify hashes, require body and geom identity, then test triangle-union membership. No existing planner reads this file.',
              'normal_convention': 'Outward from robot solid; compressive support force on robot is opposite this vector.',
              'comparisons': comparison_metrics(comparison, m, cl), 'patches': [], 'poses': []}
    recess = plane_patch(m, 53, np.array([.1225, 0, -.018125]), np.array([0, 0, -1]))
    result['rejected_regions'] = [{'id': 'gripper_recess_nearest_old_palm',
                                  'reason': 'Base_bot rim protrudes 10 mm beyond this recessed surface; a flat table reaches the rim first along this normal.',
                                  'point_body_m': recess['local_surface_point_m'],
                                  'outward_normal_body': recess['outward_normal_body'],
                                  'cad_support_plane_inset_mm': recess['cad_support_plane_inset_mm']}]
    old = {s['site']: s for s in audit['poses'][0]['sites']}
    ref = next(p for p in audit['poses'] if p['mode'] == 'forearm+wrist')
    d = mujoco.MjData(m)
    d.qpos[:] = ref['qpos']
    mujoco.mj_forward(m, d)
    b = idx(m, 'BODY', 'left_shoulder_yaw_link')
    hit = next(c for c in ref['contacts'] if c['body'] == 'left_shoulder_yaw_link')
    elbow_target = d.xmat[b].reshape(3, 3).T @ (np.array(hit['pos'])-d.xpos[b])
    elbow_down = d.xmat[b].reshape(3, 3).T @ np.array([0, 0, -1.])
    specs = [
        ('E', 'upper_arm_elbow_end', 'elbow', 40, 41, elbow_target, elbow_down,
         'Distal upper-arm housing on left_shoulder_yaw_link, upstream of the elbow joint.'),
        ('F1', 'forearm_proximal', 'forearm', 42, 43, [0, -.029, -.044], [0, 0, -1],
         'Proximal forearm housing on left_elbow_link, downstream of the elbow joint.'),
        ('F2', 'forearm_distal', 'forearm', 42, 44, [.110, -.030, -.044], [0, 0, -1],
         'Distal forearm housing on left_elbow_link, before the wrist-roll joint.'),
        ('W', 'wrist_housing_underside', 'wrist', 49, 52, [.0215, 0, -.0425], [0, 0, -1],
         'Wrist-yaw housing on left_wrist_yaw_link; a separate articulated body from the forearm.'),
        ('P', 'gripper_base_rim', 'palm', 53, 61, [.1225, 0, -.018125], [0, 0, -1],
         'Exposed Magpie base_bot rim, welded to left_magpie_gripper in the staged model; leftg_base_bot in CL_Assets.'),
    ]
    for label, ident, oldsite, meshg, collision, target, normal, physical in specs:
        patch = plane_patch(m, meshg, np.asarray(target), np.asarray(normal), exposed=oldsite == 'palm')
        p, n = np.asarray(patch['local_surface_point_m']), np.asarray(patch['outward_normal_body'])
        proxy = proxy_offset(m, collision, p, n)
        patch.update(id=ident, label=label, body=name(m, 'BODY', m.geom_bodyid[meshg]),
                     physical_support=physical, collision_geom=georec(m, collision),
                     old_site=oldsite, old_site_body=old[oldsite]['body'],
                     old_site_point_body_m=old[oldsite]['body_offset'],
                     old_site_displacement_mm=float(np.linalg.norm(p-old[oldsite]['body_offset'])*1000),
                     nominal_proxy_signed_separation_mm=1000*proxy,
                     collision_surface_agrees_at_nominal=abs(proxy) <= .0005,
                     simulation_use='Requires collision surface agreement over the patch before certification; source collision proxies remain unchanged.')
        if oldsite == 'palm':
            frame = comparison['poses'][0]['models']['current_cl']['frames']['left_magpie_gripper']
            r, t = np.asarray(frame['target_axes_in_staged_body']), np.asarray(frame['target_origin_in_staged_body_m'])
            patch['current_cl_mapping'] = {'body': 'leftg_base_bot',
                                            'local_surface_point_m': (r.T @ (p-t)).tolist(),
                                            'outward_normal_body': (r.T @ n).tolist(),
                                            'collision_geom_id': 33, 'mesh': 'magpie_base_bot'}
        else:
            patch['current_cl_mapping'] = {'body': patch['body'], 'local_surface_point_m': p.tolist(),
                                           'outward_normal_body': n.tolist(),
                                           'collision_geom_id': None,
                                           'collision_status': 'Arm visual mesh contype=conaffinity=0; no table pair in CL robot XML.'}
        result['patches'].append(patch)
    # Record geom-specific intersections. Triangle-union membership validation
    # remains for continuation after the quota checkpoint.
    for po in audit['poses']:
        d.qpos[:] = po['qpos']
        mujoco.mj_forward(m, d)
        table = idx(m, 'GEOM', 'table_top_collision')
        z = d.geom_xpos[table, 2] + m.geom_size[table, 2]
        rows, contacts = [], []
        for i in range(d.ncon):
            c = d.contact[i]
            if table not in (c.geom1, c.geom2):
                continue
            g = c.geom2 if c.geom1 == table else c.geom1
            bn = name(m, 'BODY', m.geom_bodyid[g])
            if bn not in BODIES:
                continue
            contacts.append({'body': bn, 'geom_id': int(g), 'geom_name': name(m, 'GEOM', g),
                             'position_world_m': c.pos.tolist(), 'signed_gap_mm': float(1000*c.dist)})
        for p in result['patches']:
            b = idx(m, 'BODY', p['body'])
            r = d.xmat[b].reshape(3, 3)
            point = d.xpos[b] + r @ p['local_surface_point_m']
            norm = r @ p['outward_normal_body']
            verts = d.xpos[b] + np.asarray(p['allowed_extent']['vertices_body_m']) @ r.T
            same = [c for c in contacts if c['geom_id'] == p['collision_geom']['id']]
            rows.append({'id': p['id'], 'label': p['label'], 'world_point_m': point.tolist(),
                         'world_outward_normal': norm.tolist(),
                         'nominal_height_above_table_mm': float((point[2]-z)*1000),
                         'extent_height_min_max_mm': ((np.array([verts[:, 2].min(), verts[:, 2].max()])-z)*1000).tolist(),
                         'normal_angle_to_down_deg': float(np.degrees(np.arccos(np.clip(-norm[2], -1, 1)))),
                         'same_geom_contact_count': len(same),
                         'nearest_same_geom_contact_mm': min((float(np.linalg.norm(point-c['position_world_m'])*1000) for c in same), default=None)})
        result['poses'].append({'mode': po['mode'], 'qpos_sha256': po['qpos_sha256'],
                                 'source_qpos_file': str(ROOT / 'studies/runs/wristcell' / ('q_'+po['mode']+'.txt')),
                                 'patch_measurements': rows, 'table_contacts': contacts})
    render_patches(m, audit, result)
    result['elapsed_s'] = time.monotonic()-start
    target = OUT / '20261015-surface-catalogue.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved', target)
    for p in result['patches']:
        print(p['label'], p['body'], 'point', p['local_surface_point_m'], 'normal', p['outward_normal_body'],
              'area_mm2', p['allowed_extent']['area_mm2'], 'old_offset_mm', p['old_site_displacement_mm'],
              'proxy_offset_mm', p['nominal_proxy_signed_separation_mm'])
    for p in result['poses']:
        print(p['mode'], [(r['label'], round(r['nominal_height_above_table_mm'], 2)) for r in p['patch_measurements']])


def render_patches(m, audit, result, contacts_by_mode=None, filename_prefix='20261016-patches-'):
    colors = [(1, .2, .12, 1), (.1, .7, 1, 1), (.1, .9, .4, 1), (.8, .3, 1, 1), (1, .65, .05, 1)]
    m.vis.global_.offwidth, m.vis.global_.offheight = 1000, 800
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    d = mujoco.MjData(m)
    with mujoco.Renderer(m, height=720, width=1000) as renderer:
        for po, pr in zip(audit['poses'], result['poses']):
            d.qpos[:] = po['qpos']
            mujoco.mj_forward(m, d)
            panes = []
            for title, groups, az, elev in [('CAD surfaces, above table', [1, 1, 1, 0, 0, 0], 80, -20),
                                             ('Underside, table hidden', [1, 1, 1, 0, 0, 0], 80, 25),
                                             ('Collision geometry, table hidden', [1, 0, 0, 1, 0, 0], 80, 25)]:
                opt = mujoco.MjvOption()
                opt.geomgroup[:] = groups
                opt.sitegroup[:] = 0
                cam = mujoco.MjvCamera()
                cam.lookat[:] = np.mean([p['world_point_m'] for p in pr['patch_measurements']], axis=0)
                cam.distance, cam.azimuth, cam.elevation = .85, az, elev
                renderer.update_scene(d, camera=cam, scene_option=opt)
                hidden_bodies = {idx(m, 'BODY', 'table'), 0} if 'hidden' in title else set()
                # Hide the reaching arm only in these close-ups so it cannot
                # occlude the bracing arm; the saved robot pose is unchanged.
                hidden_bodies.update(b for b in range(m.nbody)
                                     if (name(m, 'BODY', b) or '').startswith('right_')
                                     and any(s in name(m, 'BODY', b) for s in ('shoulder', 'elbow', 'wrist', 'gripper')))
                for j in range(renderer.scene.ngeom):
                    sg = renderer.scene.geoms[j]
                    if sg.objtype == mujoco.mjtObj.mjOBJ_GEOM and sg.objid >= 0:
                        gid = int(sg.objid)
                        if int(m.geom_bodyid[gid]) in hidden_bodies:
                            sg.rgba[3] = 0
                def sphere(point, color, radius=.004):
                    sg = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(sg, mujoco.mjtGeom.mjGEOM_SPHERE, np.full(3, radius),
                                      point, np.eye(3).ravel(), np.array(color, dtype=np.float32))
                    renderer.scene.ngeom += 1
                def line(a, b, color, width=.001):
                    sg = renderer.scene.geoms[renderer.scene.ngeom]
                    mujoco.mjv_initGeom(sg, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                                      np.zeros(3), np.eye(3).ravel(), np.array(color, dtype=np.float32))
                    mujoco.mjv_connector(sg, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
                    renderer.scene.ngeom += 1
                for p, row, col in zip(result['patches'], pr['patch_measurements'], colors):
                    b = idx(m, 'BODY', p['body'])
                    r = d.xmat[b].reshape(3, 3)
                    normal = np.array(row['world_outward_normal'])
                    point = np.array(row['world_point_m'])
                    sphere(point+normal*.001, col)
                    line(point, point+normal*.03, col, .0015)
                    label_offset = {'E': [-.035, 0, -.025], 'F1': [.018, 0, -.035]}.get(p['label'], [0, 0, -.04])
                    label_point = point + np.asarray(label_offset)
                    line(point+normal*.03, label_point, (.65, .65, .65, 1), .0004)
                    sphere(label_point, col, .001)
                    renderer.scene.geoms[renderer.scene.ngeom-1].label = p['label']
                    ext = p['allowed_extent']
                    verts = d.xpos[b] + np.asarray(ext['vertices_body_m']) @ r.T + normal*.0007
                    counts = {}
                    for face in ext['faces']:
                        for a, b2 in zip(face, face[1:]+face[:1]):
                            edge = tuple(sorted((a, b2)))
                            counts[edge] = counts.get(edge, 0)+1
                    for (a, b2), count in counts.items():
                        if count == 1:
                            line(verts[a], verts[b2], col, .0008)
                    oldp = d.xpos[b] + r @ p['old_site_point_body_m']
                    sphere(oldp, (.95, .95, .95, 1), .003)
                    line(oldp, point, (.8, .8, .8, 1), .0006)
                if contacts_by_mode is not None:
                    for contact in contacts_by_mode[po['mode']]:
                        witness = np.asarray(contact['robot_surface_witness_world_m'])
                        sphere(witness, (.02, .02, .02, 1), .0035)
                        # Short white cross remains visible against dark CAD.
                        for delta in (np.array([.004, 0, 0]), np.array([0, .004, 0])):
                            line(witness-delta, witness+delta, (1., 1., 1., 1.), .0006)
                im = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(im)
                draw.rectangle((0, 0, 1000, 43), fill='white')
                draw.text((12, 10), po['mode']+' | '+title, fill='black', font=font)
                panes.append(im)
            panel = Image.new('RGB', (2000, 1480), 'white')
            panel.paste(panes[0], (0, 0))
            panel.paste(panes[1], (1000, 0))
            panel.paste(panes[2], (0, 720))
            draw = ImageDraw.Draw(panel)
            draw.text((1040, 760), 'Physical surface candidates', font=font, fill='black')
            for i, (p, row, col) in enumerate(zip(result['patches'], pr['patch_measurements'], colors)):
                y = 810+100*i
                draw.text((1040, y), p['label']+'  '+p['id'].replace('_', ' '), font=font,
                          fill=tuple(int(255*x) for x in col[:3]))
                draw.text((1040, y+26), 'Height %.1f mm; facet area %.1f mm2; old-site shift %.1f mm' %
                          (row['nominal_height_above_table_mm'], p['allowed_extent']['area_mm2'],
                           p['old_site_displacement_mm']), font=small, fill='black')
                draw.text((1040, y+49), p['body'], font=small, fill='black')
            draw.text((1040, 1350), 'Colored edges: allowed CAD facet. Line: outward normal (30 mm).', font=small, fill='black')
            draw.text((1040, 1380), 'White point: old site. Gray segment: displacement to candidate.', font=small, fill='black')
            draw.text((1040, 1410), 'Reaching arm hidden in close-ups. Static pose; no load estimate.', font=small, fill='black')
            if contacts_by_mode is not None:
                draw.text((1040, 1440), 'White crosses / black dots: robot-side collision witnesses.', font=small, fill='black')
            filename = filename_prefix+po['mode'].replace('+', '_')+'.png'
            panel.save(OUT / filename)
            pr['image'] = filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--out', type=Path, default=OUT / '20261016-model-comparison.json')
    args = parser.parse_args()
    if args.probe:
        probe(args.out)
    else:
        create_catalogue()


if __name__ == '__main__':
    main()
