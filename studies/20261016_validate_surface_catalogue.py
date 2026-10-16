#!/usr/bin/env python3
"""Validate the saved opt-in catalogue without selecting new facets or stepping physics.

Run in the croco environment through resguard. The result contains source
identity checks, region-wide collision bounds and per-contact membership.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('catalogue_source', HERE / '20261016_surface_catalogue.py')
source = importlib.util.module_from_spec(spec)
spec.loader.exec_module(source)
OUT = source.OUT
CAT = OUT / '20261015-surface-catalogue.json'


def member(patch, point, outward=None, body=None, geom=None):
    """Project onto each triangle; keep the union, with no tangential dilation."""
    ext = patch['allowed_extent']
    v = np.asarray(ext['vertices_body_m'])
    n = np.asarray(patch['outward_normal_body'])
    origin = np.asarray(patch['local_surface_point_m'])
    point = np.asarray(point)
    off = float((point-origin) @ n)
    projected = point-off*n
    inside = False
    best_barycentric = None
    for face in ext['faces']:
        a, b, c = v[face]
        e = np.stack((b-a, c-a), axis=1)
        uv = np.linalg.lstsq(e, projected-a, rcond=None)[0]
        barycentric = np.array([1-uv.sum(), *uv])
        if best_barycentric is None or barycentric.min() > min(best_barycentric):
            best_barycentric = barycentric.tolist()
        inside |= bool(barycentric.min() >= -1e-8)
    angle = None if outward is None else float(np.degrees(np.arccos(np.clip(n @ outward, -1, 1))))
    same_body = body is None or body == patch['body']
    same_geom = geom is None or geom == patch['collision_geom']['id']
    plane_ok = abs(off) <= ext['normal_distance_tolerance_m']
    angle_ok = angle is None or angle <= ext['normal_angle_tolerance_deg']
    return {'accepted': bool(same_body and same_geom and inside and plane_ok and angle_ok),
            'same_body': same_body, 'same_geom': same_geom,
            'projected_inside_triangle_union': inside, 'plane_distance_mm': off*1000,
            'normal_angle_deg': angle, 'within_plane_tolerance': plane_ok,
            'within_normal_tolerance': angle_ok, 'best_barycentric': best_barycentric}


def predicate_cases():
    # Two disconnected triangles make the hole/gap case independent of CAD data.
    p = {'body': 'a', 'collision_geom': {'id': 1},
         'local_surface_point_m': [0, 0, 0], 'outward_normal_body': [0, 0, 1],
         'allowed_extent': {'vertices_body_m': [[0, 0, 0], [1, 0, 0], [0, 1, 0],
                                               [2, 0, 0], [3, 0, 0], [2, 1, 0]],
                            'faces': [[0, 1, 2], [3, 4, 5]],
                            'normal_distance_tolerance_m': .0005,
                            'normal_angle_tolerance_deg': 25}}
    cases = [('Interior', [.2, .2, 0], [0, 0, 1], 'a', 1, True),
             ('Edge', [.5, .5, 0], [0, 0, 1], 'a', 1, True),
             ('Vertex', [0, 0, 0], [0, 0, 1], 'a', 1, True),
             ('Gap', [1.5, .2, 0], [0, 0, 1], 'a', 1, False),
             ('Beyond plane', [.2, .2, .00051], [0, 0, 1], 'a', 1, False),
             ('Reversed normal', [.2, .2, 0], [0, 0, -1], 'a', 1, False),
             ('Other body', [.2, .2, 0], [0, 0, 1], 'b', 1, False),
             ('Other geom', [.2, .2, 0], [0, 0, 1], 'a', 2, False)]
    rows = []
    for label, pos, normal, body, geom, expected in cases:
        actual = member(p, pos, normal, body, geom)['accepted']
        assert actual == expected, label
        rows.append({'case': label, 'expected': expected, 'actual': actual})
    return rows


def collision_bound(model, patch):
    """Bound every point of each triangle, not only nominal points or vertices."""
    g = patch['collision_geom']['id']
    verts = np.asarray(patch['allowed_extent']['vertices_body_m'])
    triangles = verts[np.asarray(patch['allowed_extent']['faces'])]
    normal = np.asarray(patch['outward_normal_body'])
    kind = model.geom_type[g]
    R = source.rot(model.geom_quat[g])
    local = (verts-model.geom_pos[g]) @ R
    size = model.geom_size[g]
    normals = []
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        v, _ = source.mesh_triangles(model, g)
        hull = ConvexHull(v)
        eq = hull.equations
        distances = verts @ eq[:, :3].T + eq[:, 3]
        upper = float(distances.max())
        # A convex combination stays inside all half-spaces. One supporting
        # plane per triangle gives a lower bound on max(plane distances).
        lower = min(float(distances[f].min(axis=0).max()) for f in patch['allowed_extent']['faces'])
        assert upper < 1e-9, 'Outside hull: interior-distance proof does not apply'
        max_abs = max(-lower, 0.) + 1e-9
        method = 'Inside convex hull: per-triangle supporting-plane lower bound; vertex upper bound. 1 nm rounding allowance.'
        for ds in distances:
            nearest = np.flatnonzero(ds >= ds.max()-1e-9)
            normals.append(eq[nearest[np.argmax(eq[nearest, :3] @ normal)], :3])
        samples = len(verts)
        cover = 0.
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        # Select the face aligned with the CAD normal. Projection of every
        # vertex lies in that rectangle, hence so does the full triangle.
        nl = R.T @ normal
        axis = int(np.argmax(np.abs(nl)))
        sign = float(np.sign(nl[axis]))
        axes = [j for j in range(3) if j != axis]
        assert np.all(np.abs(local[:, axes]) <= size[axes]+1e-8)
        sep = sign*local[:, axis]-size[axis]
        lower, upper = float(sep.min()), float(sep.max())
        max_abs = float(np.abs(sep).max())
        face_normal = np.zeros(3)
        face_normal[axis] = sign
        normals = [R @ face_normal]
        method = 'Distance to one box face; all projected triangle vertices lie in the face rectangle.'
        samples = len(verts)
        cover = 0.
    elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        values = []
        cover = 0.
        # Each subtriangle diameter <= 0.1 mm. The signed capsule distance is
        # 1-Lipschitz, so the unsampled minimum is at least sample_min-cover.
        # Its maximum is at a triangle vertex (distance to a segment is convex).
        for tri in triangles:
            edge = max(np.linalg.norm(tri[i]-tri[j]) for i, j in ((0, 1), (1, 2), (2, 0)))
            count = int(np.ceil(edge/.0001))
            cover = max(cover, edge/count)
            uv = np.array([(i/count, j/count) for i in range(count+1) for j in range(count+1-i)])
            points = tri[0] + uv[:, :1]*(tri[1]-tri[0]) + uv[:, 1:]*(tri[2]-tri[0])
            points = (points-model.geom_pos[g]) @ R
            nearest = np.zeros_like(points)
            nearest[:, 2] = np.clip(points[:, 2], -size[1], size[1])
            delta = points-nearest
            lengths = np.linalg.norm(delta, axis=1)
            values.extend((lengths-size[0]).tolist())
            normals.extend((delta/lengths[:, None] @ R.T).tolist())
        nearest = np.zeros_like(local)
        nearest[:, 2] = np.clip(local[:, 2], -size[1], size[1])
        upper = float((np.linalg.norm(local-nearest, axis=1)-size[0]).max())
        lower = min(values)-cover
        max_abs = max(abs(lower), abs(upper))
        method = 'Capsule signed distance: convex vertex maximum and 1-Lipschitz sampled minimum with 0.1 mm cover.'
        samples = len(values)
    else:
        raise ValueError(kind)
    angle = np.degrees(np.arccos(np.clip(np.asarray(normals) @ normal, -1, 1)))
    return {'method': method, 'signed_separation_bound_mm': [1000*lower, 1000*upper],
            'maximum_boundary_distance_bound_mm': 1000*max_abs,
            'collision_normal_angle_sample_max_deg': float(angle.max()),
            'samples': samples, 'cover_radius_bound_mm': 1000*cover,
            'full_region_within_0p5_mm': bool(max_abs <= .0005)}


def validate_patch(model, patch):
    ext = patch['allowed_extent']
    p = np.asarray(patch['local_surface_point_m'])
    n = np.asarray(patch['outward_normal_body'])
    v = np.asarray(ext['vertices_body_m'])
    triangles = v[np.asarray(ext['faces'])]
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    norms = cross/lengths[:, None]
    sv, sf = source.mesh_triangles(model, patch['surface_geom_id'])
    expected = sv[sf[ext['source_face_indices']]]
    vertex_error = float(np.linalg.norm(triangles-expected, axis=2).max())
    b = source.idx(model, 'BODY', patch['body'])
    basis = np.stack([patch['tangent_u_body'], patch['tangent_v_body'], n])
    assert int(model.geom_bodyid[patch['collision_geom']['id']]) == b
    assert int(model.geom_bodyid[patch['surface_geom_id']]) == b
    assert vertex_error < 1e-10
    assert np.max(np.abs(basis @ basis.T-np.eye(3))) < 1e-10
    assert np.min(norms @ n) > 1-1e-8
    assert member(patch, p)['accepted']
    assert all(member(patch, x)['accepted'] for x in v)
    area = float(lengths.sum()*5e5)
    assert abs(area-ext['area_mm2']) < 1e-8
    # The selected seed has a guaranteed 90%-inradius disk from the catalogue.
    # Report its diameter as a guaranteed circular footprint, not pad capacity.
    result = {'id': patch['id'], 'label': patch['label'], 'source_vertex_error_mm': vertex_error*1000,
              'unit_normal_error': float(abs(np.linalg.norm(n)-1)),
              'basis_orthonormal_error_max': float(np.abs(basis @ basis.T-np.eye(3)).max()),
              'source_normal_angle_max_deg': float(np.degrees(np.arccos(np.clip(norms @ n, -1, 1))).max()),
              'nominal_member': member(patch, p), 'area_mm2': area,
              'guaranteed_seed_disk_diameter_mm': 2000*ext['nominal_inscribed_disk_radius_m'],
              'collision_region': collision_bound(model, patch)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--update-catalogue', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    cat = json.loads(CAT.read_text())
    comparison = json.loads((OUT / cat['model_comparison']).read_text())
    audit = json.loads(source.AUDIT.read_text())
    geometry_fields = ('id', 'body', 'collision_geom', 'surface_geom_id', 'surface_mesh',
                       'local_surface_point_m', 'outward_normal_body', 'tangent_u_body',
                       'tangent_v_body', 'allowed_extent', 'current_cl_mapping')
    geometry = [{k: p[k] for k in geometry_fields} for p in cat['patches']]
    geometry_hash = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
    result = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'input_catalogue_sha256': source.sha(CAT), 'model_comparison_sha256': source.sha(OUT/cat['model_comparison']),
              'validated_geometry_sha256': geometry_hash, 'geometry_hash_fields': geometry_fields,
              'mujoco_version': mujoco.__version__, 'predicate_cases': predicate_cases(),
              'source_checks': {}, 'patches': [], 'poses': [], 'frame_checks': []}
    models = {}
    for key, record in comparison['models'].items():
        xml_ok = all(source.sha(p) == h for p, h in record['xml_sha256'].items())
        mesh_ok = all(source.sha(x['path']) == x['sha256'] for x in record['mesh_files'].values())
        m = mujoco.MjModel.from_xml_path(record['path'])
        arrays_ok = all(source.arr_sha(getattr(m, k)) == h for k, h in record['loaded_model_array_sha256'].items())
        assert xml_ok and mesh_ok and arrays_ok, key
        models[key] = m
        result['source_checks'][key] = {'xml_hashes_equal': xml_ok, 'mesh_hashes_equal': mesh_ok,
                                        'loaded_array_hashes_equal': arrays_ok,
                                        'xml_file_count': len(record['xml_sha256']),
                                        'mesh_file_count': len(record['mesh_files'])}
    model = models['staged']
    result['patches'] = [validate_patch(model, p) for p in cat['patches']]
    d = mujoco.MjData(model)
    table = source.idx(model, 'GEOM', 'table_top_collision')
    for po, saved in zip(audit['poses'], comparison['poses']):
        d.qpos[:] = po['qpos']
        mujoco.mj_forward(model, d)
        pose = {'mode': po['mode'], 'loaded_qpos_array_sha256': source.arr_sha(d.qpos),
                'source_qpos_file_sha256': source.sha(source.ROOT/'studies/runs/wristcell'/('q_'+po['mode']+'.txt')),
                'contacts': [], 'patch_counts': []}
        assert pose['source_qpos_file_sha256'] == po['qpos_sha256']
        for i in range(d.ncon):
            c = d.contact[i]
            if table not in (c.geom1, c.geom2):
                continue
            robot_first = c.geom2 == table
            g = int(c.geom1 if robot_first else c.geom2)
            b = int(model.geom_bodyid[g])
            body = source.name(model, 'BODY', b)
            if body not in source.BODIES:
                continue
            # Contact position is the midpoint. The contact normal points
            # geom1 -> geom2. Reconstruct the robot-side surface witness.
            outward = np.asarray(c.frame[:3]) * (1 if robot_first else -1)
            witness = c.pos - .5*c.dist*outward
            table_witness = c.pos + .5*c.dist*outward
            R = d.xmat[b].reshape(3, 3)
            local = R.T @ (witness-d.xpos[b])
            local_normal = R.T @ outward
            record = {'contact_index': i, 'body': body, 'geom_id': g, 'geom_name': source.name(model, 'GEOM', g),
                      'geom1': int(c.geom1), 'geom2': int(c.geom2), 'signed_gap_mm': float(c.dist*1000),
                      'midpoint_world_m': c.pos.tolist(), 'robot_surface_witness_world_m': witness.tolist(),
                      'robot_surface_witness_body_m': local.tolist(), 'outward_normal_body': local_normal.tolist(),
                      'table_witness_height_residual_mm': float(1000*(table_witness[2]-d.geom_xpos[table, 2]-model.geom_size[table, 2])),
                      'patch_membership': {}}
            for p in cat['patches']:
                pb = source.idx(model, 'BODY', p['body'])
                pr = d.xmat[pb].reshape(3, 3)
                point = pr.T @ (witness-d.xpos[pb])
                normal = pr.T @ outward
                record['patch_membership'][p['label']] = member(p, point, normal, body, g)
            pose['contacts'].append(record)
        for p in cat['patches']:
            same = [c for c in pose['contacts'] if c['geom_id'] == p['collision_geom']['id']]
            pose['patch_counts'].append({'label': p['label'], 'same_geom_contacts': len(same),
                                         'accepted_contacts': sum(c['patch_membership'][p['label']]['accepted'] for c in same)})
        result['poses'].append(pose)
        for key, m in models.items():
            data = mujoco.MjData(m)
            source.set_pose(m, data, model, d.qpos)
            for body, frame in saved['models'][key]['frames'].items():
                b = source.idx(model, 'BODY', body)
                other = source.idx(m, 'BODY', frame['target_body'])
                R = d.xmat[b].reshape(3, 3)
                actual_R = R.T @ data.xmat[other].reshape(3, 3)
                actual_t = R.T @ (data.xpos[other]-d.xpos[b])
                err_R = float(np.abs(actual_R-frame['target_axes_in_staged_body']).max())
                err_t = float(np.linalg.norm(actual_t-frame['target_origin_in_staged_body_m']))
                assert err_R < 1e-10 and err_t < 1e-10
                result['frame_checks'].append({'mode': po['mode'], 'model': key, 'body': body,
                                               'rotation_entry_error_max': err_R, 'origin_error_mm': err_t*1000})
    result['all_regions_within_0p5_mm'] = all(x['collision_region']['full_region_within_0p5_mm'] for x in result['patches'])
    if args.render:
        source.render_patches(model, audit, cat,
                              contacts_by_mode={p['mode']: p['contacts'] for p in result['poses']},
                              filename_prefix='20261016-validated-patches-')
        result['images'] = {p['image']: source.sha(OUT/p['image']) for p in cat['poses']}
    result['elapsed_s'] = time.monotonic()-started
    target = OUT/'20261016-surface-validation.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    if args.update_catalogue:
        assert result['all_regions_within_0p5_mm']
        cat['geometry_validation'] = {
            'result': target.name, 'result_sha256': source.sha(target),
            'geometry_sha256': geometry_hash, 'validated_utc': result['created_utc'],
            'full_regions_within_0p5_mm': True,
            'contact_point_convention': 'Robot-side witness = MuJoCo midpoint minus half the signed gap times the robot outward contact normal.',
            'extent_review': 'Retain exact CAD facets as position candidates. Guaranteed seed disk diameters are 0.574 to 3.251 mm; finite pad size and force capacity require separate design and loading measurements.',
            'contact_sampling_limit': 'A rejected collision witness does not prove separation of the full patch. Box-face contact records sample a manifold sparsely; force relocation from a corner into a facet requires a separate wrench model.'}
        for patch, validation in zip(cat['patches'], result['patches']):
            patch['validation'] = validation
            patch['simulation_use'] = 'Opt-in geometry candidate with full-region collision boundary agreement within 0.5 mm. Recompute contact feasibility and force allocation before using it in a planner.'
        for pose, measured in zip(cat['poses'], result['poses']):
            pose['region_contact_counts'] = measured['patch_counts']
            pose['source_qpos_file_sha256'] = measured['source_qpos_file_sha256']
        CAT.write_text(json.dumps(cat, indent=2)+'\n')
    print('Sources:', result['source_checks'])
    for p in result['patches']:
        print(p['label'], 'disk diameter mm', p['guaranteed_seed_disk_diameter_mm'], 'collision', p['collision_region'])
    for p in result['poses']:
        print(p['mode'], p['patch_counts'])
    print('Saved', target, 'elapsed', result['elapsed_s'])


if __name__ == '__main__':
    main()
