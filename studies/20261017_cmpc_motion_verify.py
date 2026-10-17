#!/usr/bin/env python3
"""Review archived motion-transfer measurements without advancing physics.

Run under resguard. Reconstruct commands and contact kinematics from the saved
states; check the decision against raw measurements and artifact provenance.
"""
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('motion', ROOT/'studies/20261016_cmpc_motion_transfer.py')
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
np, mj, cs = M.np, M.mj, M.cs
MEDIA, OUT = M.MEDIA, M.OUT
report = json.loads((MEDIA/'20261016-cmpc-motion-transfer.json').read_text())
checks = []
measured = {}


def check(name, actual, expected, atol=1e-9):
    ok = bool(np.allclose(actual, expected, rtol=1e-10, atol=atol))
    checks.append(dict(name=name, passed=ok,
                       max_abs_error=float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))))
    if not ok:
        raise AssertionError((name, actual, expected))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


model, data = cs.load(ik_margin=0)
joints = model.actuator_trnid[:, 0]
qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
kp, kd = model.actuator_gainprm[:, 0], -model.actuator_biasprm[:, 2]
feet = list(M.F.FEET)
foot_bodies = {cs.bid(model, M.F.cb.SOLES[f][0]): f for f in feet}
jp, jr, contact_force = np.zeros((3, model.nv)), np.zeros((3, model.nv)), np.zeros(6)
initial = {}

for case in ['baseline', 'fixed-position', 'velocity']:
    stem = OUT/('20261016-'+case)
    cols, raw = M.load(stem)
    expected = report['cases'][case]
    t = raw['t']
    q = np.array([cols['qpos'+str(i)] for i in range(model.nq)]).T
    v = np.array([cols['qvel'+str(i)] for i in range(model.nv)]).T
    ctrl = np.array([cols['ctrl'+str(i)] for i in range(model.nu)]).T
    torque = np.array([cols['afrc'+str(i)] for i in range(model.nu)]).T
    initial[case] = (q[0], v[0])
    startup = t < .5-1e-9
    acc_mask = startup[1:] & startup[:-1]
    acceleration = np.diff(v[:, va], axis=0)/np.diff(t)[:, None]
    acc_idx = np.unravel_index(np.where(acc_mask[:, None], abs(acceleration), -1).argmax(), acceleration.shape)
    effective = np.where(model.actuator_ctrllimited,
                         np.clip(ctrl, *model.actuator_ctrlrange.T), ctrl)
    before_force_limit = kp*(effective-q[:, qa])-kd*v[:, va]
    after_force_limit = np.where(model.actuator_forcelimited,
                                np.clip(before_force_limit, *model.actuator_forcerange.T), before_force_limit)
    check(case+' actuator reconstruction', torque, after_force_limit)
    values = dict(
        startup_acceleration_rad_s2=float(abs(acceleration[acc_idx])),
        acceleration_time_s=float((t[acc_idx[0]]+t[acc_idx[0]+1])/2),
        startup_joint_speed_rad_s=float(abs(v[startup][:, va]).max()),
        startup_wrist_peak_m_s=float(np.linalg.norm(raw['point_v'][startup, 0], axis=1).max()),
        q_tracking_rms_rad=float(np.sqrt(np.mean((raw['q_des'][startup]-q[startup][:, qa])**2))),
        v_tracking_rms_rad_s=float(np.sqrt(np.mean((raw['v_des'][startup]-v[startup][:, va])**2))),
        startup_force_saturated_joint_samples=int(np.sum(abs(before_force_limit[startup]-after_force_limit[startup]) > 1e-9)),
        startup_ctrl_clipped_joint_samples=int(np.sum(abs(ctrl[startup]-effective[startup]) > 1e-9)),
        all_joint_limit_margin_min_rad=float(np.minimum(q[:, qa]-model.jnt_range[joints, 0], model.jnt_range[joints, 1]-q[:, qa]).min()))
    for key, value in values.items():
        check(case+' '+key, value, expected[key])

    # First loading and slip are recomputed at the original states and commands.
    onset = {f: None for f in feet}
    slip = {f: 0. for f in feet}
    for k, stamp in enumerate(t):
        data.qpos[:], data.qvel[:], data.ctrl[:] = q[k], v[k], ctrl[k]
        mj.mj_forward(model, data)
        for ci in range(data.ncon):
            contact = data.contact[ci]
            b = next((int(b) for b in model.geom_bodyid[contact.geom] if b in foot_bodies), None)
            if b is None or not any(model.geom_type[g] == mj.mjtGeom.mjGEOM_PLANE for g in contact.geom):
                continue
            f = foot_bodies[b]
            mj.mj_contactForce(model, data, ci, contact_force)
            if contact_force[0] > 1 and onset[f] is None:
                onset[f] = float(stamp)
            if contact_force[0] > 5:
                mj.mj_jac(model, data, jp, jr, contact.pos, b)
                slip[f] = max(slip[f], float(np.linalg.norm((jp@v[k])[:2])))
    for f in feet:
        check(case+' floor onset '+f, onset[f], expected['first_loaded_floor_s'][f])
        check(case+' loaded slip '+f, slip[f], expected['loaded_floor_slip_peak_m_s'][f])
    values.update(first_loaded_floor_s=onset, loaded_slip_m_s=slip)

    if case != 'fixed-position':
        rollout = expected['rollout']
        target = np.array(rollout['provenance']['plan_fields']['target'])
        final = t >= t[-1]-1.+1e-9
        reach = float(1000*np.sqrt(np.mean(np.sum((raw['point_p'][final, 1]-target)**2, axis=1))))
        check(case+' reach RMS', reach, expected['final_second_reach_rms_mm'])
        check(case+' final support', raw['table_normal_N'][final].min(), expected['final_second_table_normal_min_N'])
        for i, f in enumerate(feet):
            displacement = 1000*np.linalg.norm(raw['sole_p'][:, i]-raw['sole_p'][0, i], axis=1).max()
            check(case+' displacement '+f, displacement, rollout['feet'][f]['displacement_from_start_max_mm'])
        contacts = json.loads(Path(str(stem)+'-table-contacts.json').read_text())
        first = next(r for r in contacts if r['body'] == 'left_wrist_yaw_link' and r['normal_N'] > 1)
        k = int(np.argmin(abs(t-first['t'])))
        check(case+' wrist contact time', first['t'], expected['first_wrist_contact']['t'])
        check(case+' wrist contact speed', np.linalg.norm(raw['point_v'][k, 0]), expected['first_wrist_contact']['wrist_speed_m_s'])
        check(case+' solve p95', np.percentile(raw['pred_solve_s'], 95)*1000, rollout['solve_p95_ms'])
        check(case+' horizon samples', [w['T'] for w in rollout['window_checks']], 35)
        check(case+' final window', raw['pred_k'].max(), 165)
        check(case+' estimated initial base bias', raw['pred_x_est'][0, :3]-q[0, :3], rollout['observation']['bias_position_m'])
        values.update(reach_rms_mm=reach, solver_samples=len(raw['pred_t']),
                      estimated_initial_base_bias_m=(raw['pred_x_est'][0, :3]-q[0, :3]).tolist())
    measured[case] = values

for case in ['fixed-position', 'velocity']:
    for index, label in enumerate(['qpos', 'qvel']):
        check(case+' common initial '+label, initial[case][index], initial['baseline'][index])

archive = json.loads((MEDIA/'20261016-motion-archive.json').read_text())
assert sha(ROOT/archive['archive']) == archive['sha256']
for name, digest in archive['source_scripts'].items():
    assert sha(ROOT/name) == digest, name
with zipfile.ZipFile(ROOT/archive['archive']) as z:
    assert z.testzip() is None
    for name in z.namelist():
        assert hashlib.sha256(z.read(name)).hexdigest() == sha(OUT/name), name
    archive_members = len(z.namelist())

videos = {}
for case in measured:
    path = MEDIA/('20261016-motion-'+case+'.mp4')
    metadata = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == metadata['video_sha256']
    assert sha(metadata['source']) == metadata['source_sha256']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,nb_frames,duration', '-of', 'json', str(path)]))['streams'][0]
    check(case+' video frames', int(probe['nb_frames']), metadata['frames'])
    assert probe['avg_frame_rate'] == '30/1' and metadata['playback_speed'] == 1
    check(case+' video duration', float(probe['duration']), metadata['duration_s'], atol=1e-6)
    videos[case] = probe

result = dict(checkpoint='motion_transfer',review_date='2026-10-17 America/Denver',
    new_rollouts=0,cumulative_rollouts=3,rollout_limit=6,physics_steps=0,
    passed=True,checks=checks,measurements=measured,
    archive=dict(sha256=archive['sha256'],all_members_match_run_directory=True,members=archive_members,
                 source_scripts_match_manifest=True),videos=videos,
    limitations='Saved-state forward evaluation and source review; no new rollout, longer-duration solve, supported-handover test or hardware measurement.')
destination = MEDIA/'20261017-motion-review.json'
with destination.open('w') as f:
    json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(dict(passed=True,checks=len(checks),archive_members=archive_members,rollouts=0,output=str(destination))),flush=True)
