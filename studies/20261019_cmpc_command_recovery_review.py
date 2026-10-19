#!/usr/bin/env python3
"""Independent saved-command and recovery review. Run under resguard.

No plant steps, MPC solves, new rollouts, or calls to the original scorer.
Rebuild action models only to evaluate archived estimated states and commands.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT/'studies/20261018_cmpc_command_recovery.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
F, np, OUT, MEDIA = R.F, R.np, R.OUT, R.MEDIA
report = json.loads((MEDIA/'20261016-cmpc-command-recovery.json').read_text())
checks, measured, arrays = [], {}, {}


def check(name, value, expected, tolerance=1e-8):
    if value is None or expected is None:
        passed, error = value is expected, None
    else:
        error = float(np.max(abs(np.asarray(value)-np.asarray(expected))))
        passed = bool(np.allclose(value, expected, atol=tolerance, rtol=1e-10))
    checks.append(dict(name=name, passed=passed, maximum_error=error))
    assert passed, (name, value, expected)


def onset(t, mask, start=0., stop=float('inf')):
    # An independent contiguous-run calculation for 150 logged 2 ms intervals.
    indices = np.flatnonzero(mask & (t >= start-1e-9) & (t < stop-1e-9))
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    return next((float(t[g[0]]) for g in groups if len(g) >= 150), None)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            h.update(data)
    return h.hexdigest()


fl = F.build()
m, ocp = fl.m, fl.mpc.ocp
ret = ocp.build(dt=.02, n_approach=0, n_braced=0, n_return=35, cones=True)
check('return horizon', ret.T, 35)
assert all(set(am.differential.contacts.contacts.todict()) == set(F.FEET) for am in ret.runningModels)
qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
va = m.jnt_dofadr[m.actuator_trnid[:, 0]]
cache, videos = {}, {}
provenance = F.provenance(fl)

for tag, expected in report['cases'].items():
    stem = OUT/('20261018-'+tag)
    a = dict(np.load(str(stem)+'-raw.npz'))
    arrays[tag] = a
    with Path(str(stem)+'.csv').open() as f:
        reader = csv.reader(line for line in f if not line.startswith('#'))
        columns = next(reader)
        raw = np.array(list(reader), float)
    def col(prefix, n):
        return raw[:, [columns.index(prefix+str(i)) for i in range(n)]]
    q, v, ctrl, actual = col('qpos', m.nq), col('qvel', m.nv), col('ctrl', m.nu), col('afrc', m.nu)
    t, ts = a['t'], a['pred_t']
    check(tag+' physical sample interval', np.diff(t), .002)
    check(tag+' command interval', np.diff(ts), .02)
    check(tag+' horizon', [z['T'] for z in expected['summary']['window_checks']], 35)
    for key in ['plan_sha256', 'xs_sha256', 'us_sha256', 'model_sha256', 'model_arrays_sha256']:
        assert provenance[key] == expected['summary']['provenance'][key], (tag, key)

    commands = json.loads(Path(str(stem)+'-commands.json').read_text())
    gates = json.loads(Path(str(stem)+'-gate.json').read_text())
    rows = json.loads(Path(str(stem)+'-executable-after.json').read_text())
    force_map = {(r['t'], r['contact']): r for r in rows}
    next_states, torques = [], []
    wrench_error = row_error = cone_excess = 0.
    for j, (stamp, x, k) in enumerate(zip(ts, a['pred_x_est'], a['pred_k'])):
        cmd = commands[j]
        assert abs(cmd['t']-stamp) < 1e-10
        combined = np.array(cmd['q'])+(fl.kd*np.array(cmd['v'])+np.array(cmd['tau']))/fl.kp
        clipped = np.where(m.actuator_ctrllimited, np.clip(combined, *m.actuator_ctrlrange.T), combined)
        tau = fl.kp*(clipped-x[7:34])-fl.kd*x[40:]
        tau = np.where(m.actuator_forcelimited, np.clip(tau, *m.actuator_forcerange.T), tau)
        torques.append(tau)
        returning = stamp >= R.RELEASE-1e-9
        am = ret.runningModels[0] if returning else fl.mpc.models[int(k)]
        assert gates[j]['mode'] == ('return' if returning else 'brace')
        if id(am) not in cache:
            cache[id(am)] = am.createData()
        dat = cache[id(am)]
        am.calc(dat, x, tau)
        next_states.append(dat.xnext.copy())
        costs = am.differential.costs.costs.todict()
        for name, cd in dat.differential.multibody.contacts.contacts.todict().items():
            cone = costs['cone_'+name.removeprefix('brace_')].cost.residual.reference
            w = np.asarray(cd.f.vector)
            values = cone.A@w[:cone.A.shape[1]]
            excess = np.maximum(0., np.maximum(cone.lb-values, values-cone.ub))
            saved = force_map[(float(stamp), name)]
            wrench_error = max(wrench_error, float(abs(w-saved['wrench_world']).max()))
            row_error = max(row_error, float(abs(excess-saved['cone_violation']).max()))
            if 4.-1e-9 <= stamp < 9.-1e-9:
                cone_excess = max(cone_excess, float(excess.max()))
        if stamp >= 4.-1e-9:
            assert gates[j]['status'] == 'projected' and 'fallback' not in gates[j]
            assert np.max(abs(np.array(cmd['tau']))-.9*fl.tau_lim) < 1e-8
            assert np.max(abs(clipped-combined)) < 1e-8
    torques = np.array(torques)
    check(tag+' next-state reconstruction', next_states, a['pred_model_next'])
    check(tag+' torque reconstruction', torques, a['pred_total_torque'])
    check(tag+' wrench reconstruction', wrench_error, 0.)
    check(tag+' cone reconstruction', row_error, 0.)
    check(tag+' first-command cone feasibility', cone_excess, 0., 1e-6)
    indices = np.rint(ts/.002).astype(int)
    valid = indices < len(t)
    check(tag+' initial physical torque', torques[valid], actual[indices[valid]])
    check(tag+' estimated joint position', a['pred_x_est'][valid, 7:34], q[indices[valid]][:, qa])
    check(tag+' estimated joint velocity', a['pred_x_est'][valid, 40:], v[indices[valid]][:, va])
    clipped = np.where(m.actuator_ctrllimited, np.clip(ctrl, *m.actuator_ctrlrange.T), ctrl)
    servo = m.actuator_gainprm[:, 0]*(clipped-q[:, qa])+m.actuator_biasprm[:, 2]*v[:, va]
    servo = np.where(m.actuator_forcelimited, np.clip(servo, *m.actuator_forcerange.T), servo)
    check(tag+' every physical servo sample', servo, actual)

    table = json.loads(Path(str(stem)+'-table.json').read_text())
    robot_load = np.array([sum(value for body, value in z['body_vertical_force_N'].items() if body != 'object') for z in table])
    object_load = np.array([z['body_vertical_force_N'].get('object', 0.) for z in table])
    check(tag+' table accounting', robot_load+object_load, a['table_force_N'])
    both = (a['sole_w'][:, :, 2] > 5).all(axis=1)
    support = both & (robot_load > 20)
    speed = abs(v[:, va]).max(axis=1)
    wrist = np.linalg.norm(a['point_v'][:, 0], axis=1)
    quiet = (speed < .1) & (wrist < .03)
    slip = np.linalg.norm(a['sole_v'][:, :, :2], axis=2)
    ref = np.searchsorted(t, R.RELEASE)
    displacement = 1000*np.linalg.norm(a['sole_p']-a['sole_p'][ref], axis=2)
    still = (displacement < 5).all(axis=1) & (slip < .02).all(axis=1)
    stand = np.array(expected['summary']['standing_reference'])
    upright = (np.mean((q[:, qa]-stand[qa])**2, axis=1) < .1**2) & (abs(q[:, 2]-stand[2]) < .05)
    recovery = both & quiet & still & upright & (robot_load < 5)
    abort = support & quiet & still
    original_error = np.linalg.norm(a['point_p'][:, 1]-expected['summary']['target_original_m'], axis=1)
    error = np.linalg.norm(a['point_p'][:, 1]-expected['summary']['target_changed_m'], axis=1)
    timing = dict(brace_onset_s=onset(t, support), original_reach_onset_s=onset(t, support & (original_error < .03), stop=4.),
        physical_brace_release_onset_s=onset(t, both & (robot_load < 5), start=5.2),
        reach_onset_s=onset(t, support & (error < .03), start=4., stop=5.2),
        standing_recovery_onset_s=onset(t, recovery, start=5.2), supported_rest_onset_s=onset(t, abort, start=5.2))
    for key, value in timing.items():
        check(tag+' '+key, value, expected['timing'][key])
    last = t >= 8.7-1e-9
    check(tag+' final standing', int(recovery[last].all()), int(expected['final_standing']))
    check(tag+' final supported rest', int(abort[last].all()), int(expected['final_supported_rest']))
    check(tag+' complete cycle', int(timing['standing_recovery_onset_s'] is not None and timing['reach_onset_s'] is not None and recovery[last].all()), int(expected['complete_cycle']))
    check(tag+' displacement maximum', displacement[t >= 5.2].max(), expected['post_release_max_sole_displacement_mm'])
    acceleration = np.diff(v[:, va], axis=0)/np.diff(t)[:, None]
    for phase, low, high in [('approach', 0., 4.), ('command', 4., 5.2), ('recovery', 5.2, 9.), ('tail', 8.5, 9.)]:
        mask = (t >= low-1e-9) & (t < high-1e-9)
        pmask = (ts >= low-1e-9) & (ts < high-1e-9)
        e = expected['phases'][phase]
        values = dict(joint_speed_peak_rad_s=speed[mask].max(), wrist_speed_peak_m_s=wrist[mask].max(),
            joint_acceleration_peak_rad_s2=abs(acceleration[mask[:-1] & mask[1:]]).max(),
            joint_tracking_rms_rad=np.sqrt(np.mean((a['q_des'][mask]-q[mask][:, qa])**2)),
            target_error_rms_mm=1000*np.sqrt(np.mean(error[mask]**2)),
            solve_p95_ms=np.percentile(1000*a['pred_solve_s'][pmask], 95),
            policy_p95_ms=np.percentile(a['pred_policy_ms'][pmask], 95),
            policy_max_ms=a['pred_policy_ms'][pmask].max(), policy_over_20ms=np.sum(a['pred_policy_ms'][pmask] > 20),
            commands=pmask.sum(), table_load_min_N=robot_load[mask].min(), table_load_max_N=robot_load[mask].max())
        for key, value in values.items():
            check(tag+' '+phase+' '+key, value, e[key])
        for i, foot in enumerate(F.FEET):
            p = a['sole_p'][mask, i]
            loaded = a['sole_w'][mask, i, 2] > 5
            values = dict(displacement_from_phase_start_max_mm=1000*np.linalg.norm(p-p[0], axis=1).max(),
                contact_loss_s=(~loaded).sum()*.002, loaded_slip_max_m_s=slip[mask, i][loaded].max(),
                normal_min_N=a['sole_w'][mask, i, 2].min(), normal_max_N=a['sole_w'][mask, i, 2].max())
            for key, value in values.items():
                check(tag+' '+phase+' '+foot+' '+key, value, e['feet'][foot][key])
    measured[tag] = dict(evaluated_commands=len(ts), physical_samples=len(t),
        initial_torque_error_Nm=float(abs(torques[valid]-actual[indices[valid]]).max()),
        servo_error_Nm=float(abs(servo-actual).max()), wrench_error=wrench_error,
        cone_error=row_error, maximum_executable_cone_excess=cone_excess,
        final_robot_table_load_N=float(robot_load[-1]), final_table_body_loads_N=table[-1]['body_vertical_force_N'],
        release_policy_ms=float(a['pred_policy_ms'][np.argmin(abs(ts-5.2))]),
        recovery_criteria_failed_in_final_dwell=dict(foot_support=int((~both[last]).sum()),
            quiet=int((~quiet[last]).sum()), sole_motion=int((~still[last]).sum()),
            standing_posture=int((~upright[last]).sum()), table_release=int((robot_load[last] >= 5).sum())))

    path = MEDIA/('20261018-recovery-'+tag+'.mp4')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == meta['video_sha256'] and sha(meta['source']) == meta['source_sha256']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
        'stream=avg_frame_rate,nb_frames,duration', '-of', 'json', str(path)]))['streams'][0]
    check(tag+' video frames', int(probe['nb_frames']), meta['frames'])
    check(tag+' video duration', float(probe['duration']), meta['duration_s'], 1e-6)
    assert probe['avg_frame_rate'] == '30/1' and meta['playback_speed'] == 1
    subprocess.run(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path), '-threads', '1', '-f', 'null', '-'], check=True)
    videos[tag] = dict(**probe, full_decode_passed=True, timestamp_error_s=meta['max_timestamp_error_s'])

for case in ['nominal', 'pulse']:
    a, b = arrays['baseline-'+case], arrays['guarded-'+case]
    assert np.array_equal(a['pred_x_est'], b['pred_x_est'])
    for key in ['sole_p', 'sole_v', 'sole_w', 'point_p', 'point_v', 'q_des', 'v_des', 'tau_ff']:
        assert np.array_equal(a[key], b[key]), (case, key)
    check(case+' guard release accepted', int(report['cases']['guarded-'+case]['summary']['transition']['release_accepted']), 1)
for group in ['baseline', 'guarded']:
    a, b = arrays[group+'-nominal'], arrays[group+'-pulse']
    mask = a['pred_t'] < 4.5-1e-9
    assert np.array_equal(a['pred_x_est'][mask], b['pred_x_est'][mask])
for path, digest in report['source_hashes'].items():
    assert sha(ROOT/path) == digest, path
archive = MEDIA/'20261018-command-recovery-raw.zip'
manifest = json.loads((MEDIA/'20261018-command-recovery-archive.json').read_text())
assert sha(archive) == manifest['archive_sha256']
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist()) == set(manifest['members'])
    for name, digest in manifest['members'].items():
        assert hashlib.sha256(z.read(name)).hexdigest() == sha(ROOT/name) == digest, name
assert report['decision']['hypothesis_passed'] is False
assert report['decision']['guard_activations'] == 0
assert report['decision']['complete_cycles'] == 2

result = dict(checkpoint='command_recovery_transfer', review_date='2026-10-19 America/Denver',
    passed=True, new_rollouts=0, cumulative_rollouts=4, rollout_retries=0, rollout_limit=6,
    physics_steps=0, optimizer_solves=0, checks=checks, measurements=measured, videos=videos,
    source_hashes_match=True, model_hashes_match=True,
    raw_archive=dict(members=len(manifest['members']), all_members_match=True, sha256=sha(archive)),
    source_review=dict(observation='Estimated base pose through default_sense(seed=17); physical contacts used only in scoring.',
        release='At 5.2 s, evaluate one feet-only solve; accept projected status; issue a second feet-only solve. These are separate commands, and both solve costs contribute to transition latency.',
        abort='Rejection branch sets the current estimated state as rest reference and adds the existing weight-100 velocity cost. No rejection occurred; no abort validation is claimed.',
        schedule='Both feet remain fixed throughout; feet-only return excludes subsequent gripper contacts. Original site membership and certified defaults unchanged.',
        shutdown='Final command at 9.02 s is replaced by shutdown without plant integration; scored force phases end before 9 s.'),
    visual_review=dict(frame_sequences_inspected=list(videos), frames_per_sequence=8,
        nominal='Standing return visible at 6.5, 8 and 9 s; hands separate from the tabletop.',
        pulse='Torso remains inclined and hands near the table at 6.5, 8 and 9 s; body load logs establish renewed support.',
        limitation='Eight-frame sequences do not resolve millimetre foot slip; raw motion logs determine that measurement.'),
    decision='First-command release feasibility fails to predict successful disturbed recovery. Keep opt-in; no further rollout is required for this hypothesis.')
F.dump(MEDIA/'20261019-command-recovery-review.json', result)
print(json.dumps(dict(passed=True, checks=len(checks), evaluated_commands=sum(z['evaluated_commands'] for z in measured.values()),
    physical_samples=sum(z['physical_samples'] for z in measured.values()), archive_members=len(manifest['members']), new_rollouts=0)), flush=True)
