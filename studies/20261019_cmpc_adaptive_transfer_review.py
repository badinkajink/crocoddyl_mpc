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
spec = importlib.util.spec_from_file_location('experiment', ROOT/'studies/20261019_cmpc_adaptive_transfer.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
F, np, OUT, MEDIA = R.F, R.np, R.OUT, R.MEDIA
report = json.loads((MEDIA/'20261016-cmpc-adaptive-transfer.json').read_text())
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
    stem = ((ROOT/'studies/runs/20261018-cmpc-command-recovery')/('20261018-'+tag) if tag.startswith('baseline') else OUT/('20261019-'+tag))
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
        returning = expected['summary']['transition'] is not None and stamp >= expected['summary']['transition']['t']-1e-9
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
        release_policy_ms=float(a['pred_policy_ms'][np.argmin(abs(ts-expected['summary']['transition']['t']))]),
        final_joint_speed_rad_s=float(speed[-1]),final_wrist_speed_m_s=float(wrist[-1]),
        final_joint_pose_rms_rad=float(np.sqrt(np.mean((q[-1,qa]-stand[qa])**2))),
        recovery_criteria_failed_in_final_dwell=dict(foot_support=int((~both[last]).sum()),
            quiet=int((~quiet[last]).sum()), sole_motion=int((~still[last]).sum()),
            standing_posture=int((~upright[last]).sum()), table_release=int((robot_load[last] >= 5).sum())))

    path = MEDIA/expected['video']
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
    a,b=arrays['baseline-'+case],arrays['gated-'+case]
    mask=a['pred_t']<5.2-1e-9
    check(case+' common estimated trajectory',a['pred_x_est'][mask],b['pred_x_est'][mask],0.)
    mask=a['t']<5.2-1e-9
    for key in ['sole_p','sole_v','sole_w','point_p','point_v','q_des','v_des','tau_ff']:
        check(case+' common physical '+key,a[key][mask],b[key][mask],0.)
    motion=json.loads((OUT/('20261019-gated-'+case+'-motion-gate.json')).read_text())
    md=ocp.rmodel.createData();since=None;first=None;max_err=0.
    for saved,x,t in zip(motion,b['pred_x_est'],b['pred_t']):
        F.pin.computeJointJacobians(ocp.rmodel,md,x[:34]);F.pin.updateFramePlacements(ocp.rmodel,md)
        def speed(fid):
            return (F.pin.getFrameJacobian(ocp.rmodel,md,fid,F.pin.LOCAL_WORLD_ALIGNED)@x[34:])[:3]
        sole=[float(np.linalg.norm(speed(ocp.sites[f])[:2])) for f in F.FEET]
        wrist=float(np.linalg.norm(speed(ocp.rmodel.getFrameId('left_wrist_yaw_link'))))
        joint=float(abs(x[40:]).max())
        max_err=max(max_err,abs(wrist-saved['wrist_speed_m_s']),abs(joint-saved['joint_speed_rad_s']),max(abs(np.array(sole)-saved['sole_tangent_speed_m_s'])))
        quiet=joint<.1 and wrist<.03 and max(sole)<.02
        since=(float(t) if since is None else since) if quiet else None
        ready=since is not None and t-since>=.3-1e-9
        assert quiet==saved['quiet'] and ready==saved['ready']
        if first is None and ready and t>=5.2-1e-9:first=float(t)
    check(case+' independent estimated velocity',max_err,0.)
    check(case+' first motion-approved release',first,report['cases']['gated-'+case]['summary']['transition']['t'])
    at=motion[int(np.argmin(abs(b['pred_t']-5.2)))];measured['gated-'+case]['motion_at_request']=at
for path,digest in report['source_hashes'].items():assert sha(ROOT/path)==digest,path
archive=MEDIA/'20261018-command-recovery-raw.zip'
manifest=json.loads((MEDIA/'20261018-command-recovery-archive.json').read_text())
assert sha(archive)==manifest['archive_sha256']
with zipfile.ZipFile(archive) as z:
    for name,digest in manifest['members'].items():
        assert hashlib.sha256(z.read(name)).hexdigest()==sha(ROOT/name)==digest,name
assert report['decision']['hypothesis_passed'] is False
assert report['cases']['gated-pulse']['complete_cycle'] is False
result=dict(checkpoint='adaptive_transfer',date='2026-10-19',passed=True,
    new_review_rollouts=0,cumulative_rollouts=2,physics_steps=0,optimizer_solves=0,
    checks=checks,measurements=measured,videos=videos,baseline_archive_members_verified=len(manifest['members']),
    source_hashes_match=True,model_hashes_match=True,observation='Only archived estimated State enters action evaluations and gate reconstruction. True motion and forces are scoring only.',
    decision='Estimated low-motion dwell reduces disturbed foot motion and table recontact through 9 s but fails the predeclared recovery deadline; keep opt-in.')
F.dump(MEDIA/'20261019-adaptive-review.json',result)
print(json.dumps(dict(passed=True,checks=len(checks),measurements=measured)),flush=True)
