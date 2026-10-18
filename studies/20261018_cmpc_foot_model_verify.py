#!/usr/bin/env python3
"""Review foot-model-transfer commands and predictions without plant integration.

Use resguard. The six-rollout budget is exhausted. Rebuild the archived action
models and evaluate saved estimated states with torques reconstructed from issued
commands. Do not invoke the MPC solver, rollout runner, or original scorer.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT/'studies/20261017_cmpc_foot_model_transfer.py')
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
F, np, OUT, MEDIA = G.F, G.np, G.OUT, G.MEDIA
report = json.loads((MEDIA/'20261016-cmpc-foot-model-transfer.json').read_text())
checks, measured, arrays = [], {}, {}


def check(name, actual, expected, atol=1e-8):
    error = float(np.max(abs(np.asarray(actual)-np.asarray(expected))))
    ok = bool(np.allclose(actual, expected, rtol=1e-10, atol=atol))
    checks.append(dict(name=name, passed=ok, max_abs_error=error))
    assert ok, (name, actual, expected)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


fl = F.build()
m, ocp = fl.m, fl.mpc.ocp
rd = ocp.rmodel.createData()
cache = {}
provenance = F.provenance(fl)
for tag, expected in report['cases'].items():
    for key in ['plan_sha256', 'xs_sha256', 'us_sha256', 'model_sha256', 'model_arrays_sha256']:
        assert provenance[key] == expected['summary']['provenance'][key], (tag, key)
    stem = OUT/('20261017-'+tag)
    a = dict(np.load(str(stem)+'-raw.npz'))
    arrays[tag] = a
    with Path(str(stem)+'.csv').open() as f:
        reader = csv.reader(line for line in f if not line.startswith('#'))
        columns = next(reader)
        raw = np.array(list(reader), float)
    def col(prefix, count):
        return np.array([raw[:, columns.index(prefix+str(i))] for i in range(count)]).T
    q, v, actual = col('qpos', m.nq), col('qvel', m.nv), col('afrc', m.nu)
    qa = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    va = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    commands = json.loads(Path(str(stem)+'-commands.json').read_text())
    force_suffix = '-executable-recomputed.json' if tag.startswith('baseline') else '-executable-after.json'
    stored_forces = json.loads(Path(str(stem)+force_suffix).read_text())
    force_map = {(r['t'], r['contact']): r for r in stored_forces}
    gates = json.loads(Path(str(stem)+'-gate.json').read_text())
    next_states, next_soles, torques = [], [], []
    wrench_error = row_error = cone_excess = 0.
    check(tag+' horizon', [w['T'] for w in expected['summary']['window_checks']], 35)
    check(tag+' control period', np.diff(a['pred_t']), .02)
    for j, (stamp, x, k) in enumerate(zip(a['pred_t'], a['pred_x_est'], a['pred_k'])):
        cmd = commands[j]
        check_time = abs(cmd['t']-stamp)
        assert check_time < 1e-10
        ctrl = np.array(cmd['q'])+(fl.kd*np.array(cmd['v'])+np.array(cmd['tau']))/fl.kp
        effective = np.where(m.actuator_ctrllimited, np.clip(ctrl, *m.actuator_ctrlrange.T), ctrl)
        tau = fl.kp*(effective-x[7:34])-fl.kd*x[40:]
        tau = np.where(m.actuator_forcelimited, np.clip(tau, *m.actuator_forcerange.T), tau)
        torques.append(tau)
        if k not in cache:
            cache[k] = fl.mpc.models[int(k)].createData()
        am, dat = fl.mpc.models[int(k)], cache[k]
        am.calc(dat, x, tau)
        next_states.append(dat.xnext.copy())
        next_soles.append(F.poses(ocp, dat.xnext, rd)[0])
        costs = am.differential.costs.costs.todict()
        for name, cd in dat.differential.multibody.contacts.contacts.todict().items():
            cone = costs['cone_'+name.removeprefix('brace_')].cost.residual.reference
            w = np.asarray(cd.f.vector)
            values = cone.A@w[:cone.A.shape[1]]
            excess = np.maximum(np.maximum(cone.lb-values, values-cone.ub), 0.)
            saved = force_map[(float(stamp), name)]
            wrench_error = max(wrench_error, float(abs(w-saved['wrench_world']).max()))
            row_error = max(row_error, float(abs(excess-saved['cone_violation']).max()))
            if stamp >= 4.-1e-9:
                cone_excess = max(cone_excess, float(excess.max()))
        if tag.startswith('corrected') and stamp >= 4.-1e-9:
            assert gates[j]['status'] == 'projected' and 'fallback' not in gates[j]
            assert np.all(abs(np.array(cmd['tau'])) <= .9*fl.tau_lim+1e-8)
            assert np.all(abs(effective-ctrl) <= 1e-8)
            assert np.all(np.array(cmd['q']) >= m.jnt_range[m.actuator_trnid[:, 0], 0]-1e-8)
            assert np.all(np.array(cmd['q']) <= m.jnt_range[m.actuator_trnid[:, 0], 1]+1e-8)
    next_states, next_soles, torques = map(np.asarray, (next_states, next_soles, torques))
    if tag.startswith('baseline'):
        original = np.load(str(stem)+'-recomputed-model.npz')
        saved_next, saved_tau = original['xnext'], original['torque']
    else:
        saved_next, saved_tau = a['pred_model_next'], a['pred_total_torque']
        check(tag+' accepted hold cone rows', cone_excess, 0., G.TOL)
    check(tag+' next-state reconstruction', next_states, saved_next)
    check(tag+' torque reconstruction', torques, saved_tau)
    check(tag+' wrench reconstruction', wrench_error, 0.)
    check(tag+' cone reconstruction', row_error, 0.)
    indices = np.searchsorted(a['t'], a['pred_t'])
    valid = indices < len(a['t'])
    check(tag+' initial physical torque', torques[valid], actual[indices[valid]])
    check(tag+' observed joint positions', a['pred_x_est'][valid, 7:34], q[indices[valid]][:, qa])
    check(tag+' observed joint velocities', a['pred_x_est'][valid, 40:], v[indices[valid]][:, va])
    check(tag+' final reach', 1000*np.linalg.norm(a['point_p'][-1, 1]-fl.plan['target']), expected['summary']['reach_final_mm'])
    check(tag+' solve plus filter p95', np.percentile(1000*a['pred_solve_s']+a['pred_gate_ms'], 95), expected['filter']['solver_plus_filter_p95_ms'])
    contacts = json.loads(Path(str(stem)+'-reconstructed-contacts.json').read_text())
    state_delta = np.array([ocp.state.diff(xp, xn) for xp, xn in zip(next_states[:-1], a['pred_x_est'][1:])])
    for phase, lo in [('hold', 4.), ('post_pulse', 4.5)]:
        mask = a['t'] >= lo-1e-9
        pmask = a['pred_t'][:-1] >= lo-1e-9
        r = expected['phases'][phase]
        rr = state_delta[pmask]
        for label, sl, unit in [('base_position_rms_mm', slice(0, 3), 1000),
                              ('orientation_joint_rms_rad', slice(3, 33), 1),
                              ('base_linear_velocity_rms_m_s', slice(33, 36), 1),
                              ('angular_joint_velocity_rms_rad_s', slice(36, 66), 1)]:
            check(tag+' '+phase+' '+label, unit*np.sqrt(np.mean(np.sum(rr[:, sl]**2, axis=1))), r['state_model_residual'][label])
        for i, foot in enumerate(F.FEET):
            pp = a['sole_p'][mask, i]
            check(tag+' '+phase+' '+foot+' displacement', 1000*np.linalg.norm(pp-pp[0], axis=1).max(), r[foot]['displacement_max_from_phase_start_mm'])
            check(tag+' '+phase+' '+foot+' contact loss', .002*np.sum(a['loaded_contacts'][mask, i] == 0), r[foot]['contact_loss_s'])
            error = a['pred_sole_p0'][1:, i]-next_soles[:-1, i]
            check(tag+' '+phase+' '+foot+' model residual', 1000*np.sqrt(np.mean(np.sum(error[pmask]**2, axis=1))), r[foot]['executable_model_estimated_position_rms_mm'])
            delta = next_soles[:-1, i]-a['pred_sole_p0'][:-1, i]
            check(tag+' '+phase+' '+foot+' predicted motion', 1000*np.linalg.norm(delta[pmask], axis=1).max(), r[foot]['predicted_step_motion_max_mm'])
            # Exclude the last transition, whose saved physical endpoint is
            # 6.018 s. It spans 18 ms, although the estimated state reaches 6.02 s.
            ii, jj = indices[:-1], indices[1:]
            full = pmask & (jj < len(a['t']))
            motion = 1000*np.linalg.norm(a['sole_p'][jj[full], i]-a['sole_p'][ii[full], i], axis=1).max()
            check(tag+' '+phase+' '+foot+' complete simulated 20 ms motion', motion, r[foot]['simulated_step_motion_max_mm'])
            rows = [c for c in contacts if c['body'] == foot and c['t'] >= lo-1e-9]
            check(tag+' '+phase+' '+foot+' slip', max(c['material_point_tangential_speed_m_s'] for c in rows if c['normal_N'] > 5), r['simulated_contacts'][foot]['loaded_material_point_slip_peak_m_s'])
            check(tag+' '+phase+' '+foot+' native cone', max(c['native_cone_excess_N'] for c in rows), r['simulated_contacts'][foot]['native_cone_excess_max_N'])
        force_key = 'executable_force_after_hold' if phase == 'hold' else 'executable_force_after_post_pulse'
        for name, force_expected in expected[force_key].items():
            rows = [c for c in stored_forces if c['contact'] == name and c['t'] >= lo-1e-9]
            w = np.array([c['wrench_world'] for c in rows])
            cv = np.array([c['cone_violation'] for c in rows])
            values = dict(normal_min_N=w[:, 2].min(), normal_max_N=w[:, 2].max(),
                          tangent_max_N=np.linalg.norm(w[:, :2], axis=1).max(),
                          force_rows_max_N=cv[:, :5].max(),
                          violating_samples=int(np.sum(cv.max(axis=1) > G.TOL)))
            if name in F.FEET:
                w = w[w[:, 2] > 5]
                cop = np.c_[-w[:, 4]/w[:, 2], w[:, 3]/w[:, 2]]
                values.update(moment_rows_max_Nm=cv[:, 5:].max(), cop_margin_min_mm=1000*np.min(np.array([.1, .04])-abs(cop)))
            for key, value in values.items():
                check(tag+' '+phase+' '+name+' '+key, value, force_expected[key])
        # These motion measures intentionally reproduce the saved strict phase
        # mask; the physical-contact measures above tolerate timestamp rounding.
        motion_mask = a['t'] >= lo
        acceleration = np.diff(v[:, va], axis=0)/np.diff(a['t'])[:, None]
        amask = motion_mask[:-1] & motion_mask[1:]
        sr = expected['summary']['motion'][phase]
        check(tag+' '+phase+' speed', abs(v[motion_mask][:, va]).max(), sr['joint_speed_rad_s'])
        check(tag+' '+phase+' acceleration', abs(acceleration[amask]).max(), sr['joint_acceleration_rad_s2'])
        check(tag+' '+phase+' tracking', np.sqrt(np.mean((a['q_des'][motion_mask]-q[motion_mask][:, qa])**2)), sr['q_tracking_rms_rad'])
    active = np.array([g['active'] for g in gates])
    measured[tag] = dict(evaluated_states=len(next_states), observed_initial_torques=int(valid.sum()),
        projected_hold_commands=int(active.sum()), projected_commands_followed_by_integration=int(np.sum(active & valid)),
        wrench_reconstruction_max_error=wrench_error, cone_reconstruction_max_error=row_error,
        hold_cone_excess_max=cone_excess, final_command_time_s=float(a['pred_t'][-1]), last_physical_log_s=float(a['t'][-1]))

for case in ['nominal', 'pulse']:
    old = np.load(ROOT/f'studies/runs/20261016-cmpc-foot-motion/20261016-{case}-raw.npz')
    assert np.array_equal(old['pred_x_est'], arrays['baseline-'+case]['pred_x_est'])
    pre = arrays['baseline-'+case]['pred_t'] < 4.-1e-9
    assert np.array_equal(arrays['baseline-'+case]['pred_x_est'][pre], arrays['corrected-'+case]['pred_x_est'][pre])
for group in ['baseline', 'corrected']:
    pre = arrays[group+'-nominal']['pred_t'] < 4.5-1e-9
    assert np.array_equal(arrays[group+'-nominal']['pred_x_est'][pre], arrays[group+'-pulse']['pred_x_est'][pre])

for path, digest in report['source_hashes'].items():
    assert sha(ROOT/path) == digest, path
archive = MEDIA/'20261017-foot-model-raw.zip'
manifest = json.loads((MEDIA/'20261017-foot-model-archive.json').read_text())
assert sha(archive) == manifest['archive_sha256']
with zipfile.ZipFile(archive) as z:
    for name in z.namelist():
        assert hashlib.sha256(z.read(name)).hexdigest() == sha(OUT/name), name
    members = len(z.namelist())
assert members == manifest['member_count']
videos = {}
for tag in report['cases']:
    path = MEDIA/('20261017-foot-model-'+tag+'.mp4')
    meta = json.loads(path.with_suffix('.json').read_text())
    assert sha(path) == meta['video_sha256'] and sha(meta['source']) == meta['source_sha256']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate,nb_frames,duration', '-of', 'json', str(path)]))['streams'][0]
    check(tag+' video frames', int(probe['nb_frames']), meta['frames'])
    check(tag+' video duration', float(probe['duration']), meta['duration_s'], 1e-6)
    assert probe['avg_frame_rate'] == '30/1' and meta['playback_speed'] == 1
    videos[tag] = probe

cases = report['cases']
ratio = cases['corrected-pulse']['phases']['post_pulse']['sole_left']['executable_model_estimated_position_rms_mm']/cases['baseline-pulse']['phases']['post_pulse']['sole_left']['executable_model_estimated_position_rms_mm']
reach_delta = cases['corrected-nominal']['summary']['reach_final_mm']-cases['baseline-nominal']['summary']['reach_final_mm']
loss_delta = max(cases['corrected-nominal']['phases']['hold'][f]['contact_loss_s']-cases['baseline-nominal']['phases']['hold'][f]['contact_loss_s'] for f in F.FEET)
check('decision residual ratio', ratio, report['decision']['pulse_left_residual_ratio'])
check('decision reach delta', reach_delta, report['decision']['nominal_final_reach_increase_mm'])
check('decision loss delta', loss_delta, report['decision']['nominal_contact_loss_increase_s'])
assert ratio <= .9 and reach_delta <= 5 and loss_delta <= 0
assert report['decision']['hypothesis_passed'] and not report['decision']['adopt_default']

result = dict(checkpoint='foot_model_transfer', review_date='2026-10-18 America/Denver', passed=True,
    new_rollouts=0, cumulative_rollouts=6, rollout_retries=2, rollout_limit=6, physics_steps=0,
    checks=checks, measurements=measured, model_and_source_hashes_match=True,
    archive=dict(sha256=sha(archive), members=members, all_members_match=True), videos=videos,
    final_command_accounting='Each corrected trial evaluates 102 hold commands; 101 precede plant integration. The last command at 6.02 s is immediately replaced by shutdown safe_hold. All 204 model checks pass; 202 commands have physical rollout exposure. Complete 20 ms physical-motion maxima are unchanged after excluding the final 18 ms logged transition.',
    limitations='Current-node force feasibility only. No frictional sliding prediction, fallback recovery validation, deadline delay injection or hardware measurement.')
F.dump(MEDIA/'20261018-foot-model-review.json', result)
print(json.dumps(dict(passed=True, numerical_checks=len(checks), evaluated_states=sum(r['evaluated_states'] for r in measured.values()), archive_members=members, new_rollouts=0)), flush=True)
