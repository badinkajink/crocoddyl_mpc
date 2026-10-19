#!/usr/bin/env python3
"""Independent saved-data checks for the uncertainty prerequisite.

Use resguard. Observed torque and physical traces are scoring-only. No plant
steps, parameter fitting, optimization or closed-loop controller runs occur.
"""
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment', ROOT/'studies/20261018_cmpc_command_prediction.py')
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
F, np, OUT, MEDIA = E.F, E.np, E.OUT, E.MEDIA

checks = []


def check(name, actual, expected, tol=1e-9):
    err = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
    passed = bool(np.allclose(actual, expected, rtol=1e-9, atol=tol))
    checks.append(dict(name=name, passed=passed, max_error=err))
    assert passed, (name, err)


fl = F.build()
state = fl.mpc.ocp.state
results = {}
for row, case in enumerate(['nominal', 'pulse']):
    report = json.loads((OUT/f'20261018-{case}-prediction.json').read_text())
    data = np.load(OUT/f'20261018-{case}-prediction.npz')
    original = np.load(E.G.OUT/f'20261017-corrected-{case}-raw.npz')
    commands = json.loads((E.G.OUT/f'20261017-corrected-{case}-commands.json').read_text())
    ix = data['indices']
    check(case+' source estimates', data['input_estimates'], original['pred_x_est'][ix])
    check(case+' source targets', data['targets'], original['pred_x_est'][ix+1])
    check(case+' original predictor parity', data['frozen_20ms_next'], original['pred_model_next'][ix])
    check(case+' observation interval', np.diff(original['pred_t']), .02)
    if case == 'pulse':
        assert min(data['time']) >= 4.5-1e-9
        checks.append(dict(name='Reserved evaluation excludes shared pre-pulse transitions', passed=True))
    for v in E.VARIANTS:
        residual = np.array([state.diff(xp, xn) for xp, xn in zip(data[v+'_next'], data['targets'])])
        check(case+' '+v+' state residual', residual, data[v+'_residual'])
        rms = np.sqrt(np.mean(np.sum(residual[:, 36:]**2, axis=1)))
        check(case+' '+v+' velocity metric', rms, report['variants'][v]['angular_joint_velocity_rms_rad_s'])
    rows = json.loads((OUT/f'20261018-{case}-endpoint-rows.json').read_text())
    stamp_excess = {}
    for r in rows:
        stamp_excess[r['t']] = max(stamp_excess.get(r['t'], 0.), max(r['cone_violation'][:5]))
    endpoint_violating = {r['t'] for r in rows if max(r['cone_violation']) > E.G.TOL}
    # Independent issued-law validation against every physical 2 ms sample.
    # These saved physical states are scoring only and never enter prediction.
    source = E.G.OUT/f'20261017-corrected-{case}.csv'
    with source.open() as f:
        reader = csv.reader(line for line in f if not line.startswith('#'))
        cols = next(reader)
        physical = np.array(list(reader), float)
    t = physical[:, cols.index('time')]
    def cols_array(prefix, ids):
        return np.array([physical[:, cols.index(prefix+str(i))] for i in ids]).T
    q = cols_array('qpos', range(7, 34))
    v = cols_array('qvel', range(6, 33))
    actual = cols_array('afrc', range(27))
    lower, upper = np.array(report['nominal_command_interface']['torque_lower_Nm']), np.array(report['nominal_command_interface']['torque_upper_Nm'])
    error, variation, count = 0., [], 0
    for j in ix:
        stamp = original['pred_t'][j]
        mask = (t >= stamp-1e-9) & (t < stamp+.02-1e-9)
        c = commands[j]
        calculated = np.clip(fl.kp*(np.array(c['q'])-q[mask])+fl.kd*(np.array(c['v'])-v[mask])+c['tau'], lower, upper)
        error = max(error, float(abs(calculated-actual[mask]).max()))
        variation.append(float(abs(actual[mask]-actual[mask][0]).max()))
        count += int(mask.sum())
    check(case+' high-rate physical actuator law', error, 0., 1e-9)
    metrics = report['variants']
    frozen = metrics['frozen_20ms']['angular_joint_velocity_rms_rad_s']
    fine = metrics['frozen_2ms']['angular_joint_velocity_rms_rad_s']
    servo = metrics['servo_2ms']['angular_joint_velocity_rms_rad_s']
    videos = {}
    for suffix in ['mp4', 'csv']:
        path = MEDIA/f'20261017-foot-model-corrected-{case}.mp4' if suffix == 'mp4' else source
        meta = json.loads((MEDIA/f'20261017-foot-model-corrected-{case}.json').read_text())
        digest = F.sha(path)
        assert digest == meta['video_sha256' if suffix == 'mp4' else 'source_sha256']
        videos[suffix+'_sha256'] = digest
    video = MEDIA/f'20261017-foot-model-corrected-{case}.mp4'
    probe = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,duration,avg_frame_rate','-of','json',str(video)]))['streams'][0]
    check(case+' video frame count', int(probe['nb_frames']), meta['frames'])
    assert probe['avg_frame_rate'] == '30/1' and meta['playback_speed'] == 1
    checks.append(dict(name=case+' video and physical source hashes', passed=True))
    results[case] = dict(servo_improvement_over_fine_fraction=1-servo/fine,
        fine_improvement_over_original_fraction=1-fine/frozen,
        servo_improvement_over_original_fraction=1-servo/frozen,
        hypothesis_passed=bool(servo <= .9*fine),
        endpoint_violating_transitions=len(endpoint_violating), transitions=len(ix),
        physical_actuator_scoring_samples=count, physical_law_max_error_Nm=error,
        physical_within_interval_torque_change_max_Nm=max(variation), videos=videos)
review = dict(passed=True, checks=checks, cases=results,
    hypothesis_passed=all(r['hypothesis_passed'] for r in results.values()),
    new_rollouts=0, plant_steps=0, source_sha256=F.sha(Path(__file__)))
F.dump(MEDIA/'20261018-model-uncertainty-review.json', review)
print(json.dumps(review), flush=True)
