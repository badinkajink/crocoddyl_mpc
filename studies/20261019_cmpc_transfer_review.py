#!/usr/bin/env python3
"""Audit saved transfer comparisons; no model loading, solves or plant steps.

Run under resguard. Recompute comparison deltas, verify identical pre-intervention
arrays and recorded model provenance, and hash the artifacts used by the review.
The result does not turn repeated baseline data into independent trials.
"""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/lean/media/20261015-contact-audit'
OUT = MEDIA / '20261019-transfer-review-measurement.json'
checks = []
inputs = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def record(path):
    path = Path(path)
    inputs[str(path.relative_to(ROOT))] = sha(path)


def read(name):
    path = MEDIA / name
    record(path)
    return json.loads(path.read_text())


def check(name, passed, **detail):
    checks.append(dict(name=name, passed=bool(passed), **detail))


def arrays(folder, tag):
    path = ROOT / 'studies/runs' / folder / (tag + '-raw.npz')
    record(path)
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def same_before(a, b, stop, label):
    for k in ['sole_p', 'sole_R', 'sole_v', 'point_p', 'point_v',
              'pred_x_est']:
        clock = 'pred_t' if k.startswith('pred_') else 't'
        ma, mb = a[clock] < stop - 1e-8, b[clock] < stop - 1e-8
        av, bv = a[k][ma], b[k][mb]
        ok = av.shape == bv.shape and np.array_equal(av, bv)
        err = float(np.max(np.abs(av-bv))) if av.shape == bv.shape else None
        check(label + ' ' + k, ok, samples=len(av), maximum_absolute_error=err)


foot = read('20261016-cmpc-foot-model-transfer.json')
recovery = read('20261016-cmpc-command-recovery.json')
adaptive = read('20261016-cmpc-adaptive-transfer.json')
uncertainty = read('20261016-cmpc-model-uncertainty.json')
for name in ['20261015-audit.json', '20261015-surface-catalogue.json',
             '20261015-foot-motion.json', '20261015-state-frames.json',
             '20261016-cmpc-motion-transfer.json',
             '20261016-cmpc-brace-selection-transfer.json']:
    read(name)

comparison = {}
provenance_keys = ['plan_sha256', 'xs_sha256', 'us_sha256', 'model_sha256',
                   'model_arrays_sha256']
for condition in ['nominal', 'pulse']:
    b = foot['cases']['baseline-' + condition]
    c = foot['cases']['corrected-' + condition]
    for k in provenance_keys:
        check('force ' + condition + ' ' + k,
              b['summary']['provenance'][k] == c['summary']['provenance'][k])
    same_before(arrays('20261017-cmpc-foot-model-transfer', '20261017-baseline-'+condition),
                arrays('20261017-cmpc-foot-model-transfer', '20261017-corrected-'+condition),
                4.0, 'force '+condition+' before 4 s')
    rows = {}
    for label, case in [('baseline', b), ('corrected', c)]:
        phase = case['phases']['post_pulse']
        sole = phase['sole_left']
        motion = case['summary']['motion']['post_pulse']
        rows[label] = dict(
            left_sole_displacement_mm=sole['displacement_max_from_phase_start_mm'],
            left_contact_loss_ms=1000*sole['contact_loss_s'],
            left_loaded_slip_m_s=phase['simulated_contacts']['sole_left']['loaded_material_point_slip_peak_m_s'],
            left_executable_prediction_rms_mm=sole['executable_model_estimated_position_rms_mm'],
            reach_final_mm=case['summary']['reach_final_mm'],
            joint_speed_rad_s=motion['joint_speed_rad_s'],
            joint_acceleration_rad_s2=motion['joint_acceleration_rad_s2'],
            wrist_speed_m_s=motion['points']['brace_wrist']['speed_m_s'],
            joint_tracking_rms_mrad=1000*motion['q_tracking_rms_rad'],
            solve_plus_audit_or_filter_p95_ms=case['filter']['solver_plus_filter_p95_ms'],
            above_20ms=case['filter']['solver_plus_filter_above_20ms'],
            command_evaluations=case['summary']['periods'],
            executable_hold_force_rows=case['executable_force_after_hold'])
    rows['left_displacement_reduction_percent'] = 100*(1-rows['corrected']['left_sole_displacement_mm']/rows['baseline']['left_sole_displacement_mm'])
    rows['reach_change_mm'] = rows['corrected']['reach_final_mm']-rows['baseline']['reach_final_mm']
    comparison[condition] = rows

recovery_pairs = {}
for condition in ['nominal', 'pulse']:
    base = arrays('20261018-cmpc-command-recovery', '20261018-baseline-'+condition)
    guarded = arrays('20261018-cmpc-command-recovery', '20261018-guarded-'+condition)
    gated = arrays('20261019-cmpc-adaptive-transfer', '20261019-gated-'+condition)
    same_before(base, guarded, 10, 'first-command guard '+condition+' entire run')
    same_before(base, gated, 5.2, 'motion gate '+condition+' before request')
    for k in provenance_keys:
        check('motion gate '+condition+' '+k,
              recovery['cases']['baseline-'+condition]['summary']['provenance'][k]
              == adaptive['cases']['gated-'+condition]['summary']['provenance'][k])
    check('reused baseline '+condition,
          recovery['cases']['baseline-'+condition] == adaptive['cases']['baseline-'+condition])
    recovery_pairs[condition] = {}
    for label, report, tag in [('scheduled', recovery, 'baseline-'), ('gated', adaptive, 'gated-')]:
        c = report['cases'][tag+condition]
        recovery_pairs[condition][label] = dict(
            complete_cycle=c['complete_cycle'], timing=c['timing'],
            post_request_max_sole_displacement_mm=c['post_release_max_sole_displacement_mm'])

# Verify the stored archive digests without loading meshes or numerical models.
archive_checks = []
for path in sorted(MEDIA.glob('*archive.json')):
    manifest = json.loads(path.read_text())
    digest = manifest.get('archive_sha256')
    candidates = [path.with_name(path.name.replace('-archive.json', '-raw.zip'))]
    for candidate in candidates:
        if digest and candidate.exists():
            actual = sha(candidate)
            check(candidate.name+' archive digest', actual == digest)
            inputs[str(candidate.relative_to(ROOT))] = actual
            record(path)
            archive_checks.append(candidate.name)

# Pages and videos provide durable links to all nine completed checkpoints.
for pattern in ['*cmpc*.html']:
    for path in sorted((ROOT/'docs/lean').glob(pattern)):
        if path.name.startswith(('20261015-', '20261016-')) and 'transfer-assessment' not in path.name:
            record(path)
for path in sorted(MEDIA.glob('*.mp4')):
    record(path)
for path in sorted(MEDIA.glob('*run-record.json')):
    if 'transfer-review' not in path.name:
        record(path)
record(Path(__file__).resolve())
prov = foot['cases']['baseline-nominal']['summary']['provenance']
for filekey, hashkey in [('plan', 'plan_sha256'), ('model', 'model_sha256')]:
    path = Path(prov[filekey])
    check(filekey+' current bytes', path.exists() and sha(path) == prov[hashkey])

result = dict(
    checkpoint='transfer_review', date='2026-10-19',
    baseline_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
    measurement='Saved-array comparability and provenance audit; derived within-condition differences',
    passed=all(c['passed'] for c in checks), checks=checks,
    new_rollouts=0, cumulative_checkpoint_rollouts=0, allowed_regression_rollouts=2,
    plant_steps=0, optimizer_solves=0,
    force_comparison=comparison, recovery_comparison=recovery_pairs,
    command_interval_checks={k: dict(transitions=c['transitions'], initial=c['initial_force_rows'],
                                   endpoint=c['endpoint_force_rows']) for k,c in uncertainty['cases'].items()},
    source_plan_and_model=prov, archive_digest_checks=archive_checks,
    input_sha256=inputs,
    rollout_accounting=dict(geometry=0, foot_motion=2, state_frames=0, motion_transfer=3,
        foot_model_transfer=6, brace_selection_transfer=5, model_uncertainty_transfer=0,
        command_recovery_transfer=4, adaptive_transfer=2, transfer_review=0),
    rollout_accounting_total=22, rollout_retries_included=2,
    incompatible_comparisons=[
        'Physical brace selection changes wrist collision geometry and replans contact points; do not pool with original-model force-filter trials.',
        'Recovery changes target and contact schedule and lasts 9.02 s; do not pool with 6.02 s hold-only trials.',
        'Older August results use older assets; historical context only.',
        'Model-uncertainty predictor alternatives reuse saved transitions and are not closed-loop controller trials.'])
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(passed=result['passed'], checks=len(checks), inputs=len(inputs),
                     archives=len(archive_checks), rollouts=0, output=str(OUT))))
if not result['passed']:
    print(json.dumps([c for c in checks if not c['passed']], indent=2))
    raise SystemExit(1)
