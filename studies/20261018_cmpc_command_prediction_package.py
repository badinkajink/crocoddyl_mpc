#!/usr/bin/env python3
"""Package precomputed measurements, logs and metadata; no numerical job."""
import hashlib
import json
import os
import shlex
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'studies/runs/20261018-cmpc-model-uncertainty'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, obj):
    with p.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


record_path = MEDIA/'20261018-model-uncertainty-run-record.json'
record = read(record_path)
record['quota_reads'] = {p.name: read(p) for p in sorted(OUT.glob('*quota*.json'))}
for job in record['commands']:
    parts = shlex.split(job['command'])
    time_path = ROOT/parts[parts.index('-o')+1]
    log_path = ROOT/parts[parts.index('>')+1]
    job['time_output'] = time_path.read_text()
    job['log_output'] = log_path.read_text()
write(record_path, record)
review = read(MEDIA/'20261018-model-uncertainty-review.json')
assert review['passed'] and not review['hypothesis_passed']
previous = read(MEDIA/'20261016-cmpc-foot-model-transfer.json')
scripts = sorted((ROOT/'studies').glob('20261018_cmpc_command_prediction*.py'))
sources = {str(p.relative_to(ROOT)): sha(p) for p in scripts}
result = dict(
    checkpoint='model_uncertainty_transfer', status='checkpoint_complete',
    date='2026-10-18 America/Denver', baseline_commit='ba37518',
    completion_basis='Requested prerequisite branch: tested known command/integration mismatch and force feasibility between commands. Physical-parameter adaptation comparison was not performed.',
    hypothesis=read(MEDIA/'20261018-model-uncertainty-plan.json'),
    selection=read(MEDIA/'20261018-model-uncertainty-selection.json'),
    decision=dict(hypothesis_passed=False, negative_result=True, adopt_default=False,
        nominal_required_fraction=.1, reserved_required_fraction=.1,
        nominal_improvement_fraction=review['cases']['nominal']['servo_improvement_over_fine_fraction'],
        reserved_improvement_fraction=review['cases']['pulse']['servo_improvement_over_fine_fraction'],
        conclusion='The known servo law explains part of the residual but fails the nominal improvement criterion. Finer integration has the larger nominal effect. Initial feasible commands lose model feasibility at observed endpoints. No physical uncertainty range is identified.',
        physical_parameter_fitting='Not performed: command/integration and contact-model prerequisites invalidate attribution to a physical mismatch.',
        adapted_closed_loop_comparison='Not performed; all predictors score the same archived estimated states and commands.'),
    cases={case:read(OUT/f'20261018-{case}-prediction.json') for case in ['nominal','pulse']},
    inherited_task_performance={case:dict(
        source='docs/lean/media/20261015-contact-audit/20261016-cmpc-foot-model-transfer.json',
        source_case='corrected-'+case, summary=previous['cases']['corrected-'+case]['summary'],
        post_pulse=previous['cases']['corrected-'+case]['phases']['post_pulse'],
        runtime=previous['cases']['corrected-'+case]['filter'],
        new_rollout=False, same_for_all_predictors=True) for case in ['nominal','pulse']},
    review=review, source_hashes=sources, resources=dict(
        numerical_wall_s=record['numerical_wall_s'],numerical_limit_s=1200,
        peak_rss_KiB=record['peak_rss_KiB'],resguard_events_before=213,resguard_events_after=213,
        new_events=[],jobs=len(record['commands']),new_rollouts=0,cumulative_rollouts=0,
        rollout_retries=0,rollout_limit=6,session_budget=18,sessions_used=12),
    invariants=dict(horizon=35,dt_s=.02,site_membership_unchanged=True,
        certified_defaults_unchanged=True,controller_inputs='Archived default_sense(seed=17) estimated states',
        scoring_only_ground_truth=True),
    hardware_validation_needed=[
        'Synchronized issued position, velocity and feedforward commands, measured current/torque and encoders, with samples inside each 20 ms interval.',
        'Estimated base state with timestamp and calibration error, independently measured sole motion and physical loaded-body contact locations.',
        'Measured friction, compliance and command-to-actuator delay before assigning uncertainty ranges; supported recovery remains unvalidated.'],
    remaining_checkpoint_work=[],
    future_experiment='Constrain the predicted command interval and evaluate every physical substep and the next estimated endpoint under one reserved disturbance; reject for persistent force violations, contact loss or solve-plus-update latency over 20 ms.',
    limitations=['Saved-transition predictor comparison only; no new adapted controller or closed-loop benefit.',
        'Contact dynamics assume sticking and the source plant has the previously measured CAD/body mismatch.',
        'Synthetic base bias and ideal joint/velocity channels have no hardware calibration.'])
write(MEDIA/'20261016-cmpc-model-uncertainty.json',result)
archive = MEDIA/'20261018-model-uncertainty-raw.zip'
files = sorted(p for p in OUT.iterdir() if p.is_file())+scripts
files += [MEDIA/f'20261018-model-uncertainty-{s}.json' for s in ['plan','selection','review','run-record']]
manifest = {}
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:
        name=str(p.relative_to(ROOT));z.write(p,name);manifest[name]=sha(p)
with zipfile.ZipFile(archive) as z:
    assert set(z.namelist())==set(manifest)
    for name,digest in manifest.items():
        assert hashlib.sha256(z.read(name)).hexdigest()==digest
write(MEDIA/'20261018-model-uncertainty-archive.json',dict(
    archive_sha256=sha(archive),member_count=len(manifest),members=manifest,
    earlier_input_archive='docs/lean/media/20261015-contact-audit/20261017-foot-model-raw.zip',
    earlier_input_archive_sha256=sha(MEDIA/'20261017-foot-model-raw.zip')))
print(json.dumps(dict(status=result['status'],archive_members=len(manifest),checks=len(review['checks']),rollouts=0)))
