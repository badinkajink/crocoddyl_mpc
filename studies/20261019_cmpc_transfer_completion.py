#!/usr/bin/env python3
"""Review the saved transfer synthesis without loading models or advancing physics.

Run under resguard. Verify artifact bytes, local result links, copied comparisons,
and numerical statements against the preceding checkpoint records. The original
quota-limited review archive remains immutable. This script writes a new review
record; completion metadata is packaged after inspecting its result.
"""
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/lean/media/20261015-contact-audit'
OUT = MEDIA / '20261019-transfer-review-completion.json'
checks, claims, hashes = [], [], {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check(name, passed, **detail):
    checks.append(dict(name=name, passed=bool(passed), **detail))


def read(name):
    path = MEDIA / name
    hashes[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def value(name, pointer):
    v = read(name)
    for part in pointer.split('/'):
        v = v[int(part)] if isinstance(v, list) else v[part]
    return v


def claim(name, pointer, displayed, tolerance):
    actual = value(name, pointer)
    passed = abs(actual - displayed) <= tolerance
    check(name + ' ' + pointer, passed, actual=actual,
          displayed=displayed, rounding_tolerance=tolerance)
    claims.append(dict(source=name, pointer=pointer, actual=actual,
                       displayed=displayed, tolerance=tolerance))


audit = read('20261019-transfer-review-measurement.json')
decision = read('20261016-cmpc-transfer-assessment.json')
page_path = ROOT / 'docs/lean/20261016-cmpc-transfer-assessment.html'
page = page_path.read_text()
check('prior saved-array audit', audit['passed'] and
      len(audit['checks']) == 65 and all(c['passed'] for c in audit['checks']))
check('assessment points to original audit bytes',
      decision['measurement']['sha256'] == sha(MEDIA / '20261019-transfer-review-measurement.json'))
check('copied force comparisons', decision['comparisons'] == audit['force_comparison'])
check('copied recovery comparisons', decision['recovery_comparison'] == audit['recovery_comparison'])
check('zero regression rollouts', decision['resources']['cumulative_checkpoint_rollouts'] == 0)
check('research rollout count includes retries', sum(audit['rollout_accounting'].values()) == 22)
check('hardware validation remains open', decision['hardware_validation_complete'] is False)

# All earlier inputs must retain their measured bytes. The contact reassessment
# was deliberately updated during synthesis; its original snapshot is archived.
archive_path = MEDIA / '20261019-transfer-review-raw.zip'
archive_manifest = read('20261019-transfer-review-archive.json')
check('original synthesis archive digest', sha(archive_path) == archive_manifest['archive_sha256'])
with zipfile.ZipFile(archive_path) as archive:
    check('original synthesis archive CRC', archive.testzip() is None)
    old_assessment = archive.read('baseline-contact-reassessment.html')
    for relative, digest in audit['input_sha256'].items():
        path = ROOT / relative
        actual = sha(path)
        basis = 'current bytes'
        if relative == 'docs/lean/20261015-cmpc-contact-reassessment.html':
            actual = hashlib.sha256(old_assessment).hexdigest()
            basis = 'archived baseline bytes; current assessment updated intentionally'
        check('input digest ' + relative, actual == digest, basis=basis)
    preserved = ['20261019_cmpc_transfer_assessment.py',
                 '20261019_cmpc_transfer_review.py',
                 '20261019-transfer-review-measurement.json']
    for name in preserved:
        current = ROOT / 'studies' / name if name.endswith('.py') else MEDIA / name
        check('archived source ' + name, archive.read(name) == current.read_bytes())
    check('archived quota-limited status', json.loads(archive.read(
        '20261016-cmpc-transfer-assessment.json'))['status'] == 'continue')

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for k, v in attrs:
            if k in ('href', 'src') and v:
                self.links.append(v)


pages = {page_path, ROOT / 'docs/lean/20261015-cmpc-contact-reassessment.html'}
for q in decision['questions']:
    pages.update(ROOT / 'docs/lean' / (s + '.html') for s in q['sources'])
local_links = []
external_links = set()
for path in sorted(pages):
    parser = Links()
    parser.feed(path.read_text())
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            external_links.add(link)
            continue
        if not url.path:
            continue
        target = (path.parent / unquote(url.path)).resolve()
        local_links.append(dict(page=str(path.relative_to(ROOT)), link=link,
                                exists=target.is_file()))
    check('local links ' + path.name, all(x['exists'] for x in local_links
                                       if x['page'] == str(path.relative_to(ROOT))))
check('index includes assessment once', (ROOT / 'docs/index.html').read_text().count(
    'href="lean/20261016-cmpc-transfer-assessment.html"') == 1)

hardware_ids = {x['id'] for x in decision['required_hardware_data']}
check('seven question groups', len(decision['questions']) == 7)
check('all nine earlier checkpoints covered', {c for q in decision['questions'] for c in q['checkpoints']} ==
      {'geometry', 'foot_motion', 'state_frames', 'motion_transfer', 'foot_model_transfer',
       'brace_selection_transfer', 'model_uncertainty_transfer', 'command_recovery_transfer', 'adaptive_transfer'})
for q in decision['questions']:
    check('question completeness ' + q['id'], all(q[k] for k in
          ['result', 'resolved', 'unresolved', 'hardware', 'sources']) and set(q['hardware']) <= hardware_ids)
for e in decision['remaining_proposed_experiments']:
    check('unperformed proposal ' + e['id'], e['status'] == 'proposed_not_run' and
          set(e['prerequisite']) <= hardware_ids and bool(e['falsification']))

# Verify every table cell, including the nominal cost of the correction.
for condition, pair in audit['force_comparison'].items():
    for variant in ['baseline', 'corrected']:
        r = pair[variant]
        fields = [('left_sole_displacement_mm', 3), ('left_contact_loss_ms', 0),
                  ('left_loaded_slip_m_s', 4), ('left_executable_prediction_rms_mm', 3),
                  ('reach_final_mm', 3), ('joint_tracking_rms_mrad', 2),
                  ('solve_plus_audit_or_filter_p95_ms', 2)]
        row = '<tr><td>' + condition.title() + ' / ' + variant + '</td>'
        row += ''.join('<td>' + format(r[k], '.' + str(n) + 'f') + '</td>' for k, n in fields)
        check('comparison table ' + condition + ' ' + variant, row + '</tr>' in page)

# Numerical statements from the other questions, with rounding at the displayed
# precision. Full raw measurements remain in their original checkpoint records.
motion = '20261016-cmpc-motion-transfer.json'
for tag, speed, acc, samples, excursion in [('baseline', .700, 226.12, 165, 18.80),
                                          ('velocity', .083, 223.63, 5, 24.35)]:
    prefix = 'cases/' + tag + '/'
    claim(motion, prefix + 'first_wrist_contact/wrist_speed_m_s', speed, .0005)
    claim(motion, prefix + 'startup_acceleration_rad_s2', acc, .005)
    claim(motion, prefix + 'startup_force_saturated_joint_samples', samples, 0)
    claim(motion, prefix + 'rollout/feet/sole_left/displacement_from_start_max_mm', excursion, .005)
claim(motion, 'cases/fixed-position/startup_acceleration_rad_s2', 175.62, .005)
state = '20261015-state-frames.json'
claim(state, 'metrics/base_only_10mm/contact_force_delta_N', 7.009, .0005)
claim(state, 'first_control/base_only_10mm/torque_delta_max_Nm', .01883, .000005)
check('frame invariance and four scene rejections', value(state, 'frame_invariance_pass') and
      len(value(state, 'scene_mismatch_rejected')) == 4 and all(value(state, 'scene_mismatch_rejected').values()))
brace = '20261016-cmpc-brace-selection-transfer.json'
check('no selected physical strategy', value(brace, 'decision/selected_strategy') is None)
for target, support, other in [('selection', 182.21, 107.89), ('reserved', 43.90, 71.07)]:
    claim(brace, 'conditions/' + target + '/elbow_forearm/credited_support_min_N', support, .005)
    claim(brace, 'conditions/' + target + '/forearm_wrist/credited_support_min_N', 0, 0)
    claim(brace, 'conditions/' + target + '/forearm_wrist/body_support_mean_N/left_wrist_roll_link', other, .005)
recovery = '20261016-cmpc-command-recovery.json'
claim(recovery, 'cases/baseline-nominal/command_response/hand_x_motion_first_half_second_mm', 8.37, .005)
claim(recovery, 'cases/baseline-nominal/command_response/error_at_command_mm', 27.39, .005)
claim(recovery, 'cases/baseline-nominal/command_response/error_before_pulse_mm', 18.96, .005)
claim(recovery, 'cases/baseline-nominal/phases/command/target_error_rms_mm', 19.72, .005)
claim(recovery, 'cases/baseline-pulse/phases/command/target_error_rms_mm', 187.71, .005)
claim(recovery, 'cases/baseline-nominal/timing/brace_onset_s', 2.380, .0005)
claim(recovery, 'cases/baseline-nominal/timing/original_reach_onset_s', 2.402, .0005)
claim(recovery, 'cases/baseline-nominal/timing/standing_recovery_onset_s', 8.530, .0005)
claim(recovery, 'cases/baseline-nominal/timing/standing_recovery_after_request_s', 3.330, .0005)
claim(recovery, 'cases/baseline-pulse/post_release_max_sole_displacement_mm', 8.424, .0005)
check('release guard branch unexercised', value(recovery, 'decision/guard_activations') == 0)
adaptive = '20261016-cmpc-adaptive-transfer.json'
claim(adaptive, 'cases/gated-pulse/post_release_max_sole_displacement_mm', 1.306, .0005)
for tag in ['gated-nominal', 'gated-pulse']:
    check('recovery fails ' + tag, value(adaptive, 'cases/' + tag + '/complete_cycle') is False)
review = '20261019-adaptive-review.json'
claim(review, 'measurements/baseline-pulse/final_robot_table_load_N', 13.555, .0005)
claim(review, 'measurements/gated-nominal/final_robot_table_load_N', 54.799, .0005)
uncertainty = '20261016-cmpc-model-uncertainty.json'
claim(uncertainty, 'decision/nominal_improvement_fraction', .0487, .00005)
claim(uncertainty, 'decision/reserved_improvement_fraction', .1218, .00005)
for tag, n, count in [('nominal', 100, 75), ('pulse', 75, 57)]:
    claim(uncertainty, 'cases/' + tag + '/transitions', n, 0)
    claim(uncertainty, 'review/cases/' + tag + '/endpoint_violating_transitions', count, 0)

hashes[str(page_path.relative_to(ROOT))] = sha(page_path)
hashes[str(Path(__file__).relative_to(ROOT))] = sha(Path(__file__))
result = dict(checkpoint='transfer_review', date='2026-10-19',
              baseline_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              passed=all(c['passed'] for c in checks), checks=checks, numerical_claims=claims,
              local_links=local_links, external_links_not_rechecked=sorted(external_links),
              input_sha256=hashes, prior_input_digests_verified=len(audit['input_sha256']),
              new_rollouts=0, cumulative_checkpoint_rollouts=0, physics_steps=0, optimizer_solves=0,
              regression_needed=False, incompatible_model_groups_remain_separate=True,
              hardware_validation_complete=False)
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(dict(passed=result['passed'], checks=len(checks),
                     numerical_claims=len(claims), local_links=len(local_links), output=str(OUT))))
if not result['passed']:
    print(json.dumps([c for c in checks if not c['passed']], indent=2))
    raise SystemExit(1)
