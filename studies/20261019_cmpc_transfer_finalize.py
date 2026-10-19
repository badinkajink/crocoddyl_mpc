#!/usr/bin/env python3
"""Publish the reviewed local synthesis and preserve the earlier draft archive.

This is document packaging only. Run the guarded completion measurement first,
inspect its result and frame sequences, and save current quota/guard output in
the completion run directory before running this command.
"""
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / 'docs/lean/media/20261015-contact-audit'
RUN = ROOT / 'studies/runs/20261019-cmpc-transfer-completion'
PREFIX = '20261016-cmpc-transfer-assessment'
report_path = MEDIA / '20261019-transfer-review-completion.json'
report = json.loads(report_path.read_text())
assert report['passed'] and all(c['passed'] for c in report['checks'])
assert report['new_rollouts'] == 0 and not report['regression_needed']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


metrics_path = MEDIA / (PREFIX + '.json')
page_path = ROOT / 'docs/lean' / (PREFIX + '.html')
# Always start from the immutable quota-limited snapshot so packaging is repeatable.
draft_archive = MEDIA / '20261019-transfer-review-raw.zip'
draft_manifest = json.loads((MEDIA / '20261019-transfer-review-archive.json').read_text())
assert sha(draft_archive) == draft_manifest['archive_sha256']
with zipfile.ZipFile(draft_archive) as z:
    decision = json.loads(z.read(PREFIX + '.json'))
    draft_page = z.read(PREFIX + '.html').decode()
assert decision['status'] == 'continue'
command = '/home/humanoid/.claude/bin/resguard.sh run --mem 1.6G --cpu 200 -- /usr/bin/time -v -o studies/runs/20261019-cmpc-transfer-completion/20261019-review-time.txt timeout 90s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 18 /home/humanoid/miniconda3/envs/croco/bin/python studies/20261019_cmpc_transfer_completion.py > studies/runs/20261019-cmpc-transfer-completion/20261019-review.log 2>&1'
package_command = 'python studies/20261019_cmpc_transfer_finalize.py'
time_output = (RUN / '20261019-review-time.txt').read_text()
elapsed_text = re.search(r'Elapsed .*?: (\S+)', time_output).group(1)
parts = [float(x) for x in elapsed_text.split(':')]
elapsed = sum(v * 60**i for i, v in enumerate(reversed(parts)))
rss = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', time_output).group(1))
quota = json.loads((RUN / '20261019-quota-save.json').read_text())
events = Path('/home/humanoid/.claude/resguard_events.jsonl').read_text().splitlines()
assert len(events) == 213, 'Read and report any new resource event before packaging.'
resource = dict(new_rollouts=0, cumulative_checkpoint_rollouts=0, regression_limit=2,
    numerical_wall_s=elapsed, numerical_limit_s=1200, cumulative_checkpoint_numerical_wall_s=round(.40+elapsed, 2),
    peak_rss_KiB=rss, cap='1.6G', cap_basis='Earlier measured saved-data RSS 1,095,964 KiB times 1.3',
    threads=1, cpu_quota=200, core=18, resguard_events_before=213, resguard_events_after=213,
    new_resguard_events=[], quota_save_five_hour_percent=quota['quota']['primary']['used_percent'],
    quota_save_weekly_percent=quota['quota']['secondary']['used_percent'],
    session_budget=quota['queue']['session_budget'], sessions_used=quota['queue']['sessions_used'],
    stopped_for_quota=False, post_ten_minute_check_due=False, thirty_minute_checkpoint_due=False,
    rollout_accounting=decision['resources']['rollout_accounting'],
    total_research_rollouts_including_retries=22, retries_included=2)
decision['prior_quota_limited_session_resources'] = decision['resources']
decision.update(status='checkpoint_complete', simulation_review_complete=True,
    required_checkpoint_work_remaining=[], resources=resource,
    completion_baseline_commit=report['baseline_commit'],
    completion_review=dict(path=str(report_path.relative_to(ROOT)), sha256=sha(report_path),
        checks=len(report['checks']), numerical_claims=len(report['numerical_claims']),
        local_links=len(report['local_links']), prior_input_digests_verified=73,
        frame_sequences_reinspected=True, new_regression_needed=False))
decision['reproducible_commands'] += [command, package_command]
decision['comparison_conditions']['observation_equality_limit'] = (
    'Identical sensing configuration and estimated/physical arrays before intervention; observations diverge with subsequent motion.')
decision['comparison_conditions']['nominal_disturbance'] = 'None; pulse applies only to disturbed trials.'
decision['completion_editorial_changes'] = [
    'Clarify identical sensing configuration versus differing post-intervention observations.',
    'Identify wrist contact and joint-sample counting in motion measurements.',
    'Add existing pulse motion rates and executable-command pressure margins.']
for q in decision['questions']:
    if q['id'] == 'motion_speed_runtime':
        q['result'] = q['result'].replace('first-contact wrist speed', 'wrist speed at its first table contact').replace(
            'saturated startup samples', 'saturated joint samples in the first 0.5 s')
for e in decision['remaining_proposed_experiments']:
    e['experiment'] = e['experiment'].replace('foot load and velocity settle', 'foot load and velocity stabilize')

page = draft_page
page = page.replace('Measurement saved; final artifact review pending', 'Simulation review complete; hardware validation outstanding')
page = page.replace('unchanged model, plan and observations', 'the same model, plan and sensing configuration')
page = page.replace('35 nodes at 0.02 s and a 90 N world +y wrist pulse from 4.50–4.80 s.',
    '35 nodes at 0.02 s. Disturbed trials receive a 90 N world +y wrist pulse from 4.50–4.80 s; nominal trials receive none.')
page = page.replace('first-contact wrist speed', 'wrist speed at its first table contact').replace(
    'saturated startup samples', 'saturated joint samples in the first 0.5 s')
page = page.replace('foot load and velocity settle', 'foot load and velocity stabilize')
pulse = decision['comparisons']['pulse']
b, c = pulse['baseline'], pulse['corrected']
motion_text = (f'<p>Over 4.50–6.02 s, pulse joint-speed peaks are {b["joint_speed_rad_s"]:.3f}/{c["joint_speed_rad_s"]:.3f} rad/s baseline/filtered; '
    f'acceleration peaks are {b["joint_acceleration_rad_s2"]:.1f}/{c["joint_acceleration_rad_s2"]:.1f} rad/s² and wrist-speed peaks '
    f'{b["wrist_speed_m_s"]:.3f}/{c["wrist_speed_m_s"]:.3f} m/s. For filtered issued-command model evaluations from 4 s, '
    'minimum left/right pressure margins are 4.445/9.479 mm under the pulse and approximately 0/24.395 mm nominal. '
    'The nominal left foot reaches the predictor rectangle boundary. These are model pressure margins, with physical slip scored separately.</p>')
page = page.replace('<p>The physical-brace experiment', motion_text + '\n<p>The physical-brace experiment')
page = page.replace('One guarded job took 0.40 s', 'The initial guarded audit took 0.40 s')
review_text = (f'<p>The completion review passes {len(report["checks"])} checks, verifies {len(report["numerical_claims"])} numerical statements '
    f'and {len(report["local_links"])} local links, and confirms all 73 original input digests. It takes {elapsed:.2f} s with peak memory '
    f'{rss:,} KiB under the same 1.6G cap; cumulative checkpoint numerical time is {resource["cumulative_checkpoint_numerical_wall_s"]:.2f} s. '
    'The earlier contact assessment is verified against its archived baseline snapshot. The original quota-limited archive is preserved. '
    'No new resguard event or regression rollout occurred.</p>\n<pre>' + html.escape(command) + '</pre>\n'
    '<p><a href="media/20261015-contact-audit/20261019-transfer-review-completion.json">Completion checks and numerical sources</a> · '
    '<a href="media/20261015-contact-audit/20261019-transfer-review-completion-run-record.json">Completion commands and final artifact hashes</a> · '
    '<a href="media/20261015-contact-audit/20261019-transfer-review-completion-raw.zip">Completion source and log archive</a></p>')
page = page.replace('<h2>Future work</h2>', review_text + '\n<h2>Future work</h2>')
page = page.replace('The saved simulation synthesis supports selecting a research baseline; final artifact and link review remains pending.',
    'Simulation review is complete. The filter remains an opt-in research baseline, with contact-aware recovery and command-interval force feasibility unresolved.')
page_path.write_text(page)
metrics_path.write_text(json.dumps(decision, indent=2) + '\n')

assessment_path = ROOT / 'docs/lean/20261015-cmpc-contact-reassessment.html'
assessment = assessment_path.read_text().replace('Under identical model, plan and estimated observations,',
    'With the same model, plan and sensing configuration,')
assessment = assessment.replace('Final artifact and synthesis review remains pending after the 82% quota stop.',
    'Completion review passes 157 checks, including 36 numerical statements and 145 local links. All 73 original input digests pass; the archived baseline preserves this assessment before synthesis. No regression was needed. Simulation review is complete; hardware validation remains outstanding.')
assessment_path.write_text(assessment)
index_path = ROOT / 'docs/index.html'
index = index_path.read_text().replace('Saved synthesis: 65 saved-data checks, zero new rollouts; final artifact review pending.',
    'Completed synthesis: 65 saved-data checks and 157 completion checks; zero new rollouts.')
index_path.write_text(index)

record_path = MEDIA / '20261019-transfer-review-completion-run-record.json'
record = dict(checkpoint='transfer_review', status='checkpoint_complete',
    baseline_commit=report['baseline_commit'], resources=resource,
    commands=[command, package_command], log=(RUN / '20261019-review.log').read_text(), time_output=time_output,
    quota_reads={p.name: json.loads(p.read_text()) for p in sorted(RUN.glob('*quota*.json'))},
    guard_before=(RUN / '20261019-guard-before.txt').read_text(), guard_after=(RUN / '20261019-guard-after.txt').read_text(),
    completion_review_sha256=sha(report_path), prior_archive_preserved=True,
    source_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                  [Path(__file__), ROOT / 'studies/20261019_cmpc_transfer_completion.py']},
    final_artifact_sha256={str(p.relative_to(ROOT)): sha(p) for p in
                          [page_path, metrics_path, assessment_path, index_path]},
    inspected_frame_sequences=decision['video_review']['frame_sequences_inspected'])
record_path.write_text(json.dumps(record, indent=2) + '\n')
archive_path = MEDIA / '20261019-transfer-review-completion-raw.zip'
with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for p in [*sorted(RUN.iterdir()), Path(__file__), ROOT / 'studies/20261019_cmpc_transfer_completion.py',
              report_path, record_path, page_path, metrics_path, assessment_path, index_path]:
        if p.is_file():
            z.write(p, str(p.relative_to(ROOT)))

# Check every new page link after all completion targets exist. External sources
# are inherited context; no new external claim or network check is introduced.
class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
    def handle_starttag(self, tag, attrs):
        for key, val in attrs:
            if key in ('href', 'src') and val:
                url = urlsplit(val)
                if url.path and not (url.scheme or url.netloc):
                    self.targets.append((page_path.parent / unquote(url.path)).resolve())
links = Links()
links.feed(page)
assert all(p.is_file() for p in links.targets)
assert 'final artifact review pending' not in page + index
assert decision['simulation_review_complete'] and not decision['hardware_validation_complete']
manifest = dict(archive_sha256=sha(archive_path), final_page_local_links_checked=len(links.targets),
    all_links_exist=True, final_status_consistent=True, prior_archive_preserved=True,
    record_sha256=sha(record_path), note='Final document hashes are in the completion run record; initial audit bytes remain in their original archive.')
(MEDIA / '20261019-transfer-review-completion-archive.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(status='checkpoint_complete', checks=len(report['checks']),
                     final_page_local_links=len(links.targets), numerical_wall_s=elapsed, new_rollouts=0)))
