#!/usr/bin/env python3
"""Package saved measurements and the inspected videos; no numerical job."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'studies/runs/20261019-cmpc-adaptive-transfer'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'
PAGE = ROOT/'docs/lean/20261016-cmpc-adaptive-transfer.html'
def read(p): return json.loads(p.read_text())
def dump(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

metrics_path = MEDIA/'20261016-cmpc-adaptive-transfer.json'
a = read(metrics_path)
review = read(MEDIA/'20261019-adaptive-review.json')
assert review['passed'] and all(c['passed'] for c in review['checks'])
assert not a['decision']['hypothesis_passed']
commands = read(OUT/'20261019-commands.json')
total, peak = 0., 0
for c in commands:
    t = (OUT/('20261019-'+c['name']+'-time.txt')).read_text()
    elapsed = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', t).group(1)
    seconds = 0.
    for x in elapsed.split(':'): seconds = seconds*60+float(x)
    rss = int(re.search(r'Maximum resident set size \(kbytes\): (\d+)', t).group(1))
    code = int(re.search(r'Exit status: (\d+)', t).group(1))
    assert code == 0
    c.update(elapsed_s=seconds, peak_rss_KiB=rss, exit_code=code)
    total += seconds; peak = max(peak, rss)
events = Path('/home/humanoid/.claude/resguard_events.jsonl').read_text().splitlines()
record = dict(checkpoint='adaptive_transfer', date='2026-10-19', baseline_commit='c2777dd',
    status='checkpoint_complete', rollouts_this_session=2, cumulative_rollouts=2,
    reused_baseline_rollouts=2, rollout_retries=0, rollout_limit=6,
    numerical_wall_s=round(total, 2), numerical_limit_s=1200, peak_rss_KiB=peak,
    cap='4.4G', cap_basis='Prior measured 3,526,672 KiB times 1.3', cpu_quota=200, threads=1, core=18,
    resguard_events_before=213, resguard_events_after=len(events),
    new_resguard_events=[json.loads(s) for s in events[213:]], commands=commands,
    quota_save=read(OUT/'20261019-quota-save.json'),
    baseline_archive='docs/lean/media/20261015-contact-audit/20261018-command-recovery-raw.zip',
    baseline_archive_sha256=sha(MEDIA/'20261018-command-recovery-raw.zip'),
    notes=['Pulse selection was saved before the reserved nominal rollout. No parameter retuning.',
        'Both comparisons reuse previously reviewed baselines. All 68 baseline archive members and model/source hashes were verified; pre-request physical and estimated trajectories agree.',
        'Pulse source was reconstructed after cleanup of its docstring and unused legacy_protocol function; its active AST equals the nominal source. Selection scorer snapshot matches its recorded source hash. No controller behavior changed between rollouts.',
        'All six numerical jobs passed. No rollout retry, agent, hardware command, queue, runner/config/limit/timer/default/site-membership change or push.',
        'Artifact-save quota read precedes ten minutes; a post-ten-minute read and thirty-minute checkpoint were not yet due.'])
dump(MEDIA/'20261019-adaptive-run-record.json', record)
a['decision'].update(status='checkpoint_complete', measurement_complete=True, review_complete=True,
    cumulative_rollouts=2, rollout_retries=0, remaining_checkpoint_work='None',
    conclusion='Low estimated motion before release is insufficient: pulse motion improves but recovery misses 9 s; nominal validation loses recovery and retains table load.',
    limitation='One synthetic sensing seed and one pulse; later recovery after 9 s is unmeasured. Contact loads are scoring only. Contact-interval feasibility, geometry, sensing delay and hardware response remain unresolved.')
a['review'] = dict(path='20261019-adaptive-review.json', checks=len(review['checks']),
    physical_samples=sum(x['physical_samples'] for x in review['measurements'].values()),
    evaluated_commands=sum(x['evaluated_commands'] for x in review['measurements'].values()),
    frame_sequences_inspected=list(a['cases']), frames_per_sequence=8,
    visual_finding='Gated pulse returns upright by 9 s; gated nominal remains inclined over the table. Physical logs determine loads and millimetre sole motion.')
a['resources'] = record
a['transfer_assessment'] = dict(simulation='Reject motion-only release dwell as a general correction. Retain opt-in for further contact-aware planning research.',
    hardware_measurement='Synchronize table-relative gripper/wrist clearance and normal loads through unloading with issued commands, encoder/current response, estimated base/joint state and independent sole motion. Measure loaded geometry, friction/compliance and transport delay before selecting hardware thresholds.',
    later_checkpoint='transfer_review remains separate and unperformed by this worker.')
sources = [ROOT/'studies'/f for f in ['20261019_cmpc_adaptive_transfer.py',
    '20261019_cmpc_adaptive_transfer_analysis.py', '20261019_cmpc_adaptive_transfer_review.py',
    '20261019_cmpc_adaptive_transfer_package.py']]
a['source_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in sources})
dump(metrics_path, a)

order = ['baseline-pulse','gated-pulse','baseline-nominal','gated-nominal']
task_rows, motion_rows, force_rows = [], [], []
for tag in order:
    c = a['cases'][tag]; r = c['phases']['recovery']; v = review['measurements'][tag]
    recovery = c['timing']['standing_recovery_onset_s']
    recovery = 'Failed by 9 s' if recovery is None else f'{recovery:.3f}'
    task_rows.append(f"<tr><td>{tag}</td><td>{c['summary']['transition']['t']:.2f}</td><td>{recovery}</td><td>{c['post_release_max_sole_displacement_mm']:.3f}</td><td>{v['final_robot_table_load_N']:.3f}</td></tr>")
    slip = max(x['loaded_slip_max_m_s'] for x in r['feet'].values())
    motion_rows.append(f"<tr><td>{tag}</td><td>{r['joint_speed_peak_rad_s']:.3f}</td><td>{r['joint_acceleration_peak_rad_s2']:.1f}</td><td>{r['wrist_speed_peak_m_s']:.3f}</td><td>{slip:.4f}</td><td>{1000*r['joint_tracking_rms_rad']:.2f}</td><td>{r['policy_p95_ms']:.2f}</td><td>{r['policy_over_20ms']}/{r['commands']}</td></tr>")
    f, h = r['executable_forces'], r['horizon_forces']
    force_rows.append(f"<tr><td>{tag}</td><td>{min(x['normal_min_N'] for x in f.values()):.2f}</td><td>{max(x['force_row_max_N'] for x in f.values()):.2g}</td><td>{max(x['moment_row_max_Nm'] or 0 for x in f.values()):.2g}</td><td>{max(x['force_row_max_N'] for x in h.values()):.2f}</td><td>{max(x['moment_row_max_Nm'] or 0 for x in h.values()):.2f}</td></tr>")
videos = ''.join(f'<figure><video controls preload="metadata" src="media/20261015-contact-audit/{a["cases"][tag]["video"]}"></video><figcaption>{tag}. Simulation timestamps, 1x playback; eight-frame sequence inspected.</figcaption></figure>' for tag in order)
PAGE.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Estimated-motion gating for brace release</title>
<style>body{max-width:1120px;margin:36px auto;padding:0 20px;font:16px/1.55 system-ui;color:#222}h1{font-size:1.8rem}h2{font-size:1.2rem}table{width:100%;border-collapse:collapse;font-size:.86rem}th,td{padding:8px;border-bottom:1px solid #ccc;text-align:left}.terms{background:#edf3f5;padding:16px}video{width:100%;max-width:800px}a{color:#156295}</style>
<h1>Estimated-motion gating for brace release</h1><p>19 October 2026 · Adaptive transfer checkpoint · Baseline c2777dd</p>
<p>Waiting for low estimated motion reduces disturbed sole displacement from 8.424 to 1.306 mm and final table load from 13.555 to 0 N, but recovery still misses the 9 s deadline. Reserved nominal validation loses the baseline's successful recovery and finishes with 54.799 N of robot table load. The hypothesis fails in two new rollouts compared with two archived baselines. A motion-only release gate remains opt-in; contact-aware unloading remains unresolved.</p>
<div class="terms"><h2>Terms and metrics</h2><p>Release time is the first feet-only controller command. Standing recovery requires 0.3 s with each foot above 5 N, robot table load below 5 N, joint speed below 0.1 rad/s, brace-wrist speed below 0.03 m/s, standing joint root-mean-square error below 0.1 rad, base height within 50 mm of standing, sole displacement below 5 mm and sole tangential speed below 0.02 m/s. These inherited simulation screens have no hardware calibration.</p>
<p>Sole displacement is maximum three-dimensional sole-origin motion since the first physical sample at or after the 5.2 s request. Loaded slip is tangential sole speed while vertical load exceeds 5 N. Tracking is root-mean-square commanded minus physical joint position. Acceleration uses 2 ms velocity differences. Table load excludes the passive object's 2.943 N. Final values use 9.018 s; phase maxima cover 5.2–9 s. Policy p95 is the 95th percentile solve, projection and gate time; misses exceed the 20 ms period. Cone excess is violation of a model force row in N or moment row in N m, with a 1e-6 acceptance tolerance. Wrenches are world-aligned at contact origins. Current-command feasibility and horizon predictions are scored separately.</p></div>
<h2>Source failure and frozen intervention</h2><p>The <a href="20261016-cmpc-command-recovery.html">preceding recovery experiment</a> measured 8.424 mm sole motion and renewed gripper support after the pulse, despite feasible first-command forces. The <a href="media/20261015-contact-audit/20261019-adaptive-plan.json">saved plan</a> tests a 0.3 s dwell below the same joint, wrist and sole speed thresholds before executing the unchanged release feasibility test and standing controller. The braced controller continues while waiting. All velocity calculations use the existing estimated State; true motion and contact loads enter scoring only.</p>
<p>The original +20 mm target command at 4 s, 90 N world +y wrist pulse at 4.5–4.8 s, seed-17 sensing assumptions, 35-node horizon and 0.02 s step remain. The pulse decision was saved before nominal validation, with no retuning. Success requires recovery by 9 s in both conditions. Baseline archives and model hashes pass verification; estimated and physical trajectories agree before the request.</p>
<h2>Recovery and support</h2><table><tr><th>Condition</th><th>Release (s)</th><th>Recovery onset (s)</th><th>Sole maximum (mm)</th><th>Final table load (N)</th></tr>'''+''.join(task_rows)+'''</table>
<p>At the pulse request, estimated joint speed is 0.375 rad/s. The dwell first passes at 7.20 s. During the final 160 samples, foot loading, sole motion, table release and standing pose all pass; quiet motion fails throughout. Final joint and wrist speeds are 0.179 rad/s and 0.0413 m/s. Recovery after 9 s was not measured.</p>
<p>Nominal estimated speed passes at 5.20 s, but its current quiet interval is only 0.08 s; subsequent motion resets the dwell and release occurs at 6.00 s. Robot table load never drops below 13.109 N after the request. The final load is 26.328 N at the left gripper and 28.471 N at the wrist-yaw bodies, and standing joint error is 0.234 rad. Low speed therefore admits a release state that fails the physical unloading and posture criteria.</p>
<h2>Motion, commands and runtime</h2><table><tr><th>Condition</th><th>Joint peak (rad/s)</th><th>Acceleration (rad/s²)</th><th>Wrist peak (m/s)</th><th>Loaded slip (m/s)</th><th>Tracking (mrad)</th><th>Policy p95 (ms)</th><th>Misses</th></tr>'''+''.join(motion_rows)+'''</table>
<p>Both feet remain loaded throughout 5.2–9 s in every condition; the lowest corrected foot load is 42.92 N nominal and 80.50 N pulse. The delayed cases spend part of this interval solving the braced problem. Release evaluates one candidate and then solves again for the issued command, taking 29.70 ms nominal and 27.73 ms pulse. Runtime uses one thread on core 18. Startup is unchanged at 3.308 rad/s joint speed, 226.1 rad/s² acceleration and 1.183 m/s wrist speed. The target step begins inside the inherited 30 mm reach screen; its zero acquisition delay measures no tracking speed.</p>
<h2>Predicted contact-force feasibility</h2><table><tr><th>Condition</th><th>Current normal minimum (N)</th><th>Current force excess (N)</th><th>Current moment excess (N m)</th><th>Horizon force excess (N)</th><th>Horizon moment excess (N m)</th></tr>'''+''.join(force_rows)+'''</table>
<p>Every issued command after 4 s passes the model cone check without fallback. Future horizon violations persist, and the nominal physical table contacts are absent from the feet-only return model. Command reconstruction agrees with all actuator samples within 6.4e-14 N m; rebuilt contact wrenches agree within 3.8e-11 N or N m. Feasibility between command updates remains untested by this correction.</p>
<h2>Timestamped simulation videos</h2>'''+videos+f'''
<h2>Reproduction and resources</h2><p><a href="media/20261015-contact-audit/20261016-cmpc-adaptive-transfer.json">Metrics and decision</a> · <a href="media/20261015-contact-audit/20261019-adaptive-selection.json">Frozen pulse selection</a> · <a href="media/20261015-contact-audit/20261019-adaptive-review.json">Independent review</a> · <a href="media/20261015-contact-audit/20261019-adaptive-run-record.json">Exact commands and resources</a> · <a href="media/20261015-contact-audit/20261019-adaptive-raw.zip">Raw data and source archive</a> · <a href="../../studies/20261019_cmpc_adaptive_transfer.py">Opt-in implementation</a>.</p>
<p>All {len(review['checks'])} review checks pass across 1,808 saved commands and 18,040 physical samples. Four videos fully decode at 30 fps, 271 frames and 9.033 s, with at most 0.667 ms timestamp rounding. All four eight-frame sequences were inspected. Two new rollouts used 2/6; there were no retries. Six serial guarded jobs used {total:.2f} s of 1200 s, with peak RSS {peak:,} KiB under 4.4G. Resguard remained at 213 events with no new refusal or kill. No certified default or original site membership changed.</p>
<h2>Transfer assessment and Future work</h2><p>Simulation rejects low-motion dwell as a sufficient general release correction. The disturbed improvement accompanies a 2 s delay, and nominal contact unloading regresses. The remaining model-based question is whether planned gripper/wrist clearance and measured support state can select a feasible unloading path. A later comparison should retain this baseline and reject a proposed path if its nominal table load remains above 5 N or disturbed recovery still misses the fixed deadline.</p>
<p>Hardware validation needs synchronized table-relative gripper/wrist clearance and normal load through unloading, issued commands, encoder/current response, estimated base/joint state and independent sole motion. Loaded contact geometry, friction/compliance and transport delay require measurement before choosing hardware thresholds. The present sensing path uses synthetic pose bias and ideal velocity channels; no hardware trial or new queue was started. The separate transfer-review checkpoint remains unperformed here.</p></html>''')

index = ROOT/'docs/index.html'; text = index.read_text(); link = 'lean/20261016-cmpc-adaptive-transfer.html'
if link not in text:
    anchor = '<li><span class="d">10-19 · CMPC</span><a href="lean/20261016-cmpc-command-recovery.html">'
    assert anchor in text
    row = '<li><span class="d">10-19 · CMPC</span><a href="'+link+'">Estimated-motion gating for brace release</a><span class="n">Pulse sole motion 8.424 → 1.306 mm; both gated recoveries fail by 9 s. Reserved nominal run retains 54.799 N table load. Two new rollouts; 475 review checks pass.</span></li>\n'
    index.write_text(text.replace(anchor, row+anchor, 1))

note = ROOT/'docs/handoff/20261015-cmpc-research.md'; text = note.read_text()
if 'State: Adaptive-transfer checkpoint complete 2026-10-19' not in text:
    command_block = '\n'.join(c['command'] for c in commands)
    state = f'''State: Adaptive-transfer checkpoint complete 2026-10-19 from c2777dd. Two new 9.02 s rollouts compare an estimated-motion release dwell against the independently reviewed scheduled-release nominal/pulse baselines. Cumulative checkpoint rollouts 2/6, retries 0; two earlier baseline rollouts are reused, not rerun. The pulse decision was frozen before reserved nominal validation. All 475 review checks pass across 1,808 commands and 18,040 physical samples, with all 68 baseline archive members verified. Four labelled 1x video frame sequences were inspected; all four videos fully decode. No required checkpoint work remains.

State decision: Reject the hypothesis that low estimated motion suffices for release. Pulse release moves from 5.20 to 7.20 s; sole displacement decreases 8.424 to 1.306 mm and final table load 13.555 to 0 N, but final joint/wrist speeds 0.179 rad/s and 0.0413 m/s fail quiet recovery by 9 s. Reserved nominal release moves to 6.00 s and loses the baseline recovery at 8.53 s: final table load is 54.799 N, standing joint RMS error 0.234 rad, and post-request load never falls below 13.109 N. The 59.902 N value is the final-half-second maximum, not final load. Keep the correction opt-in. Recovery beyond 9 s was not performed.

State mechanism and contract: The gate requires 0.3 s continuously below the preceding screens: maximum joint speed 0.1 rad/s, brace-wrist speed 0.03 m/s and both sole tangential speeds 0.02 m/s. These are inherited simulation assumptions, not fitted hardware limits. At 5.2 s the pulse joint speed is 0.375 rad/s; nominal speed passes but its quiet interval is only 0.08 s and later resets. Only the existing default_sense(seed=17) State enters kinematics and control; true contact loads and poses remain scoring only. Full horizon 35, dt 0.02 s, original sites and certified defaults are unchanged. Pre-request physical and estimated trajectories match baseline. No force rejection or abort occurs.

State feasibility and runtime: Current issued commands after 4 s pass cone tolerance without fallback; horizon force/moment excess reaches 21.26 N / 15.58 N m in gated pulse and 9.14 N / 4.65 N m in gated nominal. Both feet remain loaded. Recovery-interval gated pulse/nominal joint peaks are 1.546/0.999 rad/s, acceleration 224.1/194.7 rad/s², wrist peaks 0.249/0.115 m/s and maximum loaded slip 0.0114/0.0066 m/s. Tracking RMS is 2.38/36.72 mrad. Policy p95 is 23.02/23.51 ms; transition costs 27.73/29.70 ms exceed the 20 ms period. Servo reconstruction error is below 6.4e-14 N m. Startup remains unchanged. Final shutdown command has no subsequent plant integration.

State artifacts: docs/lean/20261016-cmpc-adaptive-transfer.html; docs/lean/media/20261015-contact-audit/20261016-cmpc-adaptive-transfer.json; 20261019-adaptive-plan.json, selection.json, review.json, run-record.json, raw.zip and archive.json in the same directory; two 20261019-adaptive-gated-*.mp4 videos with timestamp metadata and frame sequences. Added studies/20261019_cmpc_adaptive_transfer.py, _analysis.py, _review.py and _package.py. docs/index.html updated. Exact commands, logs, timing, raw arrays, quota reads and source snapshots are archived. The pulse source snapshot was reconstructed after docstring/unused-function cleanup; active AST matches the nominal source, and the selection snapshot matches its recorded hash.

State resources: Six serial guarded jobs used {total:.2f} s / 1200 s, peak RSS {peak:,} KiB under 4.4G / CPUQuota 200 / one thread / core 18. Cap derives from the earlier measured 3,526,672 KiB times 1.3. Resguard remained 213 entries, no new refusal or kill. Final guard remained open. Save quota is {record['quota_save']['quota']['primary']['used_percent']:.0f}% five-hour / {record['quota_save']['quota']['secondary']['used_percent']:.0f}% weekly; local budget remains 18 total sessions, 15 used at the read. Work was saved before ten minutes; a post-ten-minute read and thirty-minute checkpoint were not yet due. No agent, hardware action, autonomous queue, runner/config/limit/timer/default/site change or push. Only checkpoint files are committed.

State commands:

```bash
{command_block}
```

State transfer assessment: Simulation supports smaller disturbed sole motion with delayed release but rejects the correction as a general recovery gate. Low velocity does not establish gripper/wrist unloading or a feasible return path. Hardware needs synchronized table-relative gripper/wrist clearance and normal load, commands, encoder/current response, estimated base/joint state and independent sole motion, plus loaded geometry, friction/compliance and transport delay. Later contact-aware release planning and recovery after 9 s remain unperformed. This worker completed only adaptive_transfer; the separate transfer_review remains pending. No indispensable owner decision is needed for that review. No queue was appended or armed.

Previous completed checkpoint follows.

'''
    at = text.index('State: '); note.write_text(text[:at]+state+text[at:])

files = sorted(p for p in OUT.rglob('*') if p.is_file())+sources
archive = MEDIA/'20261019-adaptive-raw.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for p in files: z.write(p, str(p.relative_to(ROOT)))
dump(MEDIA/'20261019-adaptive-archive.json', dict(archive_sha256=sha(archive),
    members={str(p.relative_to(ROOT)):sha(p) for p in files}))
print(json.dumps(dict(status='checkpoint_complete', numerical_wall_s=total, peak_rss_KiB=peak,
    archive_members=len(files), new_rollouts=2, resguard_events=len(events))))
