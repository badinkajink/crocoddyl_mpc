#!/usr/bin/env python3
"""Package the completed transfer review from saved measurements (stdlib only)."""
import hashlib
import html
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'
RUN = ROOT/'studies/runs/20261019-cmpc-transfer-review'
PREFIX = '20261016-cmpc-transfer-assessment'
audit_path = MEDIA/'20261019-transfer-review-measurement.json'
audit = json.loads(audit_path.read_text())
assert audit['passed']


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


questions = [
 dict(id='foot_slip', checkpoints=['foot_motion','foot_model_transfer'],
      result='Executable-torque filtering reduces pulse left-sole displacement 78.460 to 0.304 mm, contact loss 210 to 0 ms, loaded slip 0.5177 to 0.0091 m/s and estimated sole residual 3.638 to 0.055 mm. Reanchoring changes state and wrench by zero with foot gains [0,50].',
      resolved='Reanchoring alone is ineffective for this plan; initial executable-force constraints improve this nominal/pulse comparison.',
      unresolved='Sticking dynamics cannot predict sliding or lift. Initial cone feasibility does not persist across the command interval. Fallback remains unexercised.',
      hardware=['H1','H2','H3'], sources=['20261015-cmpc-foot-motion','20261016-cmpc-foot-model-transfer']),
 dict(id='physical_brace_selection', checkpoints=['geometry','brace_selection_transfer'],
      result='Five CAD regions agree with collision boundaries within 0.165 mm, but only 2/18 saved witnesses belong to those small regions. Seven wrist-box witnesses are 8.637–23.515 mm from same-body CAD. After geometry correction, neither elbow–forearm nor forearm–wrist passes selection or reserved-target screens. Selected-body support minima are 182.21/0 N at x=1.06 m and 43.90/0 N at x=0.98 m.',
      resolved='Body identity and force location invalidate the old site-only certification; two physical strategies were compared with reserved validation.',
      unresolved='No admissible selection benefit. Forearm–wrist receives unselected wrist-roll support averaging 107.89/71.07 N. CAD extent does not establish load capacity.',
      hardware=['H2'], sources=['20261015-cmpc-surface-catalogue','20261016-cmpc-brace-selection-transfer']),
 dict(id='estimation', checkpoints=['state_frames'],
      result='A consistent translation/yaw preserves predicted forces within 8.04e-11 N and first torque within 1.35e-12 N m. A base-only 10 mm error changes force by 7.009 N and first torque by 0.01883 N m. Four scene mismatch checks reject inconsistent references.',
      resolved='Frame conversion and relative-scene consistency pass at the tested 35-node hold state, including nonzero-velocity conversion.',
      unresolved='Known-transform adapter does not estimate base bias. Synthetic seed-17 base bias and ideal joint/velocity channels do not validate a deployed estimator, drift, delay or slip dependence.',
      hardware=['H3'], sources=['20261015-cmpc-state-frames']),
 dict(id='motion_speed_runtime', checkpoints=['motion_transfer'],
      result='Added velocity cost reduces first-contact wrist speed 0.700 to 0.083 m/s and saturated startup samples 165 to 5. Startup acceleration remains 223.63 versus 226.12 rad/s², while left-sole excursion increases 18.80 to 24.35 mm. Fixed-position startup still reaches 175.62 rad/s². Best force-filter p95 is 24.22/23.96 ms against 20 ms.',
      resolved='Velocity regularization improves contact arrival but fails the combined acceleration/sole-motion criterion; initialization contributes to acceleration.',
      unresolved='Supported handover and deadline-aware execution remain untested. No same-condition measurement explains the approximately 90 s hardware maneuver. Earlier unlabelled 3x clips were replaced by 1x clips.',
      hardware=['H1','H4'], sources=['20261016-cmpc-motion-transfer','20261016-height-playback']),
 dict(id='target_commandability', checkpoints=['command_recovery_transfer'],
      result='A +20 mm target at 4 s produces 8.37 mm hand-x movement in the first 0.5 s; error decreases 27.39 to 18.96 mm. Nominal command-phase reach RMS is 19.72 mm, pulse RMS 187.71 mm. Brace onset is 2.380 s and original reach onset 2.402 s.',
      resolved='Combined position/velocity/feedforward commands reconstruct simulated actuator torque within 6.4e-14 N m; a small target update executes.',
      unresolved='The 20 mm target starts within the 30 mm reach screen, so zero acquisition delay is uninformative. Larger commands, height changes and hardware tracking are unmeasured.',
      hardware=['H1','H4'], sources=['20261016-cmpc-command-recovery']),
 dict(id='recovery', checkpoints=['command_recovery_transfer','adaptive_transfer'],
      result='Scheduled nominal recovery passes at 8.530 s, 3.330 s after request. Pulse recovery fails by 9 s with 8.424 mm sole motion and 13.555 N renewed gripper load. A first-command guard produces identical motion. Motion dwell reduces pulse sole motion to 1.306 mm but both gated conditions fail; nominal final table load is 54.799 N.',
      resolved='Feasible first-command force and low estimated motion are each insufficient release criteria under these tests.',
      unresolved='Contact-aware unloading, abort-to-support, recovery after 9 s and feasibility of the entire return path remain unperformed.',
      hardware=['H1','H2','H3','H4'], sources=['20261016-cmpc-command-recovery','20261016-cmpc-adaptive-transfer']),
 dict(id='model_uncertainty', checkpoints=['model_uncertainty_transfer'],
      result='On 100 nominal and 75 reserved pulse transitions, known servo evolution improves angular/joint-velocity residual over 2 ms frozen torque by 4.87%/12.18%, failing the requirement that both improve by at least 10%. Previous commands violate model cones at 75/100 and 57/75 observed endpoints.',
      resolved='Command/integration and contact feasibility are prerequisites to attributing residuals to physical parameters; the prerequisite branch is complete.',
      unresolved='No physical-parameter fitting, online model update, scenario MPC or adapted closed-loop comparison was performed. Hardware uncertainty ranges remain unidentified.',
      hardware=['H1','H2','H3'], sources=['20261016-cmpc-model-uncertainty'])]
hardware = [
 dict(id='H1', measurement='Time-synchronized issued position, velocity and feedforward commands; joint encoders; current or torque; send/receive timestamps and controller duration. Include samples inside each 20 ms command (simulation scoring used 2 ms). Record supported initialization and actuator saturation.', purpose='Separate command evolution, lag, tracking and transport delay; set measured motion limits.'),
 dict(id='H2', measurement='Identify actual loaded wrist-roll, wrist-yaw, forearm and gripper surfaces in body coordinates; measure table-relative clearance and normal/tangential load through loading and release. Measure loaded footprint, sole/table friction onset and compliance under known loads; record deployed asset hashes.', purpose='Validate force locations and moment arms, unloading criteria and physical parameter ranges.'),
 dict(id='H3', measurement='Synchronize raw odometry, fused base pose/velocity and joint estimates with independent base and sole motion, table pose and calibration transforms during stance and slip.', purpose='Measure relative frame bias, drift, latency and stationary-foot estimator failure without feeding ground truth to control.'),
 dict(id='H4', measurement='Timestamp owner-supervised command, brace, reach, release and stable recovery; log interventions, contacts and independent sole motion with labelled 1x video and simulation-equivalent criteria.', purpose='Explain maneuver duration and measure command/recovery success on the actual robot.')]
experiments = [
 dict(id='E1', priority=1, status='proposed_not_run', prerequisite=['H1','H2'],
      experiment='Contact-aware gripper/wrist unloading with a planned clearance path, compared with scheduled release on the same nominal/pulse tasks and 9 s deadline.',
      falsification='Reject if nominal table load remains above 5 N, disturbed recovery misses the original criteria, or motion/support worsens. Simulation thresholds require hardware calibration.'),
 dict(id='E2', priority=2, status='proposed_not_run', prerequisite=['H1','H2'],
      experiment='Constrain known servo evolution over each command interval, retaining initial torque filtering as baseline; score every physical substep and next estimated endpoint.',
      falsification='Reject if interval force violations or contact loss persist, reach worsens beyond the existing 5 mm allowance, or full controller latency exceeds 20 ms.'),
 dict(id='E3', priority=3, status='proposed_not_run', prerequisite=['H1','H3'],
      experiment='Re-solve from a measured supported handover after foot load and velocity settle; compare the original approach and velocity correction at identical initialization.',
      falsification='Persistent acceleration near 224 rad/s² rejects contact acquisition as a sufficient explanation; preserve reach and sole motion.'),
 dict(id='E4', priority=4, status='proposed_not_run', prerequisite=['H1','H2','H3'],
      experiment='After contact and command prerequisites pass, fit one identifiable physical parameter or select two or three models using earlier transitions; reserve a separate measured condition.',
      falsification='Reject without lower reserved prediction error and task improvement at the same observation path, or if full latency exceeds 20 ms. No arbitrary uncertainty grid.')]
command='/home/humanoid/.claude/bin/resguard.sh run --mem 1.6G --cpu 200 -- /usr/bin/time -v -o studies/runs/20261019-cmpc-transfer-review/20261019-audit-time.txt timeout 90s env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 18 /home/humanoid/miniconda3/envs/croco/bin/python studies/20261019_cmpc_transfer_review.py > studies/runs/20261019-cmpc-transfer-review/20261019-audit.log 2>&1'
decision=dict(checkpoint='transfer_review',date='2026-10-19 America/Denver',status='continue',
    decision='Retain executable-torque filtering as an opt-in simulation baseline. Do not promote a complete transfer controller; obtain contact/actuator/estimator measurements before hardware validation.',
    best_supported_non_RL_variant='Existing BoxFDDP plus executable-torque cone projection from 4 s; unchanged original staged geometry and contact membership.',
    hardware_validation_complete=False, simulation_review_complete=False, required_checkpoint_work_remaining=['Final artifact/link and synthesis consistency review; no rollouts needed'],
    questions=questions, resolved_questions=[dict(id=q['id'],answer=q['resolved']) for q in questions],
    unresolved_questions=[dict(id=q['id'],limitation=q['unresolved']) for q in questions],
    required_hardware_data=hardware, remaining_proposed_experiments=experiments,
    comparisons=audit['force_comparison'], recovery_comparison=audit['recovery_comparison'],
    comparison_conditions=dict(same_model_plan_initial_state_and_pre_activation_observation_verified=True,
      seed=17, observation='default_sense estimated-state path; 10 mm/0.01 rad synthetic base error, ideal joint/velocity channels',
      horizon=35,dt_s=0.02,force_filter_activation_s=4.0,pulse='90 N world +y at reaching wrist, 4.50–4.80 s',
      hold_rollout_s=6.02,recovery_rollout_s=9.02,synchronous_without_transport_delay=True,
      incompatible=audit['incompatible_comparisons']),
    measurement=dict(path=str(audit_path.relative_to(ROOT)),sha256=sha(audit_path),checks=65,
                     source_artifacts_hashed=73,archive_digests_verified=5),
    resources=dict(new_rollouts=0,cumulative_checkpoint_rollouts=0,regression_limit=2,
      numerical_wall_s=0.40,numerical_limit_s=1200,peak_rss_KiB=61956,cap='1.6G',
      cap_basis='Prior saved-data review RSS 1,095,964 KiB times 1.3',threads=1,cpu_quota=200,core=18,
      resguard_events_before=213,resguard_events_after=213,new_resguard_events=[],quota_save_five_hour_percent=82,quota_save_weekly_percent=58,
      stopped_for_quota=True,post_ten_minute_check_due=False,thirty_minute_checkpoint_due=False,
      rollout_accounting=audit['rollout_accounting'],total_research_rollouts_including_retries=22,retries_included=2),
    reproducible_commands=[command,'python studies/20261019_cmpc_transfer_assessment.py'],
    source_baseline_commit=audit['baseline_commit'],source_plan_and_model=audit['source_plan_and_model'],
    video_review=dict(new_renders=0,frame_sequences_inspected=['20261017-foot-model-baseline-pulse-frames.jpg',
        '20261017-foot-model-corrected-pulse-frames.jpg','20261019-adaptive-gated-nominal.jpg'],
        prior_complete_decode_checks='Original foot-model, command-recovery and adaptive review records',
        observation='Pulse baseline shows large posture change; filtered case preserves the brace. Gated nominal remains at the table through 9 s. Loads and submillimetre motion come from numerical scoring.'))
(MEDIA/(PREFIX+'.json')).write_text(json.dumps(decision,indent=2)+'\n')

esc=html.escape
def links(q):
    return ' · '.join('<a href="'+s+'.html">'+s+'</a>' for s in q['sources'])
rows=''
for condition, pair in audit['force_comparison'].items():
    for variant in ['baseline','corrected']:
        r=pair[variant]
        rows+=f'<tr><td>{condition.title()} / {variant}</td><td>{r["left_sole_displacement_mm"]:.3f}</td><td>{r["left_contact_loss_ms"]:.0f}</td><td>{r["left_loaded_slip_m_s"]:.4f}</td><td>{r["left_executable_prediction_rms_mm"]:.3f}</td><td>{r["reach_final_mm"]:.3f}</td><td>{r["joint_tracking_rms_mrad"]:.2f}</td><td>{r["solve_plus_audit_or_filter_p95_ms"]:.2f}</td></tr>'
sections=''.join('<h2>'+q['id'].replace('_',' ').capitalize()+'</h2><p>'+esc(q['result'])+'</p><p>'+esc(q['unresolved'])+'</p><p>Next hardware measurement: '+', '.join(q['hardware'])+'. '+links(q)+'</p>' for q in questions)
hardware_html=''.join('<p><strong>'+h['id']+'</strong>: '+esc(h['measurement'])+' '+esc(h['purpose'])+'</p>' for h in hardware)
future=''.join('<p>'+e['id']+': '+esc(e['experiment'])+' '+esc(e['falsification'])+' Required data: '+', '.join(e['prerequisite'])+'.</p>' for e in experiments)
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Non-RL transfer assessment of the leaning MPC</title>
<style>body{max-width:1120px;margin:36px auto;padding:0 20px;font:16px/1.6 system-ui;color:#222;background:#fafafa}h1{font-size:1.8rem}h2{font-size:1.2rem;margin-top:2em}table{border-collapse:collapse;width:100%;font-size:.84rem}td,th{padding:8px;text-align:left;border-bottom:1px solid #ccc}a{color:#126397}.terms{background:#edf2f4;padding:16px}code,pre{font-size:.8rem;overflow-wrap:anywhere}pre{white-space:pre-wrap}video{max-width:100%;width:540px}.meta{color:#555}</style>
<h1>Non-RL transfer assessment of the leaning MPC</h1>
<p class="meta">19 October 2026 · Measurement saved; final artifact review pending · Required filename retains the queue's 16 October prefix · Baseline 90ce6b7</p>
<p>Executable-torque force filtering is the best supported non-RL simulation variant: disturbed left-sole displacement falls from 78.460 to 0.304 mm with unchanged model, plan and observations. Nominal final reach error increases by 1.624 mm. Physical brace selection yields no admissible winner, and both proposed release corrections fail their recovery hypotheses. Retain the filter as an opt-in baseline. Hardware transfer remains unvalidated; the next measurements must establish loaded contact geometry, actuator response and estimator consistency.</p>
<div class="terms"><h2>Terms and metrics</h2><p>MPC is model predictive control; RL is reinforcement learning. Displacement is maximum sole-origin distance from the interval's first sample, in mm. The comparison table covers 4.50–6.02 s. Contact loss counts 2 ms samples without a floor witness above 1 N; loaded slip is tangential material-point speed above 5 N contact load. Residual is root-mean-square (RMS) error between the 20 ms executable-torque prediction and next estimated sole position. Reach is hand-to-target distance. Tracking is commanded-minus-physical joint-position RMS in mrad. Runtime p95 is the 95th percentile over 302 evaluations, including baseline audit or correction filtering.</p><p>Force cone excess measures violation of a model linear inequality: force rows use N and moment rows N m, with a 1e-6 acceptance tolerance. Forces are world-aligned at each contact origin. Normal force is positive in compression. Centre-of-pressure margin is distance inside the predictor's 200 × 80 mm rectangle; the physical collision sole is 250 mm long and up to 76 mm wide. CAD means computer-aided design geometry. Recovery requires 0.3 s of loaded feet, table load below 5 N, joint/wrist speed below 0.1 rad/s and 0.03 m/s, standing joint RMS below 0.1 rad, base height within 50 mm, sole displacement below 5 mm and sole speed below 0.02 m/s. These thresholds are simulation assumptions.</p></div>
<h2>Comparison with the original baseline</h2>
<p>The four force-filter conditions use the h0885 elbow–forearm plan, original staged wrist-box geometry, identical seed-17 estimated-state path, one BoxFDDP iteration, 35 nodes at 0.02 s and a 90 N world +y wrist pulse from 4.50–4.80 s. Filtering starts at 4 s. The review verifies identical pre-activation estimated states and physical motion plus model-array and plan hashes. The nominal and pulse cases each contain one trial per version. Two earlier incorrect filtered trials are excluded and counted as retries. No confidence interval or population success rate follows from these deterministic pairs.</p>
<table><tr><th>Condition / version</th><th>Left displacement (mm)</th><th>Loss (ms)</th><th>Slip (m/s)</th><th>Residual (mm)</th><th>Final reach (mm)</th><th>Tracking (mrad)</th><th>p95 (ms)</th></tr>'''+rows+'''</table>
<p>All 204 corrected hold-command model evaluations pass initial cone checks without fallback; 202 precede plant integration. Minimum forearm normal force is 1.000/1.010 N nominal/pulse. Original optimizer horizon force excess still reaches 13.183/27.009 N. Corrected p95 remains above 20 ms, with 91/94 of 302 evaluations exceeding the period versus baseline 72/65. Simulation executes synchronously and injects no computation or transport delay. Filtering leaves startup unchanged at 3.308 rad/s joint speed, 226.1 rad/s² acceleration and 1.183 m/s wrist speed.</p>
<p>The physical-brace experiment changes collision geometry and re-solves physical contact points. Recovery adds a target update and a return schedule, extending runs to 9.02 s. Those outcomes remain separate from the six-second force-filter comparison. Older August results use older assets. The predictor alternatives reuse saved transitions and supply no adapted-controller success rate.</p>
'''+sections+'''
<h2>Required hardware data</h2>'''+hardware_html+'''
<h2>Timestamped simulation records</h2>
<p>Existing videos retain their 1x labels and simulation clocks. The two pulse frame sequences and gated nominal sequence were inspected again for this review. Foot-model clips contain 181 frames at 30 fps; recovery clips contain 271. Prior checkpoint reviews verified their complete decoding and timestamp sampling.</p>
<video controls preload="metadata" src="media/20261015-contact-audit/20261017-foot-model-baseline-pulse.mp4"></video>
<video controls preload="metadata" src="media/20261015-contact-audit/20261017-foot-model-corrected-pulse.mp4"></video>
<p><a href="media/20261015-contact-audit/20261017-foot-model-baseline-pulse.mp4">Baseline pulse</a> · <a href="media/20261015-contact-audit/20261017-foot-model-corrected-pulse.mp4">Filtered pulse</a> · <a href="media/20261015-contact-audit/20261019-adaptive-gated-nominal.mp4">Failed gated nominal recovery</a></p>
<h2>Review measurement and reproducibility</h2>
<p>A saved-data audit passes 65 checks, hashes 73 source artifacts and verifies five recorded archive digests. It confirms trajectory equality before each intervention and exact reuse of recovery baseline records. No inconsistency required a regression rollout. This checkpoint uses zero of the two allowed regression rollouts; the research records contain 22 actual rollouts including two retries and excluding reused baselines. One guarded job took 0.40 s with peak resident memory 61,956 KiB under 1.6G / CPUQuota 200, one thread on core 18. Resguard remains at 213 events with no new refusal or kill. No plant step, optimizer solve, hardware action or new queue was performed.</p>
<p>Plan SHA-256: <code>56151d89b63733769d52901b5b5906bf52a6450ac2b1df73cd1265b3c5d1c9d6</code>. Staged top-level model SHA-256: <code>37b78e65b7c5cf0ce2d5106eeea54a630bd9f5d3439c77af94d1d7c0646fa696</code>. Loaded model-array hashes, per-checkpoint commands, source revisions and raw archives are linked from the audit and decision record. The top-level XML hash alone does not describe in-memory geometry changes.</p>
<pre>'''+esc(command)+'''</pre>
<p><a href="media/20261015-contact-audit/20261016-cmpc-transfer-assessment.json">Machine-readable questions, decision and proposed experiments</a> · <a href="media/20261015-contact-audit/20261019-transfer-review-measurement.json">Measured comparisons and artifact hashes</a> · <a href="media/20261015-contact-audit/20261019-transfer-review-run-record.json">Commands and resource record</a> · <a href="media/20261015-contact-audit/20261019-transfer-review-raw.zip">Review source, logs and quota archive</a> · <a href="../../studies/20261019_cmpc_transfer_review.py">Audit source</a></p>
<h2>Future work</h2>'''+future+'''
<p>These experiments are proposals and have not been queued. The saved simulation synthesis supports selecting a research baseline; final artifact and link review remains pending. Hardware validation requires the measurements above and an owner-supervised experiment.</p></html>'''
(ROOT/'docs/lean'/ (PREFIX+'.html')).write_text(page)
run_record=dict(checkpoint='transfer_review',status='continue',baseline_commit=audit['baseline_commit'],
    resources=decision['resources'],commands=decision['reproducible_commands'],
    log=(RUN/'20261019-audit.log').read_text(),time_output=(RUN/'20261019-audit-time.txt').read_text(),
    quota_reads={p.name:json.loads(p.read_text()) for p in RUN.glob('*quota*.json')},
    guard_before=(RUN/'20261019-guard-before.txt').read_text(),
    source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'studies/20261019_cmpc_transfer_review.py']},
    page_sha256=sha(ROOT/'docs/lean'/(PREFIX+'.html')),metrics_sha256=sha(MEDIA/(PREFIX+'.json')),
    measurement_sha256=sha(audit_path),video_review=decision['video_review'])
(MEDIA/'20261019-transfer-review-run-record.json').write_text(json.dumps(run_record,indent=2)+'\n')
archive=MEDIA/'20261019-transfer-review-raw.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RUN.iterdir()):
        if p.is_file():z.write(p,'run/'+p.name)
    for p in [Path(__file__),ROOT/'studies/20261019_cmpc_transfer_review.py',audit_path,MEDIA/(PREFIX+'.json'),ROOT/'docs/lean'/(PREFIX+'.html')]:
        z.write(p,p.name)
    historical='docs/lean/20261015-cmpc-contact-reassessment.html'
    z.writestr('baseline-contact-reassessment.html',subprocess.check_output(['git','show',audit['baseline_commit']+':'+historical],cwd=ROOT))
(MEDIA/'20261019-transfer-review-archive.json').write_text(json.dumps(dict(archive_sha256=sha(archive),
    note='Includes baseline assessment snapshot; source-input hashes describe the review baseline before the assessment update.'),indent=2)+'\n')
print('Saved transfer assessment, machine-readable decision and review archive.')
