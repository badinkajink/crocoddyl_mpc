#!/usr/bin/env python3
"""Package already measured data using only the standard library."""
import hashlib
import json
import re
import zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'studies/runs/20261018-cmpc-command-recovery'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
PAGE=ROOT/'docs/lean/20261016-cmpc-command-recovery.html'
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=MEDIA/'20261016-cmpc-command-recovery.json';a=json.loads(p.read_text())
a['decision'].update(status='continue',measurement_complete=True,review_complete=False,
    remaining='Inspect four 1x videos/frame sequences, independently review saved support/motion/command metrics and release scheduling, finalize page and note. Four of six rollouts used; no further rollout needed.',
    stop_reason='Local five-hour quota reached 82%; owner requires save and return continue.')
commands=json.loads((OUT/'20261018-commands.json').read_text());total=0;peak=0
for r in commands:
 t=(OUT/('20261018-'+r['name']+'-time.txt')).read_text()
 elapsed=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)',t).group(1)
 sec=0
 for z in elapsed.split(':'):sec=sec*60+float(z)
 rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',t).group(1))
 r.update(elapsed_s=sec,peak_rss_KiB=rss,exit_code=int(re.search(r'Exit status: (\d+)',t).group(1)))
 total+=sec;peak=max(peak,rss)
events=Path('/home/humanoid/.claude/resguard_events.jsonl').read_text().splitlines()
record=dict(date='2026-10-18',checkpoint='command_recovery_transfer',status='continue',
    baseline_commit='a83ac6c',rollouts_this_session=4,cumulative_rollouts=4,rollout_limit=6,rollout_retries=0,
    numerical_wall_s=total,numerical_limit_s=1200,peak_rss_KiB=peak,cap='4.4G',cpu_quota=200,threads=1,core=18,
    resguard_events_before=213,resguard_events_after=len(events),new_resguard_events=[json.loads(x) for x in events[213:]],
    commands=commands,quota_save=json.loads((OUT/'20261018-quota-save.json').read_text()),
    notes=['First saved-data score included the passive object load; corrected score excludes its 2.943 N. Both versions and the first scorer source are archived.',
        'No failed numerical invocation, no rollout retry. One patch operation to update the command log failed and was corrected; no experiment was affected.',
        'Session stopped for quota before ten minutes; a post-ten-minute quota read was not yet due.',
        'Four render jobs were executed serially within one guarded render command. Video/frame inspection and independent saved-data review remain.',
        'No agent, queue, runner, timer, default, site-membership, hardware action or push.'])
dump(MEDIA/'20261018-command-recovery-run-record.json',record)
sources=[ROOT/'studies'/f for f in ['20261018_cmpc_command_recovery.py','20261018_cmpc_command_recovery_analysis.py','20261018_cmpc_command_recovery_package.py']]
a['source_hashes'].update({str(p.relative_to(ROOT)):sha(p) for p in sources})
a['resources']=record
dump(p,a)
rows=[];motion=[];force=[]
for tag,z in a['cases'].items():
 t=z['timing'];r=z['phases']['recovery'];c=z['phases']['command']
 rec='Failed by 9 s' if t['standing_recovery_after_request_s'] is None else f"{t['standing_recovery_after_request_s']:.3f}"
 rows.append(f"<tr><td>{tag}</td><td>{t['brace_onset_s']:.3f}</td><td>{t['original_reach_onset_s']:.3f}</td><td>{rec}</td><td>{z['post_release_max_sole_displacement_mm']:.3f}</td></tr>")
 motion.append(f"<tr><td>{tag}</td><td>{r['joint_speed_peak_rad_s']:.3f}</td><td>{r['joint_acceleration_peak_rad_s2']:.1f}</td><td>{r['wrist_speed_peak_m_s']:.3f}</td><td>{1000*r['joint_tracking_rms_rad']:.2f}</td><td>{r['policy_p95_ms']:.2f}</td><td>{r['policy_over_20ms']}/{r['commands']}</td></tr>")
 f=r['executable_forces'];h=r['horizon_forces']
 force.append(f"<tr><td>{tag}</td><td>{min(x['normal_min_N'] for x in f.values()):.2f}</td><td>{max(x['force_row_max_N'] for x in f.values()):.2g}</td><td>{max(x['moment_row_max_Nm'] for x in f.values()):.2g}</td><td>{max(x['moment_row_max_Nm'] for x in h.values()):.3f}</td></tr>")
video=''.join(f'<figure><video controls preload="metadata" src="media/20261015-contact-audit/20261018-recovery-{tag}.mp4"></video><figcaption>{tag}. Simulation time and 1x playback labelled. Visual review pending.</figcaption></figure>' for tag in a['cases'])
PAGE.write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Command transitions and disturbed recovery</title>
<style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.55 system-ui;color:#222}h1{font-size:1.8rem}h2{font-size:1.2rem}table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}.terms{background:#edf3f5;padding:16px}video{width:100%;max-width:800px}a{color:#156295}</style>
<h1>Command transitions and disturbed recovery</h1><p>18 October 2026 · Saved checkpoint, review pending · Baseline a83ac6c</p>
<p>Nominal recovery passes at 8.530 s, 3.330 s after the release request. Both pulse recoveries fail by 9 s: maximum post-release sole displacement is 8.424 mm, against a 5 mm screen, and gripper contact returns after release. A first-command force-feasibility guard accepts both releases and produces identical trajectories to scheduled release. Four rollouts are complete; the guard's abort branch remains unexercised. Local quota reached 82%, so this checkpoint is saved as continue pending independent data and video review.</p>
<div class="terms"><h2>Terms and metrics</h2><p>Brace onset requires both feet above 5 N vertical load and robot table load above 20 N for 0.3 s. Reach onset adds hand error below 30 mm. Standing recovery requires 0.3 s of both feet loaded, robot table load below 5 N, joint speed below 0.1 rad/s, wrist speed below 0.03 m/s, standing joint root-mean-square error below 0.1 rad, base height within 50 mm of standing, sole displacement below 5 mm since the release request, and sole tangential speed below 0.02 m/s. These are simulation screens, not hardware limits.</p>
<p>Sole displacement is the maximum three-dimensional sole-origin motion. Slip is tangential sole speed while vertical load exceeds 5 N. Tracking is root-mean-square commanded minus physical joint position. Acceleration uses velocity differences at 2 ms. Force excess is positive violation of a model cone row: N for the first five rows and N m for the remaining foot rows. Forces are world-aligned at each contact origin. Policy p95 is the 95th percentile compute time, including solve, projection and transition work, excluding scoring. The control deadline is 20 ms. The passive object's 2.943 N table load is excluded from robot support.</p></div>
<h2>Comparison and command schedule</h2><p>Both versions retain the previous executable-torque projection and the same seed-17 estimated-state path. The target changes from x=1.06 m to 1.08 m at 4 s, with unchanged height 0.9982 m and table height 0.885 m. The earlier 90 N world +y reaching-wrist pulse is applied from 4.5 to 4.8 s. At 5.2 s, a 35-node feet-only return problem uses the existing standing costs and previous solution as its initial guess. Its first executable command is tested; the baseline always releases, while the guard rejects an infeasible projection and would regulate the current estimated braced posture toward zero velocity. Every horizon is 35 nodes at 0.02 s. No true pose or contact force enters either policy.</p>
<p>The saved hypothesis requires the guarded controller to finish supported at rest in both conditions without increasing sole motion. The disturbed result fails that screen. No projection rejection occurs, so the abort implementation has no rollout validation. The guard evaluates a first command; it neither detects physical release nor guarantees the later contact schedule. Release evaluation and the issued first return solve are two consecutive solver calls at the transition; policy latency includes both.</p>
<h2>Task times and support</h2><table><tr><th>Condition</th><th>Brace onset (s)</th><th>Original reach onset (s)</th><th>Recovery after request (s)</th><th>Post-release sole maximum (mm)</th></tr>'''+''.join(rows)+'''</table>
<p>The +20 mm command starts inside the 30 mm reach screen, so the reported zero acquisition delay does not measure tracking speed. Before the pulse, hand x moves 8.37 mm in 0.5 s and error decreases from 27.39 to 18.96 mm. Nominal command-phase error is 19.72 mm RMS; the pulse raises it to 187.71 mm. Physical robot table load first stays below 5 N at 5.274 s nominal and 5.418 s disturbed, but disturbed gripper contact later returns. At 9.018 s the left and right grippers transmit 5.04 and 8.51 N to the table.</p>
<p>During recovery, both feet remain above 5 N in all four runs. Left/right displacement is 0.576/0.577 mm nominal and 2.585/8.423 mm disturbed over 5.2–9 s. Loaded slip peaks at 0.0084/0.0146 m/s nominal and 0.0101/0.0151 m/s disturbed. Startup remains severe: 3.308 rad/s joint speed, 226.1 rad/s² acceleration and 1.183 m/s wrist speed; foot displacement reaches 18.80/29.91 mm, with 30/24 ms below the 5 N support threshold.</p>
<h2>Recovery motion and runtime</h2><table><tr><th>Condition</th><th>Joint peak (rad/s)</th><th>Acceleration (rad/s²)</th><th>Wrist peak (m/s)</th><th>Tracking RMS (mrad)</th><th>Policy p95 (ms)</th><th>Deadline misses</th></tr>'''+''.join(motion)+'''</table><p>Command-phase policy p95 is 23.67–24.16 ms, above the 20 ms period. Simulation runs synchronously without transport latency; these timings do not establish real-time hardware operation. Motion equality is bit-for-bit between baseline and guarded versions; timing varies between serial runs.</p>
<h2>Force feasibility</h2><table><tr><th>Condition</th><th>Minimum predicted normal (N)</th><th>Executable force excess (N)</th><th>Executable moment excess (N m)</th><th>Horizon moment excess (N m)</th></tr>'''+''.join(force)+'''</table><p>Every scored executable command from 4 to 9 s satisfies the first-node cone tolerance of 1e-6, without fallback. Before projection, horizon force excess reaches 6.65 N nominal and 43.83 N disturbed during the command phase; minimum forearm normal is −1.80/−2.90 N. Physical individual floor contacts have zero recorded Coulomb-cone excess. Comparing their aggregate wrench against the smaller predictor rectangle gives whole-episode excess up to 36.30 N and 15.72 N m; the local centre-of-pressure margin relative to that rectangle reaches −40.76 mm. Actual collision geometry and predictor support rectangles differ, as measured in the earlier foot audit. Command-interval force validity remains unresolved.</p>
<h2>Timestamped simulation videos</h2>'''+video+f'''
<h2>Resources and saved review</h2><p>Four of six allowed rollouts are used, with no retries. Seven serial guarded jobs used {total:.2f} s of the 1200 s numerical allowance; peak RSS was {peak:,} KiB under 4.4G, CPU quota 200, one thread on core 18. Resguard events remained at 213. All 24 current scorer assertions pass. The initial score included passive-object load; its source and output are preserved beside the corrected result. Video inspection and independent score review remain unperformed. No hardware action, agent, default change or queue change occurred.</p>
<p><a href="media/20261015-contact-audit/20261016-cmpc-command-recovery.json">Metrics and decision</a> · <a href="media/20261015-contact-audit/20261018-command-recovery-run-record.json">Exact commands, resource use and quotas</a> · <a href="media/20261015-contact-audit/20261018-command-recovery-raw.zip">Raw data, sources and logs</a></p>
<h2>Transfer limits and Future work</h2><p>Nominal standing recovery is supported in this simulation; disturbed recovery fails despite a feasible first release command. The prescribed feet-only schedule omits renewed gripper support. Synthetic base-pose bias and ideal joint/velocity channels remain the observation assumptions. Hardware needs synchronized estimated state, encoders, issued commands and torque/current, independent sole motion, contact geometry and load, friction/compliance and transport delay.</p><p>Resume this checkpoint by reviewing the saved videos and independently recomputing support and motion criteria; no new rollout is needed. The emerging simulation question is whether estimated contact-state and motion checks before release can prevent the measured gripper recontact and 8.42 mm right-sole motion. A later comparison must preserve this baseline and test its pulse condition before changing any defaults.</p></html>''')
index=ROOT/'docs/index.html';text=index.read_text();link='lean/20261016-cmpc-command-recovery.html'
if link not in text:
 anchor='<li><span class="d">10-18 · CMPC</span><a href="lean/20261016-cmpc-model-uncertainty.html">'
 row='<li><span class="d">10-18 · CMPC</span><a href="'+link+'">Command transitions and disturbed recovery</a><span class="n">Saved, review pending: four rollouts; nominal recovery 3.33 s after release, pulse recovery fails with 8.42 mm sole motion. First-command guard gives no benefit.</span></li>\n'
 assert anchor in text;text=text.replace(anchor,row+anchor,1);index.write_text(text)
archive=MEDIA/'20261018-command-recovery-raw.zip'
files=sorted(p for p in OUT.rglob('*') if p.is_file())+sources
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for f in files:z.write(f,str(f.relative_to(ROOT)))
dump(MEDIA/'20261018-command-recovery-archive.json',dict(archive_sha256=sha(archive),members={str(f.relative_to(ROOT)):sha(f) for f in files}))
print(json.dumps(dict(numerical_wall_s=total,peak_rss_KiB=peak,events=len(events),files=len(files),status='continue')))
