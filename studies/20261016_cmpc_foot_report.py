#!/usr/bin/env python3
"""Package measured foot-motion results. Standard-library file processing only."""
from pathlib import Path
import hashlib
import html
import json
import re
import subprocess
import zipfile
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'studies/runs/20261016-cmpc-foot-motion'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
REPORT=ROOT/'docs/lean/20261015-cmpc-foot-motion.html'
DATA=MEDIA/'20261015-foot-motion.json'
a=json.loads(DATA.read_text());n=a['nominal'];p=a['pulse']

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def esc(x):return html.escape(str(x))
def table(headers,rows):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def f(x,k=2):return f'{x:.{k}f}'

jobs=[]
for name,script,extra,cap,timeout,kind in [
 ('probe','20261016_cmpc_foot_motion.py','--probe','3G',90,'Failed before model loading: missing CL_ASSETS_DIR'),
 ('probe2','20261016_cmpc_foot_motion.py','--probe','3G',90,'Full-horizon and anchor-null probe'),
 ('nominal','20261016_cmpc_foot_motion.py','--case nominal','3.4G',180,'Nominal rollout'),
 ('pulse','20261016_cmpc_foot_motion.py','--case pulse','3.4G',180,'Pulse rollout'),
 ('analysis','20261016_cmpc_foot_analysis.py','','1.6G',120,'Failed scoring: a future brace contact had no node-zero samples'),
 ('analysis2','20261016_cmpc_foot_analysis.py','','1.6G',120,'Contact and phase scoring'),
 ('render','20261016_cmpc_foot_analysis.py','--render','1.6G',120,'Two labelled 1x videos'),
 ('analysis3','20261016_cmpc_foot_analysis.py','','1.6G',120,'Actuator clipping and physical sole geometry checks'),
 ('analysis4','20261016_cmpc_foot_analysis.py','','1.6G',120,'Include final 10 ms in hold and post-pulse summaries')]:
 text=(RUN/f'20261016-{name}-time.txt').read_text()
 elapsed=re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)',text).group(1)
 seconds=0
 for v in elapsed.split(':'):seconds=60*seconds+float(v)
 rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',text).group(1))
 code=int(re.search(r'Exit status: (\d+)',text).group(1))
 env='OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '+('CMPC_THREADS=1 ' if 'motion.py' in script else '')+'MUJOCO_GL=egl'
 cmd=f'/home/humanoid/.claude/bin/resguard.sh run --mem {cap} --cpu 200 -- /usr/bin/time -v -o studies/runs/20261016-cmpc-foot-motion/20261016-{name}-time.txt timeout {timeout}s env {env} taskset -c 18 /home/humanoid/miniconda3/envs/croco/bin/python studies/{script} {extra} > studies/runs/20261016-cmpc-foot-motion/20261016-{name}.log 2>&1'
 jobs.append(dict(name=name,command=cmd,wall_s=seconds,max_rss_KiB=rss,exit_code=code,purpose=kind,cap=cap))
quota=json.loads((RUN/'20261016-quota-after-10min.json').read_text())['quota']
resource=dict(date='2026-10-16 America/Denver',checkpoint='foot_motion',rollouts_started=2,rollouts_completed=2,rollout_retries=0,
    checkpoint_rollout_limit=6,serial_numerical_wall_s=sum(j['wall_s'] for j in jobs),serial_numerical_limit_s=1200,jobs=jobs,
    events_before=197,events_after=len(Path('/home/humanoid/.claude/resguard_events.jsonl').read_text().splitlines()),new_events=[],
    quota_after_ten_minutes=quota,
    quota_final=json.loads((RUN/'20261016-quota-final.json').read_text())['quota'],
    memory_sizing='Probe cap 3G; measured 2685052 KiB. Both rollout caps were 3.4G, sized at 1.3 times that probe. Actual rollout peak 3038720 KiB; future full rollout cap must be >=3.8G. Replay/render cap 1.6G derives from prior measured height-render RSS 1175792 KiB.',
    failures=['First probe: missing asset environment path; fixed locally in the new script, zero physics steps.','First analysis: max() on an empty set of node-zero brace samples during approach; raw rollout files retained; fixed and rescored without integration.'],
    invariants='35 nodes, dt 0.02 s, certified plans and defaults, SITES5 and ARM_SITES unchanged. No solver installation, hardware command, agent, queue or push.',
    future='Motion/command and contact-feasibility corrections remain separate checkpoints; hardware friction, state error and actuator bandwidth are unmeasured.')
save(MEDIA/'20261016-foot-motion-run-record.json',resource)
provenance=json.loads((RUN/'20261016-probe.json').read_text())
urdf=ROOT.parent/'CL_Assets/ros_assets/h1_2_magpie.urdf'
provenance['urdf']=str(urdf);provenance['urdf_sha256']=hashlib.sha256(urdf.read_bytes()).hexdigest()
provenance['source_hashes']={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [ROOT/'studies/20261016_cmpc_foot_motion.py',ROOT/'studies/20261016_cmpc_foot_analysis.py']}
save(MEDIA/'20261016-foot-motion-provenance.json',provenance)
verification={}
for case in ['nominal','pulse']:
 video=MEDIA/f'20261016-foot-{case}.mp4'
 ff=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=duration,nb_frames,avg_frame_rate','-of','json',str(video)],text=True))['streams'][0]
 assert ff['nb_frames']=='181' and ff['avg_frame_rate']=='30/1'
 verification[case]=dict(ffprobe=ff,metadata=json.loads(video.with_suffix('.json').read_text()),inspected_frames_s=[0,.13,.5,2.4,4.5,4.8,5.5])
save(MEDIA/'20261016-foot-video-verification.json',verification)
a['decision']=json.loads((RUN/'20261016-decision.json').read_text());a['resources']=resource;a['video_verification']=verification
save(DATA,a)
archive=MEDIA/'20261016-foot-motion-raw.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for path in sorted(RUN.iterdir()):
  if path.is_file():z.write(path,arcname=path.name)
 z.writestr('20261016-archive-readme.txt','Two actual rollouts. CSV: qpos/qvel/ctrl/actuator force at 500 Hz. NPZ: sole poses, velocities, forces, estimated and predicted states, executable commands. Force JSON: all 35 nodes sampled every 0.1 s. Commands JSON: 50 Hz after command clamp. Reconstructed contacts JSON: forward scoring only, no physics integration. Original native foot-contact records remain separate. Summary JSON has per-rollout metrics; docs/lean/media/20261015-contact-audit/20261015-foot-motion.json adds phase scoring and corrections. Reproduction commands, failures and resource usage are in the adjacent 20261016-foot-motion-run-record.json.\n')
save(MEDIA/'20261016-foot-motion-archive.json',dict(path=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,files=len(list(RUN.iterdir()))))

motion_rows=[]
for name,x in [('Nominal',n),('Pulse',p)]:
 for phase in ['startup','hold']:
  y=x['summary']['motion'][phase];pts=y['points']
  motion_rows.append([name+' / '+phase,f(y['joint_speed_rad_s'],3),f(y['joint_acceleration_rad_s2'],1),f(pts['brace_wrist']['speed_m_s'],3),f(pts['brace_wrist']['acceleration_m_s2'],1),f(pts['reaching_hand']['speed_m_s'],3),f(pts['reaching_hand']['acceleration_m_s2'],1)])
foot_rows=[]
for name,x in [('Nominal',n),('Pulse',p)]:
 for foot in ['sole_left','sole_right']:
  y=x['phases']['post_pulse'][foot];slip=x['phases']['post_pulse']['simulated_contacts'][foot]['loaded_material_point_slip_peak_m_s']
  foot_rows.append([name+' / '+foot.replace('sole_',''),f(y['displacement_max_from_phase_start_mm']),f(y['tangential_max_from_phase_start_mm']),f(y['rotation_max_from_phase_start_deg']),f(y['contact_loss_s']*1000,0),f(slip,3),f(y['estimated_one_step_position_rms_mm'],3)])
force_rows=[]
for name,x in [('Nominal',n),('Pulse',p)]:
 for contact,y in x['phases']['hold']['predicted'].items():
  force_rows.append([name+' / '+contact.replace('brace_','').replace('sole_','foot '),f(y['normal_min_N'],1),f(y['tangent_max_N'],1),f(y['force_rows_max_N'],2),f(y['moment_rows_max_Nm'],2) if 'moment_rows_max_Nm' in y else '—',f(y['cop_margin_min_mm'],1) if 'cop_margin_min_mm' in y else '—'])
sim_rows=[]
for name,x in [('Nominal',n),('Pulse',p)]:
 for foot in ['sole_left','sole_right']:
  y=x['summary']['simulated'][foot];h=x['phases']['hold'][foot]
  sim_rows.append([name+' / '+foot.replace('sole_',''),f(y['world_normal_min_N'],1)+' to '+f(y['world_normal_max_N'],1),f(y['world_tangent_max_N'],1),f(h['simulated_local_cop_margin_min_mm'],2),f(h['simulated_pressure_centre_mesh_margin_min_mm'],3)])
command_rows=[]
for name,x in [('Nominal',n),('Pulse',p)]:
 y=x['summary'];command_rows.append([name,f(max(w['control_violation_Nm'] for w in y['window_checks']),3),f(max(w['joint_violation_rad'] for w in y['window_checks'])*1000,2),f(y['joint_violation_max_rad']*1000,2),f(y['ctrl_bound_violation_max_rad'],3),f(x['servo_check']['position_servo_identity_torque_max_error_Nm'],2),f(y['motion']['hold']['q_tracking_rms_rad']*1000,2),f(y['reach_final_mm'],1)])
style='body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.55 system-ui;color:#222;background:#fafafa}h1{font-size:1.8rem}h2{font-size:1.2rem;margin-top:2em}a{color:#146496}table{border-collapse:collapse;width:100%;font-size:.86rem}th,td{text-align:left;padding:8px;border-bottom:1px solid #ccc}.scroll{overflow:auto}.terms{padding:14px 20px;background:#edf2f4}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}video,img{width:100%}figure{margin:18px 0}figcaption,.meta{font-size:.9rem;color:#555}code{font-size:.85em}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:750px){.grid{display:block}}'
page=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Foot motion and command feasibility during a lateral wrist pulse</title><style>{style}</style>
<h1>Foot motion and command feasibility during a lateral wrist pulse</h1>
<p class="meta">16 October 2026 · Foot-motion checkpoint · Two 6.02 s MuJoCo rollouts · Repository baseline 7c0240e</p>
<p>A 90 N lateral wrist pulse increased left-sole displacement from 0.14 to 78.46 mm after disturbance onset, with 210 ms of lost loaded floor contact. Final reach error increased from 17.6 to 271.1 mm. The predictor retained all 35 nodes but requested negative contact forces and infeasible wrenches. Reactive reanchoring changed neither the next state nor contact force because this plan has zero foot-position stabilization gain. The actuator audit found a second failure: position-servo input clipping changes requested torque by up to 44.67 N m in the disturbed run.</p>
<div class="terms"><h2>Terms and metrics</h2>
<p>CMPC is this repository’s Crocoddyl model predictive controller. Displacement and rotation are the largest sole-centre translation and rotation-angle change from the named interval’s first sample. Tangential displacement uses world x/y. Contact loss means no floor witness on that ankle body above 1 N, sampled every 2 ms. Loaded slip is the largest horizontal velocity of a foot material point at a floor contact above 5 N. It distinguishes sliding from sole-centre motion caused by rotation.</p>
<p>Prediction residual is the root-mean-square (RMS) distance between the next estimated sole pose and the solver’s first future state, 20 ms later. It includes shooting defects and actuator realization. Fixed-anchor residual is distance to the original plan’s sole centre. Normal force is signed compression; negative values request tension. Tangential force is the horizontal force magnitude. Cone violation is distance outside each linear row bound: rows 0–4 use N, foot rows 5–16 use N m. These rows are soft penalties.</p>
<p>Centre-of-pressure (CoP) margin is the smallest remaining distance to an assumed support boundary; negative is outside. Wrench CoP uses signed force and moment at the sole centre and requires normal force above 5 N. Mesh pressure margin uses the normal-load-weighted contact position projected into the sole’s bottom polygon. Speed is the largest joint absolute velocity or point translational speed. Acceleration is an adjacent 2 ms velocity difference. Startup covers 0–0.5 s; hold covers 4–6.02 s; post-pulse covers 4.5–6.02 s. Position tracking is RMS joint command minus simulated position.</p></div>
<h2>Plan, observation and experiment</h2>
<p>The archived <code>h0885_x1060/cell/plan_elbow_forearm.json</code> uses a 0.885 m tabletop, a 2.4 s approach and 80 braced nodes. Its target is (1.060, −0.2348, 0.9982) m and its additional velocity cost is zero. The reconstructed staged Magpie model has 200 action nodes. Index 165 selects the final 35-node, 0.70 s window; all 302 solves in each episode retained 35 nodes at 0.02 s. The saved plan reports <code>converged=false</code>; its <code>is_feasible=true</code> field does not certify contact inequalities.</p>
<p>Both runs use <code>MuJoCoPlant.read → default_sense(seed=17) → State → policy</code>. This existing estimated-state error path adds constant base-position and orientation biases drawn with 10 mm and 0.01 rad standard deviations. Joint and velocity channels are ideal. Those S17 assumptions have no new hardware calibration. Scoring and rendering use ground truth. The opt-in runner uses one solver iteration, one thread and zero transport compensation for synchronous simulation. No controller, plan, site membership or default was edited.</p>
<p>The measured premise was the archived plan’s 12.11/7.54 mm foot-reference errors and the height audit’s startup motion. The hypothesis predicted greater sole motion and residual under one 90 N world +y force on the reaching wrist from 4.50 to 4.80 s. Amplitude and duration come from the earlier 30/60/90 N, 0.3 s study; direction and onset are scenario assumptions. The conditions are identical before 4.5 s to machine precision. Both displacement and estimated residual increased, satisfying the pre-recorded comparison.</p>
<div class="grid"><figure><video controls preload="metadata" src="media/20261015-contact-audit/20261016-foot-nominal.mp4"></video><figcaption>Nominal, labelled 1×. No disturbance.</figcaption></figure><figure><video controls preload="metadata" src="media/20261015-contact-audit/20261016-foot-pulse.mp4"></video><figcaption>Pulse, labelled 1×. The force indicator is active from 4.50 to 4.80 s.</figcaption></figure></div>
<p>Both clips contain 181 frames at 30 fps and last 6.033 s. Simulation timestamps select the nearest recorded pose with at most 0.667 ms rounding. The final recorded state is at 6.018 s; the last video sample is 6.000 s. Frame sequences at startup, approach and pulse onset/release were inspected.</p>
<h2>Sole motion and reactive reanchoring</h2>
{table(['Case / foot','Displacement (mm)','Tangential (mm)','Rotation (deg)','Contact loss (ms)','Loaded slip (m/s)','Estimated residual RMS (mm)'],foot_rows)}
<p>These measurements cover 4.5–6.02 s. Before the disturbance, startup already displaced the left/right soles by 18.80/29.91 mm and produced 30/24 ms without loaded floor contact. Whole-run fixed-anchor residuals were 18.80/29.91 mm nominal and 74.41/29.91 mm disturbed. The estimated counterparts were 43.58/54.44 and 98.66/54.44 mm, including base-pose bias.</p>
<p>The opt-in correction replaced each foot reference with its estimated contact pose at identical state and control. All 122 sampled comparisons returned exactly zero next-state and wrench difference. The foot gains are 0 for position and 50 for velocity. Reference updates therefore have no dynamical effect here; a separate reanchored rollout would repeat the same controller. No physical sliding model or stepping transition was tested.</p>
<h2>Predicted force feasibility</h2>
{table(['Hold case / contact','Minimum normal (N)','Maximum tangent (N)','Force-row violation (N)','Moment-row violation (N m)','Minimum CoP margin (mm)'],force_rows)}
<p>The table covers all 35 prediction nodes sampled every 0.1 s during hold. The disturbed forearm’s 1665.33 N row violation also occurs at node zero. Negative forces and moment violations already occur in the nominal run. The controller’s world-aligned contact force is evaluated at the named site origin; cone matrices use identity orientation. The predictor’s 200 × 80 mm foot rectangle consequently stays aligned with world x/y. Raw matrices, bounds, row units and signed force vectors are saved.</p>
<h2>Simulated contacts and sole geometry</h2>
{table(['Case / foot','Whole-run normal range (N)','Maximum tangent (N)','Hold CoP / model rectangle (mm)','Hold mesh pressure margin (mm)'],sim_rows)}
<p>MuJoCo contact force uses the normal and two tangent axes in the contact frame; it acts on geom 2 and is sign-reversed for geom 1. Forces are summed on each ankle body, moments shifted to the sole centre, then rotated into the moving sole frame for the CoP calculation. Per-contact normals remained nonnegative. Native pyramidal-cone excess was below 0.000001 N for the recorded contacts, while the robot still slipped. Reconstructed foot wrenches agree with live logs within 0.000069 N and 0.000069 N m.</p>
<p>The collision sole has a 250 mm long, at most 76 mm wide bottom polygon at body z = −45 mm. The predictor uses a 200 × 80 mm rectangle centred at x = 33 mm, z = −55 mm. MuJoCo floor friction is 1.0; the predictor coefficient is 0.6. Thus a negative margin against the predictor rectangle alone does not establish an impossible simulated pressure distribution. Nominal pressure reaches the mesh edge; the disturbed left pressure projection extends 2.50 mm beyond the bottom polygon as contact moves onto other foot surfaces. The wrist-roll body and other unplanned bodies also contact the table; their forces are retained in the raw records.</p>
<h2>Startup motion and actuator commands</h2>
{table(['Case / interval','Joint speed (rad/s)','Joint acceleration (rad/s²)','Brace wrist (m/s)','Brace wrist (m/s²)','Reach hand (m/s)','Reach hand (m/s²)'],motion_rows)}
<p>The shared startup peak is 3.308 rad/s at left hip pitch, 70 ms after release. The 226.1 rad/s² peak occurs at right ankle pitch around 129 ms. Acceleration uses 2 ms samples here, versus 10 ms in the earlier height audit. These sampling rates and the changed observation path prevent a direct acceleration ranking against the archived 133.9 rad/s² knee peak. The nominal hold also contains short ankle acceleration peaks despite low wrist speed.</p>
{table(['Case','Predicted torque violation (N m)','Predicted joint violation (mrad)','Simulated joint violation (mrad)','Servo-input bound excess (rad)','Torque loss/change from input clip (N m)','Hold tracking RMS (mrad)','Final reach error (mm)'],command_rows)}
<p>Joint-position commands are clamped before conversion to MuJoCo’s position-servo input. Feedforward and desired-velocity terms then shift that input outside its own bounds. Clipping changes the realized torque by 11.16 N m at left hip yaw in the nominal run and 44.67 N m at right hip yaw in the disturbed run. Reconstructing the servo with both input and force clipping matches logged actuator force within 0.1 piconewton metre. Actuator force-bound violations are zero. This comparison measures executable simulation commands; hardware torque and velocity tracking remain unmeasured.</p>
<h2>Runtime, artifacts and limitations</h2>
<p>Solver p95 time was {f(n['summary']['solve_p95_ms'],2)} ms nominal and {f(p['summary']['solve_p95_ms'],2)} ms disturbed on one CPU. {n['summary']['solve_above_20ms']}/{n['summary']['periods']} and {p['summary']['solve_above_20ms']}/{p['summary']['periods']} solves exceeded the 20 ms period. Simulation ran synchronously with no wall-clock delay injection. Rollout loops took {f(n['summary']['wall_s'],2)} and {f(p['summary']['wall_s'],2)} s; all guarded numerical jobs, including failed probes, rescoring and video, used {f(resource['serial_numerical_wall_s'],2)} s. Two of six permitted rollouts were used, with no rollout retries. Resguard recorded no new event (197 entries before and after). Peak RSS was 3,038,720 KiB; subsequent full runs require at least a 3.8 GB cap. The first asset-path probe and first scoring attempt failed before completion and are retained.</p>
<p>This two-run comparison measures one synthetic state-error realization and one pulse. Hardware friction, estimator slip response and actuator bandwidth remain unmeasured. The archived schedule has no recovery phase. Force scoring samples the full horizon at 10 Hz and can miss intermediate peaks.</p>
<p><a href="media/20261015-contact-audit/20261015-foot-motion.json">Metrics and decision</a> · <a href="media/20261015-contact-audit/20261016-foot-motion-raw.zip">Raw CSV, NPZ, contacts, commands and logs</a> · <a href="media/20261015-contact-audit/20261016-foot-motion-provenance.json">Plan/model hashes and cone rows</a> · <a href="media/20261015-contact-audit/20261016-foot-motion-run-record.json">Exact commands, failures and resource accounting</a> · <a href="media/20261015-contact-audit/20261016-foot-video-verification.json">Video checks</a> · <a href="../../studies/20261016_cmpc_foot_motion.py">Rollout script</a> · <a href="../../studies/20261016_cmpc_foot_analysis.py">Scoring/render script</a></p>
<h2>Future work</h2>
<p>The next contact-model comparison should first correct the opt-in sole geometry/friction representation and reject or fall back from tensile forces and violated wrench rows. It should reuse this nominal/pulse pair and require feasible node-zero commands plus lower estimated one-step residual. A subsequent sliding model must predict tangential motion and contact loss; reference updates alone cannot do this.</p>
<p>The motion-transfer checkpoint should compare the original startup against one command-realization or initialization correction before re-solving a longer or velocity-regularized approach. Measure first-0.5 s joint acceleration, wrist speed at first brace load, combined-input clipping and realized torque. Improvement requires smaller peaks and clipping error while retaining reach and support. Hardware validation needs synchronized command, encoder, IMU/base estimate and current/torque logs, plus independently measured sole motion and surface friction. Recovery still needs a separate release/abort experiment.</p>
</html>'''
REPORT.write_text(page)
index=ROOT/'docs/index.html';s=index.read_text();row='''<li><span class="d">10-16</span>
  <a href="lean/20261015-cmpc-foot-motion.html">Foot motion and command feasibility during a lateral wrist pulse</a>
  <span class="n">Two estimated-state rollouts: 78.46 mm left-foot motion, tensile predicted contact forces, input clipping and labelled 1× videos.</span></li>
'''
if 'lean/20261015-cmpc-foot-motion.html' not in s:
 marker='<li><span class="d">10-16</span>\n  <a href="lean/20261016-height-playback.html">'
 if marker not in s:marker='<li><span class="d">10-15</span>\n  <a href="lean/20261015-cmpc-surface-catalogue.html">'
 assert marker in s;s=s.replace(marker,row+marker,1);index.write_text(s)
print(json.dumps(dict(page=str(REPORT),archive_MB=archive.stat().st_size/1e6,numerical_s=resource['serial_numerical_wall_s'],jobs=len(jobs))))
