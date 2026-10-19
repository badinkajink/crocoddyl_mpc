#!/usr/bin/env python3
"""Saved-data scores and timestamped 1x rendering; all modes use resguard."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('recovery',ROOT/'studies/20261019_cmpc_adaptive_transfer.py')
R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
F,np,mj,OUT,MEDIA=R.F,R.np,R.mj,R.OUT,R.MEDIA
TAGS=['baseline-pulse','gated-pulse','baseline-nominal','gated-nominal']
OLD=ROOT/'studies/runs/20261018-cmpc-command-recovery'

def load(tag):
    stem=(OLD/('20261018-'+tag) if tag.startswith('baseline') else OUT/('20261019-'+tag))
    with Path(str(stem)+'.csv').open() as f:
        rd=csv.reader(x for x in f if not x.startswith('#'));cols=next(rd);values=np.array(list(rd),float)
    return stem,dict(zip(cols,values.T)),dict(np.load(str(stem)+'-raw.npz')),json.loads(Path(str(stem)+'-summary.json').read_text())

def achieved(t,condition,start=0):
    # Return first onset only after verifying 0.3 s of consecutive success.
    count=0;need=150
    for i,ok in enumerate(condition & (t>=start-1e-9)):
        count=count+1 if ok else 0
        if count>=need:return float(t[i-need+1])
    return None

def forces(rows,lo,hi):
    out={}
    for name in sorted(set(r['contact'] for r in rows)):
        rr=[r for r in rows if r['contact']==name and lo-1e-9<=r['t']<hi-1e-9]
        if not rr:continue
        w=np.array([r['wrench_world'] for r in rr]);v=np.array([r['cone_violation'] for r in rr])
        out[name]=dict(samples=len(rr),normal_min_N=float(w[:,2].min()),normal_max_N=float(w[:,2].max()),
            tangential_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),force_row_max_N=float(v[:,:5].max()),
            moment_row_max_Nm=float(v[:,5:].max()) if v.shape[1]>5 else None,
            violating_samples=int(np.sum(v.max(axis=1)>1e-6)))
    return out

def score(selection=False):
    global TAGS
    if selection:TAGS=TAGS[:2]
    m,_=F.cs.load(ik_margin=0);jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    result={};checks={};arrays={}
    for tag in TAGS:
        stem,c,a,s=load(tag);arrays[tag]=a;t=a['t']
        q=np.array([c['qpos'+str(i)] for i in range(m.nq)]).T
        v=np.array([c['qvel'+str(i)] for i in range(m.nv)]).T
        ctrl=np.array([c['ctrl'+str(i)] for i in range(m.nu)]).T
        af=np.array([c['afrc'+str(i)] for i in range(m.nu)]).T
        gates=json.loads(Path(str(stem)+'-gate.json').read_text())
        before=json.loads(Path(str(stem)+'-executable-before.json').read_text())
        after=json.loads(Path(str(stem)+'-executable-after.json').read_text())
        forecast=json.loads(Path(str(stem)+'-predicted-forces.json').read_text())
        table=json.loads(Path(str(stem)+'-table.json').read_text())
        # The scene's passive 0.3 kg object also rests on the table. Its 2.943 N
        # load is recorded for audit but is excluded from robot support.
        a['table_force_N']=np.array([sum(z for b,z in r['body_vertical_force_N'].items() if b!='object') for r in table])
        both=np.all(a['sole_w'][:,:,2]>5,axis=1)
        support=both&(a['table_force_N']>20)
        js=np.max(abs(v[:,va]),axis=1);ws=np.linalg.norm(a['point_v'][:,0],axis=1)
        quiet=(js<.1)&(ws<.03)
        target=np.array(s['target_changed_m']);err=np.linalg.norm(a['point_p'][:,1]-target,axis=1)
        ir=np.searchsorted(t,R.RELEASE)
        disp=1000*np.linalg.norm(a['sole_p']-a['sole_p'][ir],axis=2)
        slip=np.linalg.norm(a['sole_v'][:,:,:2],axis=2)
        still=np.all(disp<5,axis=1)&np.all(slip<.02,axis=1)
        stand=np.array(s['standing_reference'])
        upright=(np.sqrt(np.mean((q[:,qa]-stand[qa])**2,axis=1))<.1)&(abs(q[:,2]-stand[2])<.05)
        recovered=both&quiet&upright&(a['table_force_N']<5)&still
        abort=support&quiet&still
        brace_t=achieved(t,support)
        original_err=np.linalg.norm(a['point_p'][:,1]-np.array(s['target_original_m']),axis=1)
        original_reach_t=achieved(t,support&(original_err<.03)&(t<R.SHIFT))
        physical_release_t=achieved(t,both&(a['table_force_N']<5),R.RELEASE)
        reach_t=achieved(t,support&(err<.03)&(t<R.RELEASE),R.SHIFT)
        recover_t=achieved(t,recovered,R.RELEASE)
        abort_t=achieved(t,abort,R.RELEASE)
        phase={}
        for name,lo,hi in [('approach',0,4),('command',4,5.2),('recovery',5.2,9),('tail',8.5,9)]:
            mask=(t>=lo-1e-9)&(t<hi-1e-9);acc=np.diff(v[:,va],axis=0)/np.diff(t)[:,None];amask=mask[1:]&mask[:-1]
            ff={}
            for i,foot in enumerate(F.FEET):
                p=a['sole_p'][mask,i];vv=slip[mask,i];loaded=a['sole_w'][mask,i,2]>5
                ff[foot]=dict(displacement_from_phase_start_max_mm=float(1000*np.linalg.norm(p-p[0],axis=1).max()),
                    contact_loss_s=float(np.sum(~loaded)*.002),loaded_slip_max_m_s=float(vv[loaded].max()) if loaded.any() else None,
                    normal_min_N=float(a['sole_w'][mask,i,2].min()),normal_max_N=float(a['sole_w'][mask,i,2].max()),
                    rotation_from_phase_start_max_deg=float(max(F.rot_angle(a['sole_R'][np.flatnonzero(mask)[0],i],rr) for rr in a['sole_R'][mask,i])*180/np.pi))
            selected=[];other=[]
            for j in np.flatnonzero(mask):
                loads=table[j]['body_vertical_force_N']
                selected.append(sum(loads.get(b,0) for b in ['left_shoulder_yaw_link','left_elbow_link']))
                other.append(sum(z for b,z in loads.items() if b not in ['left_shoulder_yaw_link','left_elbow_link','object']))
            times=a['pred_t'];cmask=(times>=lo-1e-9)&(times<hi-1e-9)
            phase[name]=dict(joint_speed_peak_rad_s=float(js[mask].max()),joint_acceleration_peak_rad_s2=float(abs(acc[amask]).max()),
                wrist_speed_peak_m_s=float(ws[mask].max()),hand_speed_peak_m_s=float(np.linalg.norm(a['point_v'][mask,1],axis=1).max()),
                joint_tracking_rms_rad=float(np.sqrt(np.mean((a['q_des'][mask]-q[mask][:,qa])**2))),
                target_error_rms_mm=float(1000*np.sqrt(np.mean(err[mask]**2))),feet=ff,
                table_load_min_N=float(a['table_force_N'][mask].min()),table_load_max_N=float(a['table_force_N'][mask].max()),
                selected_body_load_min_N=float(min(selected)),unselected_body_load_max_N=float(max(other)),
                solve_p95_ms=float(np.percentile(1000*a['pred_solve_s'][cmask],95)),
                policy_p95_ms=float(np.percentile(a['pred_policy_ms'][cmask],95)),
                policy_max_ms=float(a['pred_policy_ms'][cmask].max()),policy_over_20ms=int(np.sum(a['pred_policy_ms'][cmask]>20)),commands=int(cmask.sum()),
                executable_forces=forces(after,lo,hi),unfiltered_executable_forces=forces(before,lo,hi),horizon_forces=forces(forecast,lo,hi))
        clipped=np.where(m.actuator_ctrllimited,np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),ctrl)
        realized=m.actuator_gainprm[:,0]*(clipped-q[:,qa])+m.actuator_biasprm[:,2]*v[:,va]
        realized=np.where(m.actuator_forcelimited,np.clip(realized,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]),realized)
        torque_error=float(abs(realized-af).max());checks[tag+'_servo_identity']=torque_error<1e-8
        checks[tag+'_horizon']=all(z['T']==35 for z in s['window_checks'])
        checks[tag+'_dt']=bool(np.allclose(np.diff(t),.002))
        checks[tag+'_finite']=bool(np.isfinite(q).all() and np.isfinite(v).all())
        checks[tag+'_periods']=len(gates)==len(a['pred_t'])
        last=t>=8.7-1e-9
        result[tag]=dict(summary=s,phases=phase,timing=dict(brace_onset_s=brace_t,
            original_reach_onset_s=original_reach_t,physical_brace_release_onset_s=physical_release_t,
            reach_onset_s=reach_t,reach_after_command_s=None if reach_t is None else max(0.,reach_t-R.SHIFT),
            standing_recovery_onset_s=recover_t,standing_recovery_after_request_s=None if recover_t is None else recover_t-R.RELEASE,
            supported_rest_onset_s=abort_t,supported_rest_after_request_s=None if abort_t is None else abort_t-R.RELEASE),
            complete_cycle=bool(recover_t is not None and reach_t is not None and np.all(recovered[last])),
            final_supported_rest=bool(np.all(abort[last])),final_standing=bool(np.all(recovered[last])),
            post_release_max_sole_displacement_mm=float(disp[t>=R.RELEASE].max()),
            initial_servo_reconstruction_error_Nm=torque_error,
            gate_status_counts={z:sum(r['status']==z for r in gates) for z in sorted(set(r['status'] for r in gates))},
            fallback_commands=sum('fallback' in r for r in gates),
            command_response=dict(hand_x_motion_first_half_second_mm=float(1000*(a['point_p'][np.searchsorted(t,4.5),1,0]-a['point_p'][np.searchsorted(t,4.),1,0])),
                error_at_command_mm=float(1000*err[np.searchsorted(t,4.)]),error_before_pulse_mm=float(1000*err[np.searchsorted(t,4.5)-1]),
                note='The 30 mm reach screen is wider than the 20 mm target step; zero acquisition time does not measure tracking response.'),
            final_command_excluded_from_motion='Command at 9.02 s precedes shutdown only; force phase scores stop at 9 s.',
            video=('20261018-recovery-' if tag.startswith('baseline') else '20261019-adaptive-')+tag+'.mp4')
    for case in (['pulse'] if selection else ['nominal','pulse']):
        a,b=arrays['baseline-'+case],arrays['gated-'+case];mask=a['pred_t']<R.RELEASE-1e-9
        checks[case+'_identical_before_release']=bool(np.array_equal(a['pred_x_est'][mask],b['pred_x_est'][mask]))
        checks[case+'_same_provenance']=result['baseline-'+case]['summary']['provenance']==result['gated-'+case]['summary']['provenance']
    passed=all(result['gated-'+c]['final_standing'] and
        result['gated-'+c]['post_release_max_sole_displacement_mm']<5 for c in (['pulse'] if selection else ['nominal','pulse']))
    activations=sum((z['summary']['transition'] is None or z['summary']['transition']['t']>R.RELEASE+1e-9) for k,z in result.items() if k.startswith('gated'))
    out=dict(checkpoint='adaptive_transfer',date='2026-10-19',protocol=R.protocol(),cases=result,checks=checks,
        decision=dict(status='selection_frozen' if selection else 'measurement_complete',hypothesis_passed=bool(passed),
            guard_activations=activations,motion_gate_test='delayed_release' if activations else 'ready_at_requested_release',adopt_default=False,
            complete_cycles=sum(z['complete_cycle'] for z in result.values()),new_rollouts=1 if selection else 2,reused_baseline_rollouts=1 if selection else 2,
            limitation='First-command feasibility cannot certify a release trajectory or forces between updates. Contacts remain prescribed. Synthetic estimated-state path is not a deployed EKF. Standing recovery and hardware require separate validation.'),
        frames='Predicted forces: world-aligned at contact origin, first five cone rows N, remaining foot rows N m. Physical foot wrenches at sole origin; individual contact friction recorded by MuJoCo. Table forces are physical body loads; original site geometry remains uncertain.',
        source_hashes={str(p.relative_to(ROOT)):F.sha(p) for p in [Path(__file__),ROOT/'studies/20261019_cmpc_adaptive_transfer.py',ROOT/'studies/20261017_cmpc_foot_model_transfer.py']})
    F.dump(MEDIA/('20261019-adaptive-selection.json' if selection else '20261016-cmpc-adaptive-transfer.json'),out)
    print(json.dumps(dict(decision=out['decision'],checks=checks,timing={k:v['timing'] for k,v in result.items()})),flush=True)
    assert all(checks.values()),checks

def render():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from height_video import frame_samples
    m,d=F.cs.load(ik_margin=0);W,H=800,450;m.vis.global_.offwidth=W;m.vis.global_.offheight=H
    ren=mj.Renderer(m,H,W);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
    cam.azimuth,cam.elevation,cam.distance=140,-12,3.4;cam.lookat[:]=[.55,-.1,.95]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    for tag in ['gated-pulse','gated-nominal']:
        stem,c,a,s=load(tag);t=a['t'];requested,idx=frame_samples(t,30,1)
        path=MEDIA/('20261019-adaptive-'+tag+'.mp4');frames=[]
        selected={int(np.argmin(abs(t[idx]-tt))) for tt in [0,.13,2.4,4.5,5.2,6.5,8,9]}
        with imageio.get_writer(path,fps=30,macro_block_size=1,ffmpeg_params=['-threads','1']) as writer:
            for vi,k in enumerate(idx):
                d.qpos[:]=[c['qpos'+str(j)][k] for j in range(m.nq)];mj.mj_forward(m,d)
                ren.update_scene(d,cam);im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,W,60),fill=(12,18,25))
                draw.text((8,4),f'{tag} | Simulation t = {t[k]:.2f} s | Playback 1x',font=font,fill='white')
                label='Approach' if t[k]<4 else '+20 mm reach' if t[k]<5.2 else ('Waiting for low estimated motion' if s['transition'] is None or t[k]<s['transition']['t'] else s['transition']['decision'])
                if 'pulse' in tag and 4.5<=t[k]<4.8:label+=' | 90 N +y pulse ON'
                draw.text((8,31),label,font=font,fill='white');writer.append_data(np.asarray(im))
                if vi in selected:frames.append(im.resize((400,225)))
        strip=Image.new('RGB',(1600,450),'white')
        for i,im in enumerate(frames):strip.paste(im,((i%4)*400,(i//4)*225))
        strip.save(path.with_suffix('.jpg'))
        F.dump(path.with_suffix('.json'),dict(source=str(stem)+'.csv',source_sha256=F.sha(str(stem)+'.csv'),
            fps=30,playback_speed=1,frames=len(idx),duration_s=len(idx)/30,simulation_end_s=float(t[-1]),
            max_timestamp_error_s=float(max(abs(t[idx]-requested))),video_sha256=F.sha(path)))
    ren.close()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selection','score','render']);a=p.parse_args()
    render() if a.mode=='render' else score(a.mode=='selection')
