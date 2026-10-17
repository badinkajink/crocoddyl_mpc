#!/usr/bin/env python3
"""Score saved motion-transfer states and render labelled 1x videos.

No integration occurs. Ground-truth saved states are used only for scoring.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('motion',ROOT/'studies/20261016_cmpc_motion_transfer.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
F,np,mj,cs=M.F,M.np,M.mj,M.cs
OUT,MEDIA=M.OUT,M.MEDIA
CASES=['baseline','fixed-position','velocity']


def score():
    m,d=cs.load(ik_margin=0);results={};arrays={}
    feet=list(F.FEET);fb={cs.bid(m,F.cb.SOLES[f][0]):f for f in feet}
    for case in CASES:
        stem=OUT/('20261016-'+case);c,a=M.load(stem);arrays[case]=(c,a)
        t=a['t'];last=t>=t[-1]-1.+1e-9
        ss=json.loads(Path(str(stem)+'-startup.json').read_text())
        rows=json.loads(Path(str(stem)+'-table-contacts.json').read_text())
        first=next((r for r in rows if r['normal_N']>1),None)
        first_wrist=next((r for r in rows if r['normal_N']>1 and r['body']=='left_wrist_yaw_link'),None)
        for obj in [first,first_wrist]:
            if obj is not None:
                k=int(np.argmin(abs(t-obj['t'])))
                obj['wrist_speed_m_s']=float(np.linalg.norm(a['point_v'][k,0]))
        # Material-point floor slip and actual support onset from saved states.
        slip={f:0. for f in feet};slip_hold={f:0. for f in feet};floor_onset={f:None for f in feet}
        force_native=0.;sim_normal_min=None;sim_tangent_max=0.;slip_at_accel={}
        jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));ff=np.zeros(6)
        q=np.array([c['qpos'+str(i)] for i in range(m.nq)]).T
        v=np.array([c['qvel'+str(i)] for i in range(m.nv)]).T
        ctrl=np.array([c['ctrl'+str(i)] for i in range(m.nu)]).T
        for k,stamp in enumerate(t):
            d.qpos[:]=q[k];d.qvel[:]=v[k];d.ctrl[:]=ctrl[k];mj.mj_forward(m,d)
            for ci in range(d.ncon):
                con=d.contact[ci];bs=m.geom_bodyid[con.geom]
                b=next((int(b) for b in bs if b in fb),None)
                if b is None or not any(m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE for g in con.geom):continue
                mj.mj_contactForce(m,d,ci,ff);f=fb[b]
                if ff[0]>1 and floor_onset[f] is None:floor_onset[f]=float(stamp)
                dim=con.dim;mu=con.friction[:dim-1]
                scaled=np.divide(ff[1:dim],mu,out=np.zeros(dim-1),where=mu>0)
                native=(np.linalg.norm(scaled) if m.opt.cone==mj.mjtCone.mjCONE_ELLIPTIC else np.sum(abs(scaled)))-ff[0]
                force_native=max(force_native,float(native));sim_normal_min=float(ff[0]) if sim_normal_min is None else min(sim_normal_min,float(ff[0]))
                sim_tangent_max=max(sim_tangent_max,float(np.linalg.norm(ff[1:3])))
                if ff[0]<=5:continue
                mj.mj_jac(m,d,jp,jr,con.pos,b)
                speed=float(np.linalg.norm((jp@d.qvel)[:2]));slip[f]=max(slip[f],speed)
                if last[k]:slip_hold[f]=max(slip_hold[f],speed)
        ss.update(first_table_contact=first,first_wrist_contact=first_wrist,
            first_loaded_floor_s=floor_onset,loaded_floor_slip_peak_m_s=slip,
            final_second_loaded_floor_slip_peak_m_s=slip_hold,
            simulated_floor_native_cone_excess_max_N=force_native,
            simulated_floor_normal_min_N=sim_normal_min,simulated_floor_tangent_max_N=sim_tangent_max,
            final_second_table_normal_min_N=float(a['table_normal_N'][last].min()),
            final_second_table_normal_mean_N=float(a['table_normal_N'][last].mean()),
            table_lateral_cone_excess_max_N=max([r['circular_cone_excess_N'] for r in rows],default=0.),
            final_second_table_force_by_body_N={b:float(sum(r['normal_N'] for r in rows if r['body']==b and r['t']>=t[-1]-1.+1e-9)/sum(last)) for b in sorted({r['body'] for r in rows})},
            final_second_reach_rms_mm=float(1000*np.sqrt(np.mean(np.sum((a['point_p'][last,1]-[1.06,-.2348,.9982])**2,axis=1)))),
            initial_qpos=q[0].tolist(),initial_qvel=v[0].tolist())
        if case!='fixed-position':
            summary=json.loads(Path(str(stem)+'-summary.json').read_text())
            ss['rollout']=summary
        results[case]=ss
    b,r=results['baseline'],results['velocity']
    comparison={key:dict(baseline=b[key],velocity=r[key],change_percent=100*(r[key]/b[key]-1)) for key in
        ['startup_acceleration_rad_s2','startup_joint_speed_rad_s','startup_wrist_peak_m_s','q_tracking_rms_rad','v_tracking_rms_rad_s','final_second_reach_rms_mm']}
    feet_ok=all(r['rollout']['feet'][f]['displacement_from_start_max_mm']<=b['rollout']['feet'][f]['displacement_from_start_max_mm']+5 for f in feet)
    support_ok=r['final_second_table_normal_min_N']>=5
    reach_ok=r['final_second_reach_rms_mm']<=b['final_second_reach_rms_mm']+5
    acceleration_ok=r['startup_acceleration_rad_s2']<=.9*b['startup_acceleration_rad_s2']
    decision=dict(acceleration_benefit=acceleration_ok,retains_table_support=support_ok,reach_within_5mm=reach_ok,
        sole_displacement_within_5mm=feet_ok,passes_predeclared_combined_criterion=all([acceleration_ok,support_ok,reach_ok,feet_ok]),
        initial_states_identical=bool(np.array_equal(b['initial_qpos'],r['initial_qpos']) and np.array_equal(b['initial_qvel'],r['initial_qvel'])),
        model_array_hashes_identical=b['rollout']['provenance']['model_arrays_sha256']==r['rollout']['provenance']['model_arrays_sha256'],
        recovery='Unavailable: both plans have n_return=0; no recovery claim.',
        hardware='No hardware data or commands. Synthetic base-state error, ideal joint/velocity observations and synchronous free-running simulation. Hardware actuator bandwidth and supported handover require measurement.')
    # Confirm baseline reproduction against previously saved measured state.
    oldc,olda=M.load(ROOT/'studies/runs/20261016-cmpc-foot-motion/20261016-nominal')
    decision['baseline_previous_qvel_max_abs_difference']=float(max(abs(arrays['baseline'][0][k]-oldc[k]).max() for k in oldc if k.startswith('qvel')))
    data=dict(checkpoint='motion_transfer',date='2026-10-16 America/Denver',rollouts_started=3,rollouts_completed=3,rollout_retries=0,
        hypothesis=json.loads((OUT/'20261016-pre-intervention-decision.json').read_text()),
        cases=results,comparison=comparison,decision=decision,
        metrics=dict(startup='First 0.5 s; peak joint acceleration is adjacent velocity finite difference over 2 ms.',
            wrist='World translational speed of left wrist-yaw body origin, J(q) v. Contact onset: first physical witness normal >1 N.',
            tracking='RMS over 27 actuated joints and first 0.5 s; logged command minus sampled joint state.',
            slip='World-horizontal speed of foot material point at a floor contact witness with normal >5 N.',
            support='Sum of robot-table contact normal force, measured separately from prescribed predictor forces.',
            forces='Predicted world-aligned force at contact origin; cone rows 0..4 N, moment rows N m. MuJoCo force in contact frame; native cone includes torsion and rolling.',
            joint_margin='Minimum distance of actual joint position to either MJCF bound, radians; negative is a violation.',
            runtime='One solver thread, one BoxFDDP iteration, 35 nodes at 0.02 s; latency compensation disabled. Percentiles cover full 6 s simulation.'),
        source_scripts=['studies/20261016_cmpc_motion_transfer.py','studies/20261016_cmpc_motion_analysis.py','studies/20261016_cmpc_foot_motion.py'])
    F.dump(MEDIA/'20261016-cmpc-motion-transfer.json',data)
    print(json.dumps(dict(comparison=comparison,decision=decision)),flush=True)


def render():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from height_video import frame_samples
    m,d=cs.load(ik_margin=0);W,H=960,540;m.vis.global_.offwidth=W;m.vis.global_.offheight=H
    ren=mj.Renderer(m,H,W);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
    cam.azimuth,cam.elevation,cam.distance=140.,-12.,3.4;cam.lookat[:]=[.55,-.1,.95]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for case in CASES:
        stem=OUT/('20261016-'+case);c,a=M.load(stem);t=a['t'];req,idx=frame_samples(t,30,1)
        out=MEDIA/('20261016-motion-'+case+'.mp4');images=[]
        targets=[0,.13,.3,.48] if case=='fixed-position' else [0,.13,.5,1.,2.4,4.,6.]
        selected={int(np.argmin(abs(t[idx]-tt))) for tt in targets}
        with imageio.get_writer(out,fps=30,macro_block_size=1,ffmpeg_params=['-threads','1']) as w:
            for vi,k in enumerate(idx):
                d.qpos[:]=[c['qpos'+str(i)][k] for i in range(m.nq)];mj.mj_forward(m,d)
                ren.update_scene(d,cam);im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,W,38),fill=(12,18,25))
                draw.text((10,5),f'{case} | Simulation t = {t[k]:.2f} s | Playback 1x | Table 0.885 m',font=font,fill='white')
                w.append_data(np.asarray(im))
                if vi in selected:images.append(im.resize((480,270)))
        strip=Image.new('RGB',(1920,540),(12,18,25))
        for i,im in enumerate(images):strip.paste(im,((i%4)*480,(i//4)*270))
        strip.save(out.with_name(out.stem+'-frames.jpg'))
        F.dump(out.with_suffix('.json'),dict(source=str(stem)+'.csv',source_sha256=F.sha(str(stem)+'.csv'),
            fps=30,playback_speed=1,frames=len(idx),duration_s=len(idx)/30,simulation_start_s=float(t[0]),
            simulation_end_s=float(t[-1]),max_timestamp_error_s=float(max(abs(t[idx]-req))),video_sha256=F.sha(out)))
        print(str(out),flush=True)
    ren.close()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');a=ap.parse_args()
    render() if a.render else score()
