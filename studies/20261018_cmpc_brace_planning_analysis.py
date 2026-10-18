#!/usr/bin/env python3
"""Saved-state physical support scoring and labelled 1x rendering; no rollout."""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('planning',ROOT/'studies/20261018_cmpc_brace_planning.py');P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
F,np,mj,cs=P.F,P.np,P.mj,P.cs
OUT,MEDIA=P.OUT,P.MEDIA

def read(stem):
    with Path(str(stem)+'.csv').open() as f:
        r=csv.reader(l for l in f if not l.startswith('#'));head=next(r);rows=np.array(list(r),float)
    return head,rows,dict(np.load(str(stem)+'-raw.npz'))

def score(strategy,target):
    design=json.loads((OUT/'20261018-design.json').read_text());spec=design['strategies'][strategy]
    stem=OUT/('20261018-'+strategy+'-'+target);summary=json.loads(Path(str(stem)+'-summary.json').read_text())
    head,rows,a=read(stem);t=a['t'];hold=(t>=4-1e-9)&(t<6-1e-9)
    m,d=cs.load(ik_margin=0);geo=P.Geometry(m)
    cols=lambda p:[i for i,h in enumerate(head) if h.startswith(p)]
    qi,vi,ci=cols('qpos'),cols('qvel'),cols('ctrl');jids=m.actuator_trnid[:,0];va=m.jnt_dofadr[jids]
    velocity=rows[:,vi][:,va];motion=float(np.sum(np.sum(velocity[:-1]**2,axis=1)*np.diff(t)))
    reach=np.linalg.norm(a['point_p'][:,1]-summary['provenance']['plan_fields']['target'],axis=1)*1000
    supports=[];all_supports=[];body_support=[];records=[];model_points=[]
    for k in np.flatnonzero(hold)[::10]:
        d.qpos[:]=rows[k,qi];d.qvel[:]=rows[k,vi];d.ctrl[:]=rows[k,ci];mj.mj_forward(m,d)
        rr=geo.contacts(d,spec['sites'])
        for r in rr:r['t']=float(t[k])
        records.extend(rr)
        credited=[r for r in rr if r['is_brace'] and r['closed'] and r['region_member'] and any(r['geom_id'] in P.GEOMS[s] for s in spec['subset'])]
        supports.append(sum(r['normal_N'] for r in credited))
        by={};total=0
        for i in range(d.ncon):
            c=d.contact[i]
            if 99 not in c.geom:continue
            g=int(c.geom1 if c.geom2==99 else c.geom2);b=int(m.geom_bodyid[g]);name=P.G.S.name(m,'BODY',b)
            if 'object' in name:continue
            ff=np.zeros(6);mj.mj_contactForce(m,d,i,ff);by[name]=by.get(name,0)+float(ff[0]);total+=ff[0]
        body_support.append(by);all_supports.append(float(total))
        for site in spec['subset']:
            v=spec['sites'][site];pw=cs.point_world(m,d,v['body'],v['point_body_m'])
            candidates=[r for r in rr if r['is_brace'] and r['geom_id'] in P.GEOMS[site] and r['closed'] and r['region_member']]
            nearest=min(candidates,key=lambda r:r['sites'][site]['point_error_mm']) if candidates else None
            model_points.append(dict(t=float(t[k]),site=site,point_world_m=pw.tolist(),
                height_mm=float(1000*(pw[2]-cs.table_top_z(m,d))),
                nearest_valid_witness=nearest['sites'][site] if nearest else None))
    forces=json.loads(Path(str(stem)+'-predicted-forces.json').read_text())
    ff=[r for r in forces if 4-1e-9<=r['t']<6-1e-9]
    force_ex=max(max(r['cone_violation'][:5]) for r in ff)
    moment_ex=max([max(r['cone_violation'][5:]) for r in ff if len(r['cone_violation'])>5] or [0])
    by_contact={}
    for c in sorted({r['contact'] for r in ff}):
        group=[r for r in ff if r['contact']==c]
        by_contact[c]=dict(normal_min_N=min(r['wrench_world'][2] for r in group),normal_max_N=max(r['wrench_world'][2] for r in group),
            cone_force_max_N=max(max(r['cone_violation'][:5]) for r in group),
            tangent_max_N=max(float(np.linalg.norm(r['wrench_world'][:2])) for r in group))
    feet={}
    for i,f in enumerate(F.FEET):
        loaded=hold&(a['sole_w'][:,i,2]>1)
        feet[f]=dict(hold_contact_loss_ms=float(np.sum(hold&(a['loaded_contacts'][:,i]==0))*m.opt.timestep*1000),
            loaded_hold_slip_m_s=float(np.linalg.norm(a['sole_v'][loaded,i,:2],axis=1).max()) if loaded.any() else None,
            hold_displacement_mm=float(1000*np.linalg.norm(a['sole_p'][hold,i]-a['sole_p'][np.flatnonzero(hold)[0],i],axis=1).max()),
            episode_displacement_mm=summary['feet'][f]['displacement_from_start_max_mm'])
    rms=float(np.sqrt(np.mean(reach[hold]**2)));support=float(min(supports));loss=max(f['hold_contact_loss_ms'] for f in feet.values())
    reasons=[]
    if rms>30:reasons.append('Reach RMS exceeds 30 mm')
    if support<20:reasons.append('CAD-valid selected-body support below 20 N')
    if loss>20:reasons.append('Hold floor-contact loss exceeds 20 ms')
    if force_ex>1:reasons.append('Predicted cone force excess exceeds 1 N')
    if moment_ex>.1:reasons.append('Predicted cone moment excess exceeds 0.1 N m')
    raw_score=rms/30+20/support+motion/10 if support>0 else None
    outcome=dict(strategy=strategy,target=target,available=not reasons,rejection_reasons=reasons,
        selection_score=raw_score if not reasons else None,raw_cost_without_feasibility_gate=raw_score,
        reach_hold_rms_mm=rms,reach_final_mm=float(reach[-1]),motion_integral_rad2_per_s=motion,
        credited_support_min_N=support,credited_support_max_N=float(max(supports)),total_table_support_min_N=min(all_supports),
        predicted_hold_force_excess_N=force_ex,predicted_hold_moment_excess_Nm=moment_ex,predicted_hold_by_contact=by_contact,
        feet=feet,solve_p95_ms=summary['solve_p95_ms'],solve_above_20ms=summary['solve_above_20ms'],
        motion=summary['motion'],joint_violation_max_rad=summary['joint_violation_max_rad'],
        body_support_mean_N={n:float(np.mean([r.get(n,0) for r in body_support])) for n in sorted({n for r in body_support for n in r})},
        physical_friction_excess_max_N=max(r['physical_cone_excess_N'] for r in records),
        model_point_witness_errors={s:dict(
            no_valid_closed_witness_samples=sum(r['nearest_valid_witness'] is None for r in model_points if r['site']==s),
            max_nearest_witness_distance_mm=max([r['nearest_valid_witness']['point_error_mm'] for r in model_points if r['site']==s and r['nearest_valid_witness']] or [0]),
            max_100N_moment_error_Nm=max([r['nearest_valid_witness']['moment_error_per_100N_Nm'] for r in model_points if r['site']==s and r['nearest_valid_witness']] or [0])) for s in spec['subset']},
        source_csv_sha256=F.sha(str(stem)+'.csv'),source_plan_sha256=F.sha(OUT/(strategy+'-'+target)/'plan_elbow_forearm.json'),
        horizon=summary['horizon_min'],dt_s=summary['provenance']['dt_s'],observation=summary['observation'])
    F.dump(str(stem)+'-physical.json',dict(outcome=outcome,contacts=records,model_points=model_points,
        credited_support_N=supports,total_table_support_N=all_supports))
    print(strategy,target,json.dumps({k:outcome[k] for k in ['available','rejection_reasons','reach_hold_rms_mm','credited_support_min_N','predicted_hold_force_excess_N','raw_cost_without_feasibility_gate']}),flush=True)
    return outcome

def analysis(target):
    design=json.loads((OUT/'20261018-design.json').read_text())
    outcomes={s:score(s,target) for s in design['compared']}
    F.dump(OUT/('20261018-'+target+'-scores.json'),outcomes)
    if target=='selection':
        path=OUT/'20261018-frozen-selection.json'
        assert not path.exists(), 'Selection is immutable after first scoring'
        feasible=[s for s,v in outcomes.items() if v['available']]
        winner=min(feasible,key=lambda s:outcomes[s]['selection_score']) if feasible else None
        F.dump(path,dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),selected_strategy=winner,
            fixed_baseline='elbow_forearm',fallback_when_none='No feasible strategy; retain baseline only as experimental comparator.',
            selection_scores_sha256=F.sha(OUT/'20261018-selection-scores.json'),protocol_sha256=F.sha(OUT/'20261018-protocol.json'),
            reserved_rollouts_exist=any(OUT.glob('*-reserved.csv'))))
    else:
        frozen=json.loads((OUT/'20261018-frozen-selection.json').read_text());winner=frozen['selected_strategy']
        base=outcomes['elbow_forearm'];selected=outcomes.get(winner)
        gain=(1-selected['selection_score']/base['selection_score'])*100 if selected and selected['available'] and base['available'] else None
        F.dump(OUT/'20261018-decision.json',dict(selected_strategy=winner,reserved_score_improvement_percent=gain,
            hypothesis_supported=bool(gain is not None and gain>=10),
            reason='Both targets evaluated; invalid candidates receive no feasible score.' if gain is None else 'Frozen selection evaluated on reserved target.',
            frozen_selection_sha256=F.sha(OUT/'20261018-frozen-selection.json')))

def render():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    import height_video as hv
    import croco_replay as cr
    design=json.loads((OUT/'20261018-design.json').read_text());videos=[];scene_contacts=[]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for target in ['selection','reserved']:
        for strategy in design['compared']:
            stem=OUT/('20261018-'+strategy+'-'+target)
            if not Path(str(stem)+'.csv').exists():continue
            head,rows,a=read(stem);t=a['t'];qi=[i for i,h in enumerate(head) if h.startswith('qpos')];vi=[i for i,h in enumerate(head) if h.startswith('qvel')];ci=[i for i,h in enumerate(head) if h.startswith('ctrl')]
            req,ix=hv.frame_samples(t,30,1.);m,d=cs.load(ik_margin=0)
            cr.show_gripper(m);m.vis.global_.offwidth=960;m.vis.global_.offheight=540
            ren=mj.Renderer(m,540,960);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam);cam.azimuth=140;cam.elevation=-12;cam.distance=3.4;cam.lookat[:]=[.55,-.1,.95]
            name='20261018-brace-'+strategy+'-'+target;path=MEDIA/(name+'.mp4');frames=[];spec=design['strategies'][strategy]
            with imageio.get_writer(path,fps=30,macro_block_size=1,ffmpeg_params=['-threads','1']) as writer:
                for j,k in enumerate(ix):
                    d.qpos[:]=rows[k,qi];d.qvel[:]=rows[k,vi];d.ctrl[:]=rows[k,ci];mj.mj_forward(m,d);ren.update_scene(d,cam)
                    for site in spec['subset']:
                        v=spec['sites'][site];pw=cs.point_world(m,d,v['body'],v['point_body_m']);cr._marker(ren.scene,pw,(1.,.25,.15,1.),.009)
                    for con in d.contact:
                        if 99 in con.geom:cr._marker(ren.scene,con.pos,(.1,.9,.9,1.),.005)
                    im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,40),fill=(15,20,27));draw.text((10,9),f'{strategy} | {target} | Simulation t = {t[k]:.2f} s | Playback 1x',font=font,fill='white');writer.append_data(np.asarray(im))
                    if j in [0,15,72,120,150,180]:frames.append(im.resize((480,270)))
            sheet=Image.new('RGB',(1440,540))
            for j,im in enumerate(frames):sheet.paste(im,((j%3)*480,(j//3)*270))
            sheet.save(MEDIA/(name+'.png'))
            # Same state, enlarged brace scene, with witness and model points.
            k=int(np.argmin(abs(t-5)));d.qpos[:]=rows[k,qi];d.qvel[:]=rows[k,vi];d.ctrl[:]=rows[k,ci];mj.mj_forward(m,d)
            # Retain all robot/table contacts, including unselected bodies that
            # the candidate-only CAD query intentionally cannot credit.
            for i in range(d.ncon):
                con=d.contact[i]
                if 99 not in con.geom:continue
                g=int(con.geom1 if con.geom2==99 else con.geom2);b=int(m.geom_bodyid[g]);body=P.G.S.name(m,'BODY',b)
                if 'object' in body:continue
                ff=np.zeros(6);mj.mj_contactForce(m,d,i,ff);normal=con.frame[:3]*(1 if con.geom1==g else -1)
                pw=con.pos-.5*con.dist*normal;rot=d.xmat[b].reshape(3,3)
                scene_contacts.append(dict(strategy=strategy,target=target,t=float(t[k]),body=body,body_id=b,geom_id=g,
                    geom_name=P.G.S.name(m,'GEOM',g),normal_N=float(ff[0]),point_world_m=pw.tolist(),
                    point_body_m=(rot.T@(pw-d.xpos[b])).tolist(),normal_body=(rot.T@normal).tolist(),
                    lever_arm_body_m=(rot.T@(pw-d.xpos[b])).tolist(),signed_gap_mm=float(con.dist*1000),
                    candidate_geom=g in P.SURFACES))
            cam.distance=1.05;cam.azimuth=120;cam.elevation=-8;cam.lookat[:]=[.68,.25,.88];ren.update_scene(d,cam)
            for site in spec['subset']:
                v=spec['sites'][site];cr._marker(ren.scene,cs.point_world(m,d,v['body'],v['point_body_m']),(1.,.25,.15,1.),.006)
            for con in d.contact:
                if 99 in con.geom:cr._marker(ren.scene,con.pos,(.1,.9,.9,1.),.004)
            im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,40),fill=(15,20,27));draw.text((10,9),f'{strategy} | {target} | t = 5.00 s | Red: model point, cyan: contact',font=font,fill='white');im.save(MEDIA/(name+'-scene.png'));ren.close()
            videos.append(dict(path=str(path.relative_to(ROOT)),sha256=F.sha(path),source_sha256=F.sha(str(stem)+'.csv'),frames=len(ix),fps=30,duration_s=len(ix)/30,speed=1.,sample_error_s=float(max(abs(t[ix]-req)))))
    F.dump(OUT/'20261018-videos.json',videos)
    F.dump(OUT/'20261018-scene-contacts.json',scene_contacts)

def verify():
    import subprocess
    from PIL import Image,ImageDraw,ImageFont
    import croco_replay as cr
    checks=[];initial=[];models=[]
    def check(name,value):
        checks.append(dict(name=name,passed=bool(value)))
        assert value,name
    design=json.loads((OUT/'20261018-design.json').read_text())
    frozen=json.loads((OUT/'20261018-frozen-selection.json').read_text())
    check('Selection frozen before reserved rollouts',not frozen['reserved_rollouts_exist'])
    check('Selection input hash unchanged',frozen['selection_scores_sha256']==F.sha(OUT/'20261018-selection-scores.json'))
    check('Protocol unchanged',frozen['protocol_sha256']==F.sha(OUT/'20261018-protocol.json'))
    for target in ['selection','reserved']:
        for strategy in design['compared']:
            stem=OUT/('20261018-'+strategy+'-'+target);summary=json.loads(Path(str(stem)+'-summary.json').read_text())
            head,rows,a=read(stem);initial.append(rows[0,[i for i,h in enumerate(head) if h.startswith('qpos')]])
            models.append(summary['provenance']['model_arrays_sha256'])
            tag=strategy+' '+target
            check(tag+' 35 nodes at 20 ms',summary['provenance']['horizon']==35 and summary['provenance']['dt_s']==.02 and all(r['T']==35 for r in summary['window_checks']))
            check(tag+' estimated-state contract',summary['observation']['path']=='MuJoCoPlant.read -> default_sense(seed=17) -> State -> policy')
            check(tag+' physics episode',abs(summary['simulation_end_s']-6.018)<1e-6 and len(a['t'])==3010)
            physical=json.loads(Path(str(stem)+'-physical.json').read_text());rec=physical['contacts'];valid=[]
            for t in sorted({r['t'] for r in rec}):
                valid.append(sum(r['normal_N'] for r in rec if r['t']==t and r['closed'] and r['region_member'] and r['is_brace'] and any(r['geom_id'] in P.GEOMS[s] for s in design['strategies'][strategy]['subset'])))
            check(tag+' physical support reconstruction',np.max(abs(np.array(valid)-physical['credited_support_N']))<1e-9)
            # MuJoCo manifold witnesses have small plane reconstruction errors;
            # apply the same 0.5 mm geometry tolerance used by the catalogue.
            check(tag+' table witness plane within 0.5 mm',max(abs(r['plane_error_mm']) for r in rec if r['is_brace'])<.5)
            check(tag+' source CSV hash',physical['outcome']['source_csv_sha256']==F.sha(str(stem)+'.csv'))
            # Transparent still views expose the candidate points and witnesses
            # hidden by both grippers in the first opaque scene renders.
            m,d=cs.load(ik_margin=0);cr.show_gripper(m)
            cr.ghost_arm(m,arm='left',alpha=.22);cr.ghost_arm(m,arm='right',alpha=.08)
            qi=[i for i,h in enumerate(head) if h.startswith('qpos')];k=int(np.argmin(abs(a['t']-5)))
            d.qpos[:]=rows[k,qi];mj.mj_forward(m,d);m.vis.global_.offwidth=960;m.vis.global_.offheight=540
            ren=mj.Renderer(m,540,960);cam=mj.MjvCamera();mj.mjv_defaultCamera(cam)
            cam.distance=1.2;cam.azimuth=100;cam.elevation=-30;cam.lookat[:]=[.65,.27,.90];ren.update_scene(d,cam)
            spec=design['strategies'][strategy]
            for site in spec['subset']:
                v=spec['sites'][site];cr._marker(ren.scene,cs.point_world(m,d,v['body'],v['point_body_m']),(1.,.15,.05,1.),.009)
            for con in d.contact:
                if 99 in con.geom:cr._marker(ren.scene,con.pos,(.05,.85,.9,1.),.006)
            im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,40),fill=(15,20,27))
            draw.text((10,9),f'{strategy} | {target} | t = 5.00 s | Red: model point; cyan: contact',font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18),fill='white')
            im.save(MEDIA/('20261018-brace-'+strategy+'-'+target+'-scene.png'));ren.close()
    check('Same initial physical state',all(np.max(abs(x-initial[0]))==0 for x in initial))
    check('Same plant arrays',all(x==models[0] for x in models))
    for v in json.loads((OUT/'20261018-videos.json').read_text()):
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=nb_frames,avg_frame_rate,duration','-of','json',str(ROOT/v['path'])]))['streams'][0]
        check(v['path']+' hash',v['sha256']==F.sha(ROOT/v['path']))
        check(v['path']+' labelled 1x timing',int(probe['nb_frames'])==181 and probe['avg_frame_rate']=='30/1' and abs(float(probe['duration'])-181/30)<1e-5 and v['sample_error_s']<=.0021)
    F.dump(OUT/'20261018-verification.json',dict(checks=checks,passed=len(checks),physics_steps=0,rollouts_added=0))
    print('Passed',len(checks),'checks',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['selection','reserved','render','verify']);a=p.parse_args()
    if a.mode=='render':render()
    elif a.mode=='verify':verify()
    else:analysis(a.mode)
