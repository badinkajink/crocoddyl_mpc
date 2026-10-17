#!/usr/bin/env python3
"""Score and render saved foot-motion data; no physics integration or controller.

All replay forces below are forward-dynamics reconstructions from stored q,v,ctrl
with the logged disturbance. Native contact state, force and friction vectors
from the live rollout remain in the original raw files.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
os.environ['TABLE_H']='0.885'
os.environ.setdefault('LEAN_TASK_DIR',str(ROOT/'studies/runs/_stage/mjpc/tasks/humanoid_bench/lean'))
os.environ.setdefault('CL_ASSETS_DIR',str(ROOT.parent/'CL_Assets'))
import contact_select as cs
import mujoco
OUT=ROOT/'studies/runs/20261016-cmpc-foot-motion'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
FEET={'sole_left':('left_ankle_roll_link',np.array([.033,0,-.055])), 'sole_right':('right_ankle_roll_link',np.array([.033,0,-.055]))}

def dump(path,obj):
    with Path(path).open('w') as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def load(case):
    stem=OUT/('20261016-'+case)
    with open(str(stem)+'.csv') as f:
        rd=csv.reader(x for x in f if not x.startswith('#'));cols=next(rd);vals=np.array(list(rd),float)
    return stem,dict(zip(cols,vals.T)),dict(np.load(str(stem)+'-raw.npz')),json.loads(Path(str(stem)+'-summary.json').read_text())
def angle(a,b):return np.arccos(np.clip((np.trace(a.T@b)-1)/2,-1,1))
def scalar_stats(x):return dict(min=float(np.min(x)),max=float(np.max(x)),rms=float(np.sqrt(np.mean(np.array(x)**2))))

def analyze():
    m,d=cs.load(ik_margin=0); res={};all_a={};check={}
    from scipy.spatial import ConvexHull
    geometry={};hulls={}
    for f,(body,off) in FEET.items():
        b=cs.bid(m,body)
        gs=[g for g in range(m.ngeom) if m.geom_bodyid[g]==b and m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[g] or m.geom_conaffinity[g])]
        vertices=[]
        for g in gs:
            mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid]
            R=np.zeros(9);mujoco.mju_quat2Mat(R,m.geom_quat[g])
            vertices.extend(m.mesh_vert[start:start+count]@R.reshape(3,3).T+m.geom_pos[g])
        vertices=np.asarray(vertices);bottom=vertices[vertices[:,2]<=vertices[:,2].min()+1e-4]
        hull=ConvexHull(bottom[:,:2]);hulls[f]=hull
        geometry[f]=dict(bottom_z_m=float(vertices[:,2].min()),polygon_xy_m=bottom[hull.vertices,:2].tolist(),
            planner_centre_body_m=off.tolist(),planner_half_extents_m=[.1,.04],collision_geoms=gs,
            friction=[m.geom_friction[g].tolist() for g in gs])
    table=cs.bid(m,'table');foot_bids={cs.bid(m,b):f for f,(b,o) in FEET.items()}
    jids=m.actuator_trnid[:,0];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    for case in ['nominal','pulse']:
        stem,c,a,base=load(case);t=a['t'];all_a[case]=a
        phase={};pred=json.loads(Path(str(stem)+'-predicted-forces.json').read_text())
        q=np.array([c['qpos'+str(i)] for i in range(m.nq)]).T;v=np.array([c['qvel'+str(i)] for i in range(m.nv)]).T
        ctrl=np.array([c['ctrl'+str(i)] for i in range(m.nu)]).T;af=np.array([c['afrc'+str(i)] for i in range(m.nu)]).T
        expected_torque=m.actuator_gainprm[:,0]*(a['q_des']-q[:,qa])-m.actuator_biasprm[:,2]*(a['v_des']-v[:,va])+a['tau_ff']
        unclipped_expected=expected_torque.copy()
        for j in range(m.nu):
            if m.actuator_forcelimited[j]: expected_torque[:,j]=np.clip(expected_torque[:,j],*m.actuator_forcerange[j])
        check[case]=dict(position_servo_identity_torque_max_error_Nm=float(np.max(np.abs(expected_torque-af))),
            tau_ff_to_actuator_rms_Nm=float(np.sqrt(np.mean((af-a['tau_ff'])**2))),
            total_torque_saturation_max_Nm=float(np.max(np.abs(unclipped_expected-expected_torque))),
            total_torque_saturation_samples=int(np.sum(np.abs(unclipped_expected-expected_torque)>1e-6)))
        limited_ctrl=ctrl.copy()
        for j in range(m.nu):
            if m.actuator_ctrllimited[j]:limited_ctrl[:,j]=np.clip(limited_ctrl[:,j],*m.actuator_ctrlrange[j])
        realized=m.actuator_gainprm[:,0]*(limited_ctrl-q[:,qa])+m.actuator_biasprm[:,2]*v[:,va]
        for j in range(m.nu):
            if m.actuator_forcelimited[j]:realized[:,j]=np.clip(realized[:,j],*m.actuator_forcerange[j])
        check[case]['with_ctrl_clipping_torque_max_error_Nm']=float(np.max(abs(realized-af)))
        ci=np.unravel_index(np.abs(expected_torque-af).argmax(),af.shape)
        check[case]['largest_ctrl_clipping_torque_difference']=dict(t=float(t[ci[0]]),joint=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(jids[ci[1]])),Nm=float(abs(expected_torque-af)[ci]))
        pressure_margin={f:np.full(len(t),np.nan) for f in FEET}
        simrows=[];reconstruction_error=0.;table_first=None;extra_load_max=0.;foot_wrench_error=0.
        # Replay at the original 500 Hz for contact witness and wrench scoring.
        for k,stamp in enumerate(t):
            d.qpos[:]=q[k];d.qvel[:]=v[k];d.ctrl[:]=ctrl[k];d.xfrc_applied[:]=0
            d.xfrc_applied[cs.bid(m,cs.REACH_BODY),:3]=[c['dfx'][k],c['dfy'][k],c['dfz'][k]]
            mujoco.mj_forward(m,d)
            reconstruction_error=max(reconstruction_error,float(np.max(abs(d.actuator_force-af[k]))))
            tabload=0.;replay_w={f:np.zeros(6) for f in FEET};pressure={f:np.zeros(3) for f in FEET}
            for ci in range(d.ncon):
                con=d.contact[ci];bs=m.geom_bodyid[con.geom]
                isfoot=next((int(b) for b in bs if b in foot_bids),None)
                floor=any(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in con.geom)
                if isfoot is not None and floor: name=foot_bids[isfoot];rb=isfoot
                elif table in bs:
                    rb=int(bs[1] if bs[0]==table else bs[0]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,rb)
                    if name and 'object' in name:continue
                else:continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                sign=1 if bs[1]==rb else -1;world=sign*con.frame.reshape(3,3).T@f[:3]
                dim=con.dim;mu=con.friction[:dim-1]
                scaled=np.divide(f[1:dim],mu,out=np.zeros(dim-1),where=mu>0)
                native=max(0.,(np.linalg.norm(scaled) if m.opt.cone==mujoco.mjtCone.mjCONE_ELLIPTIC else np.sum(abs(scaled)))-f[0])
                cp=con.pos;rot=d.xmat[rb].reshape(3,3);local=rot.T@(cp-d.xpos[rb])
                jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,cp,rb)
                point_velocity=jp@d.qvel
                if name in FEET:
                    sole_centre=d.xpos[rb]+rot@FEET[name][1]
                    replay_w[name]+=np.r_[world,sign*con.frame.reshape(3,3).T@f[3:]+np.cross(cp-sole_centre,world)]
                    pressure[name]+=np.r_[f[0]*local[:2],f[0]]
                simrows.append(dict(t=float(stamp),body=name,geoms=[int(x) for x in con.geom],dim=int(dim),
                    normal_N=float(f[0]),tangent_N=float(np.linalg.norm(f[1:3])),world_force_N=world.tolist(),
                    friction=list(con.friction),native_cone_excess_N=float(native),
                    contact_position_body_m=local.tolist(),contact_position_world_m=cp.tolist(),
                    material_point_tangential_speed_m_s=float(np.linalg.norm(point_velocity[:2]))))
                if table in bs:
                    tabload+=float(f[0])
                    if f[0]>1 and table_first is None:table_first=dict(t=float(stamp),body=name,wrist_speed_m_s=float(np.linalg.norm(a['point_v'][k,0])))
            extra_load_max=max(extra_load_max,tabload)
            foot_wrench_error=max(foot_wrench_error,max(float(np.max(abs(replay_w[f]-a['sole_w'][k,i]))) for i,f in enumerate(FEET)))
            for f in FEET:
                if pressure[f][2]>5:
                    xy=pressure[f][:2]/pressure[f][2];eq=hulls[f].equations
                    pressure_margin[f][k]=float(np.min(-(eq[:,:2]@xy+eq[:,2])))
        dump(str(stem)+'-reconstructed-contacts.json',simrows)
        for label,lo,hi in [('startup',0,.5),('approach',0,2.4),('hold',4,7.),('post_pulse',4.5,7.)]:
            mask=(t>=lo-1e-9)&(t<hi-1e-9);idx0=int(np.flatnonzero(mask)[0]);part={}
            for i,f in enumerate(FEET):
                pp=a['sole_p'][:,i];rr=a['sole_R'][:,i];w=a['sole_w'][:,i]
                local=np.concatenate([np.einsum('nji,nj->ni',rr,w[:,:3]),np.einsum('nji,nj->ni',rr,w[:,3:])],axis=1)
                good=mask&(local[:,2]>5.)
                cop=np.c_[-local[good,4]/local[good,2],local[good,3]/local[good,2]]
                cmask=(a['pred_t'][:-1]>=lo-1e-9)&(a['pred_t'][1:]<hi-1e-9)
                e=a['pred_sole_p0'][1:,i]-a['pred_sole_p1'][:-1,i]
                rotres=np.array([angle(x,y) for x,y in zip(a['pred_sole_R1'][:-1,i],a['pred_sole_R0'][1:,i])])
                # Loaded slip is instantaneous velocity of the body material point at the witness.
                # Sole-centre velocity also includes rotation and can overestimate sliding.
                footprint=np.array([r['contact_position_body_m'][:2] for r in simrows if r['body']==f and lo-1e-9<=r['t']<hi-1e-9 and r['normal_N']>5.])
                part[f]=dict(displacement_max_from_phase_start_mm=float(1000*np.linalg.norm(pp[mask]-pp[idx0],axis=1).max()),
                    tangential_max_from_phase_start_mm=float(1000*np.linalg.norm(pp[mask,:2]-pp[idx0,:2],axis=1).max()),
                    rotation_max_from_phase_start_deg=float(max(angle(rr[idx0],r) for r in rr[mask])*180/np.pi),
                    contact_loss_s=float(np.sum(a['loaded_contacts'][mask,i]==0)*m.opt.timestep),
                    sole_tangential_speed_max_m_s=float(np.linalg.norm(a['sole_v'][mask,i,:2],axis=1).max()),
                    estimated_one_step_position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(e[cmask]**2,axis=1)))),
                    estimated_one_step_rotation_rms_deg=float(np.sqrt(np.mean(rotres[cmask]**2))*180/np.pi),
                    simulated_local_cop_margin_min_mm=float(1000*np.min(np.array([.1,.04])-abs(cop))) if len(cop) else None,
                    simulated_pressure_centre_mesh_margin_min_mm=float(1000*np.nanmin(pressure_margin[f][mask])),
                    witness_body_xy_extent_m=[footprint.min(axis=0).tolist(),footprint.max(axis=0).tolist()] if len(footprint) else None)
            part['predicted']={}
            for name in sorted(set(r['contact'] for r in pred)):
                g=[r for r in pred if r['contact']==name and lo-1e-9<=r['t']<hi-1e-9]
                if not g:continue
                w=np.array([r['wrench_world'] for r in g]);cv=np.array([r['cone_violation'] for r in g]);loaded=w[:,2]>5
                vv=dict(normal_min_N=float(w[:,2].min()),normal_max_N=float(w[:,2].max()),negative_normal_count=int(np.sum(w[:,2]<0)),
                    tangent_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),force_rows_max_N=float(cv[:,:5].max()),
                    first_node_force_rows_max_N=max((max(r['cone_violation'][:5]) for r in g if r['node']==0),default=None),
                    samples=len(g))
                if name in FEET:
                    cop=np.c_[-w[loaded,4]/w[loaded,2],w[loaded,3]/w[loaded,2]]
                    vv.update(moment_rows_max_Nm=float(cv[:,5:].max()),cop_margin_min_mm=float(1000*np.min(np.array([.1,.04])-abs(cop))))
                part['predicted'][name]=vv
            part['simulated_contacts']={}
            for name in sorted(set(r['body'] for r in simrows)):
                g=[r for r in simrows if r['body']==name and lo-1e-9<=r['t']<hi-1e-9]
                if not g:continue
                part['simulated_contacts'][name]=dict(normal_min_N=min(r['normal_N'] for r in g),normal_max_N=max(r['normal_N'] for r in g),
                    tangent_max_N=max(r['tangent_N'] for r in g),native_cone_excess_max_N=max(r['native_cone_excess_N'] for r in g),samples=len(g))
                loaded=[r for r in g if r['normal_N']>5.]
                if loaded: part['simulated_contacts'][name]['loaded_material_point_slip_peak_m_s']=max(r['material_point_tangential_speed_m_s'] for r in loaded)
            phase[label]=part
        res[case]=dict(phases=phase,first_table_load=table_first,table_load_peak_N=extra_load_max,
            contact_replay_actuator_error_max_Nm=reconstruction_error,
            contact_replay_foot_wrench_max_component_error=foot_wrench_error,
            native_mujoco_cone=str(m.opt.cone),servo_check=check[case],summary=base)
    pre=all_a['nominal']['t']<4.5-1e-8
    res['comparison']=dict(pre_pulse_q_est_max_difference=float(np.max(abs(all_a['nominal']['pred_x_est'][all_a['nominal']['pred_t']<4.5-1e-8]-all_a['pulse']['pred_x_est'][all_a['pulse']['pred_t']<4.5-1e-8]))),
        pre_pulse_sole_position_max_difference_m=float(np.max(abs(all_a['nominal']['sole_p'][pre]-all_a['pulse']['sole_p'][pre]))),
        pulse_minus_nominal={f:{key:res['pulse']['phases']['post_pulse'][f][key]-res['nominal']['phases']['post_pulse'][f][key]
            for key in ['displacement_max_from_phase_start_mm','estimated_one_step_position_rms_mm','rotation_max_from_phase_start_deg']} for f in FEET})
    res['frames']=dict(predicted='World-aligned force and moment at named contact origin; cone matrix orientation identity. CoP model rectangle 200 x 80 mm in world x/y.',
        simulated='mj_contactForce: signed normal f[0] toward geom2, tangents f[1:3] in rows 1:3 of con.frame. World force on robot changes sign for geom1. Sum moments about sole centre then rotate into moving sole frame for physical CoP.',
        cone_units='Crocoddyl rows 0..4 N; rows 5..16 N m. MuJoCo native normalized tangential/torsion/rolling cone excess N; torsion/rolling friction coefficients have length units.',
        cop='Minimum of half-length minus abs(CoP x), half-width minus abs(CoP y); negative outside assumed rectangle. Report for normal >5 N. Rectangle describes predictor; loaded collision witness extent is also recorded.',
        prediction_residual='Next measured estimated sole pose minus solver xs[1], 20 ms. Includes solver defects and actuator realization; not an isolated contact integrator error.',
        contact_loss='No floor witness on that ankle body with normal >1 N, sampled at 2 ms. This includes unloaded contact, not necessarily positive separation.')
    res['collision_sole_geometry']=geometry
    res['frames']['mesh_pressure_margin']='Normal-load-weighted contact positions projected into sole body XY, distance to bottom mesh convex polygon edge. Normal load >5 N; this separates actual collision support from the smaller predictor rectangle and its reference-plane offset.'
    dump(MEDIA/'20261015-foot-motion.json',res)
    print(json.dumps(res['comparison']),flush=True)


def render():
    import imageio.v2 as imageio
    from PIL import Image,ImageDraw,ImageFont
    from height_video import frame_samples
    m,d=cs.load(ik_margin=0);W,H=960,540;m.vis.global_.offwidth=W;m.vis.global_.offheight=H
    ren=mujoco.Renderer(m,H,W);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam)
    cam.azimuth,cam.elevation,cam.distance=140.,-12.,3.4;cam.lookat[:]=[.55,-.1,.95]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
    for case in ['nominal','pulse']:
        stem,c,a,s=load(case);t=a['t'];requested,idx=frame_samples(t,30,1)
        out=MEDIA/('20261016-foot-'+case+'.mp4');frames=[];target_times=[0,.13,.5,2.4,4.5,4.8,5.5]
        frame_targets={int(np.argmin(abs(t[idx]-tt))) for tt in target_times}
        with imageio.get_writer(out,fps=30,macro_block_size=1,ffmpeg_params=['-threads','1']) as w:
            for vi,k in enumerate(idx):
                d.qpos[:]=[c['qpos'+str(i)][k] for i in range(m.nq)];mujoco.mj_forward(m,d)
                ren.update_scene(d,cam);im=Image.fromarray(ren.render());draw=ImageDraw.Draw(im)
                draw.rectangle((0,0,W,62),fill=(12,18,25))
                draw.text((10,4),f'{case.capitalize()} | Simulation t = {t[k]:.2f} s | Playback 1x | Table 0.885 m',font=font,fill='white')
                on=case=='pulse' and 4.5-1e-8<=t[k]<4.8-1e-8
                draw.text((10,32),'90 N world +y at reaching wrist: '+('ON' if on else 'OFF'),font=font,fill=(255,190,70) if on else 'white')
                w.append_data(np.asarray(im))
                if vi in frame_targets:frames.append(im.resize((480,270)))
        strip=Image.new('RGB',(480*4,270*2),'white')
        for i,im in enumerate(frames):strip.paste(im,((i%4)*480,(i//4)*270))
        strip.save(MEDIA/('20261016-foot-'+case+'-frames.jpg'))
        dump(out.with_suffix('.json'),dict(source=str(stem)+'.csv',source_sha256=hashlib.sha256(Path(str(stem)+'.csv').read_bytes()).hexdigest(),fps=30,playback_speed=1,
            frames=len(idx),duration_s=len(idx)/30,simulation_start_s=float(t[0]),simulation_end_s=float(t[-1]),max_timestamp_error_s=float(max(abs(t[idx]-requested))),video_sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    ren.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--render',action='store_true');args=ap.parse_args()
    if args.render:render()
    else:analyze()
