#!/usr/bin/env python3
"""Opt-in foot measurement. No default/controller/model files are modified.

Use resguard. The baseline and pulse use the existing estimated-state error
path, identical seed, full 35-node MPC and the archived h0885 plan.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import types

ROOT = Path(__file__).resolve().parents[1]
os.environ['TABLE_H'] = '0.885'
os.environ.setdefault('CL_ASSETS_DIR', str(ROOT.parent/'CL_Assets'))
os.environ.setdefault('LEAN_TASK_DIR', str(ROOT/'studies/runs/_stage/mjpc/tasks/humanoid_bench/lean'))
import croco_bridge as cb
import numpy as np
import mujoco
import pinocchio as pin
import crocoddyl
import contact_select as cs
import cmpc_brace_vs_stand as BVS
import croco_twin as CT
from croco.plant.mujoco_plant import MuJoCoPlant, default_sense
from croco.runtime.loop import ControlLoop, LoopConfig

OUT = ROOT/'studies/runs/20261016-cmpc-foot-motion'
CELL = ROOT/'studies/runs/2026-09-08_height_dyn/h0885_x1060/cell'
FEET = list(cb.SOLES)

def dump(path, obj):
    with Path(path).open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def aslist(a): return np.asarray(a).tolist()
def finite(a):
    a = np.asarray(a, float)
    return [None if not np.isfinite(x) else float(x) for x in a.flat]
def violation(v, lo, hi): return np.maximum(np.maximum(lo-v, v-hi), 0)
def rot_angle(a, b): return np.arccos(np.clip((np.trace(a.T@b)-1)/2, -1, 1))

def poses(ocp, x, rd):
    pin.forwardKinematics(ocp.rmodel, rd, x[:ocp.state.nq])
    pin.updateFramePlacements(ocp.rmodel, rd)
    return np.array([rd.oMf[ocp.sites[f]].translation.copy() for f in FEET]), np.array([rd.oMf[ocp.sites[f]].rotation.copy() for f in FEET])

def build():
    return BVS.Flight(str(CELL), 'elbow_forearm', horizon=35, iters=1, threads=1)

def cone_spec(c):
    return dict(A=aslist(c.A), lower=finite(c.lb), upper=finite(c.ub),
                row_units=['N']*5+(['N m']*12 if c.A.shape[1]==6 else []))

def provenance(fl):
    mpc=fl.mpc
    return dict(plan=str(CELL/'plan_elbow_forearm.json'), plan_sha256=sha(CELL/'plan_elbow_forearm.json'),
        xs_sha256=sha(CELL/'xs_elbow_forearm.npy'), us_sha256=sha(CELL/'us_elbow_forearm.npy'),
        model=str(cs.MODEL), model_sha256=sha(cs.MODEL), plan_fields=fl.plan,
        model_arrays_sha256={n:hashlib.sha256(np.ascontiguousarray(getattr(fl.m,n))).hexdigest()
            for n in ['body_pos','body_quat','body_mass','body_inertia','geom_pos','geom_quat','geom_size','geom_friction','mesh_vert','mesh_face','dof_damping','dof_armature','actuator_gainprm','actuator_biasprm']},
        model_nodes=len(mpc.models), control_nodes=len(fl.us), hold_index=len(fl.us)-35,
        hold_window_s=35*fl.dt, dt_s=fl.dt, horizon=mpc.H,
        foot_gains=aslist(mpc.ocp.foot_gains), foot_anchors={f:aslist(mpc.ocp.foot_ref[f].homogeneous) for f in FEET})

def probe():
    fl=build(); p=provenance(fl); mp=fl.mpc; k=len(fl.us)-35
    assert len(mp.models)==len(fl.us)==200 and mp.H==35 and fl.dt==.02
    u,x=mp(k,fl.xs[k].copy()); assert mp.problem.T==35 and len(mp.xs)==36
    p['verified_hold_nodes']=mp.problem.T
    p['hold_contacts']=[list(m.differential.contacts.contacts.todict()) for m in mp.problem.runningModels]
    m=mp.problem.runningModels[0]; d=m.createData(); m.calc(d,mp.xs[0],mp.us[0]); old=d.xnext.copy()
    contacts=m.differential.contacts.contacts.todict()
    refs={f:contacts[f].contact.reference.copy() for f in FEET}
    rd=mp.ocp.rmodel.createData(); pp,rr=poses(mp.ocp,mp.xs[0],rd)
    for i,f in enumerate(FEET): contacts[f].contact.reference=pin.SE3(rr[i],pp[i])
    m.calc(d,mp.xs[0],mp.us[0]); p['reanchor_same_state_xnext_max_abs']=float(np.max(np.abs(d.xnext-old)))
    for f in FEET: contacts[f].contact.reference=refs[f]
    p['reanchor_test']='Saved-plan state k=165; reference changed to its contact pose. No plant rollout. Zero foot Kp predicts no effect.'
    p['cones']={n:cone_spec(c.cost.residual.reference) for n,c in m.differential.costs.costs.todict().items() if n.startswith('cone_')}
    p['contact_data_frames']='ContactModel6D/3D LOCAL_WORLD_ALIGNED; f is world-aligned at contact origin. WrenchCone A uses identity orientation, so its CoP rectangle is world-aligned.'
    dump(OUT/'20261016-probe.json',p)
    print(json.dumps({k:p[k] for k in ['verified_hold_nodes','hold_index','foot_gains','reanchor_same_state_xnext_max_abs']}),flush=True)

class Recorder(MuJoCoPlant):
    def __init__(self, fl, pulse):
        m,d=cs.load(ik_margin=0)
        super().__init__(m,d,sense=default_sense(np.random.default_rng(17)),tau_limit=fl.tau_lim,nu=27)
        self.fl=fl; self.pulse=pulse; self.rows=[]; self.extra=[]; self.commands=[]
        self.cmd=None; self.phase=0.; self.cost=0.; self.dfx=np.zeros(3)
        self.bids=[cs.bid(m,cb.SOLES[f][0]) for f in FEET]
        self.wrist=cs.bid(m,'left_wrist_yaw_link'); self.hand=cs.bid(m,cs.REACH_BODY)
        self.jids=m.actuator_trnid[:,0]; self.qa=m.jnt_qposadr[self.jids]; self.va=m.jnt_dofadr[self.jids]
        self.cone=crocoddyl.WrenchCone(np.eye(3),fl.plan['mu'],np.array(cb.FOOT_HALF)*2,4,False,0.,1e4)
        self.jac=np.zeros((3,m.nv));self.jacr=np.zeros((3,m.nv))
        self.contacts=[]
    def write(self,cmd):
        super().write(cmd);self.cmd=cmd
        self.commands.append(dict(t=float(self.d.time),q=aslist(cmd.q_des),v=aslist(cmd.v_des),tau=aslist(cmd.tau_ff)))
    def sample(self):
        m,d=self.m,self.d
        self.rows.append(np.concatenate(([d.time,self.cost],d.qpos,d.qvel,d.ctrl,d.actuator_force,[self.phase],self.dfx)))
        pp=[];rr=[];pv=[];wrenches=[];cn=[]
        for i,f in enumerate(FEET):
            b=self.bids[i];r=d.xmat[b].reshape(3,3);p=d.xpos[b]+r@cb.SOLES[f][1]
            pp.append(p.copy());rr.append(r.copy());mujoco.mj_jac(m,d,self.jac,self.jacr,p,b);pv.append(self.jac@d.qvel)
            w=np.zeros(6); count=0
            for ci in range(d.ncon):
                c=d.contact[ci];bs=m.geom_bodyid[c.geom]
                if b not in bs or not any(m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE for g in c.geom):continue
                ff=np.zeros(6);mujoco.mj_contactForce(m,d,ci,ff)
                R=c.frame.reshape(3,3);sign=1 if bs[1]==b else -1
                force=sign*R.T@ff[:3];torque=sign*R.T@ff[3:]+np.cross(c.pos-p,force)
                w+=np.r_[force,torque];count+=int(ff[0]>1.)
                self.contacts.append(dict(t=float(d.time),foot=f,geoms=aslist(c.geom),position_world_m=aslist(c.pos),
                    frame_rows=aslist(R),force_contact=aslist(ff),mu=aslist(c.friction),
                    circular_friction_violation_N=float(max(0,np.linalg.norm(ff[1:3])-c.friction[0]*ff[0]))))
            wrenches.append(w);cn.append(count)
        points=[];vel=[]
        for b,o in [(self.wrist,np.zeros(3)),(self.hand,cs.REACH_OFF)]:
            p=d.xpos[b]+d.xmat[b].reshape(3,3)@o
            mujoco.mj_jac(m,d,self.jac,self.jacr,p,b);points.append(p.copy());vel.append(self.jac@d.qvel)
        self.extra.append(dict(t=float(d.time),sole_p=pp,sole_R=rr,sole_v=pv,sole_w=wrenches,loaded_contacts=cn,
            point_p=points,point_v=vel,q_des=self.cmd.q_des.copy(),v_des=self.cmd.v_des.copy(),tau_ff=self.cmd.tau_ff.copy()))
    def step(self,dt):
        m,d=self.m,self.d
        for _ in range(int(round(dt/m.opt.timestep))):
            self.dfx=np.array([0.,90.,0.]) if self.pulse and 4.5-1e-9<=d.time<4.8-1e-9 else np.zeros(3)
            d.xfrc_applied[self.hand,:3]=self.dfx
            mujoco.mj_forward(m,d)
            self.sample();mujoco.mj_step(m,d)


def run(case):
    fl=build();mp=fl.mpc;ocp=mp.ocp;plant=Recorder(fl,case=='pulse');m,d=plant.m,plant.d
    d.qpos[:]=fl.start_qpos();d.qvel[:]=0;d.ctrl[:]=d.qpos[7:34];d.time=0;mujoco.mj_forward(m,d)
    mp.reset();rd=ocp.rmodel.createData();pred=[];forces=[];windows=[];tick=0
    data_cache={}
    def policy(t,st):
        k=min(int(round(t/.02)),165)
        q=np.r_[st.base_pos,st.base_quat,st.q];v=np.r_[st.base_linvel,st.base_angvel,st.v]
        x=np.r_[cb.mj_to_pin(q),cb.mj_to_pin_v(v,CT._quat_to_mat(st.base_quat))]
        u,x1=mp(k,x);assert mp.problem.T==35
        p0,r0=poses(ocp,x,rd);p1,r1=poses(ocp,x1,rd)
        pred.append(dict(t=float(d.time),x_est=x.copy(),x_next=x1.copy(),sole_p0=p0,sole_R0=r0,sole_p1=p1,sole_R1=r1,solve_s=mp.solve_times[-1],k=k))
        plant.phase=k;plant.cost=mp.solver.cost
        return x1[7:34],x1[40:],np.clip(u,-fl.tau_lim,fl.tau_lim)
    def audit(row,st,cmd):
        nonlocal tick
        if tick%5==0:
            max_u=max_j=0.
            reanchor_dx=reanchor_df=0.
            for j,(am,x,u) in enumerate(zip(mp.problem.runningModels,mp.xs[:-1],mp.us)):
                key=id(am)
                # Data are private; evaluating forces cannot overwrite solver data.
                if key not in data_cache: data_cache[key]=am.createData()
                dat=data_cache[key]
                am.calc(dat,x,u)
                max_u=max(max_u,float(violation(np.asarray(u),am.u_lb,am.u_ub).max()))
                max_j=max(max_j,float(violation(np.asarray(x)[7:34],m.jnt_range[plant.jids,0],m.jnt_range[plant.jids,1]).max()))
                costs=am.differential.costs.costs.todict()
                for name,cd in dat.differential.multibody.contacts.contacts.todict().items():
                    w=np.array(cd.f.vector);cname='cone_'+name.removeprefix('brace_')
                    cone=costs[cname].cost.residual.reference
                    cv=cone.A@(w if cone.A.shape[1]==6 else w[:3])
                    forces.append(dict(t=float(d.time),node=j,contact=name,wrench_world=aslist(w),
                        cone_values=aslist(cv),cone_violation=aslist(violation(cv,cone.lb,cone.ub))))
                if j==0:
                    old_x=dat.xnext.copy()
                    old_f={n:np.array(c.f.vector) for n,c in dat.differential.multibody.contacts.contacts.todict().items()}
                    contacts=am.differential.contacts.contacts.todict()
                    refs={f:contacts[f].contact.reference.copy() for f in FEET}
                    pp,rr=poses(ocp,x,rd)
                    for i,f in enumerate(FEET): contacts[f].contact.reference=pin.SE3(rr[i],pp[i])
                    am.calc(dat,x,u)
                    reanchor_dx=float(np.max(np.abs(dat.xnext-old_x)))
                    reanchor_df=max(float(np.max(np.abs(np.array(c.f.vector)-old_f[n]))) for n,c in dat.differential.multibody.contacts.contacts.todict().items())
                    for f in FEET: contacts[f].contact.reference=refs[f]
            windows.append(dict(t=float(d.time),T=mp.problem.T,control_violation_Nm=max_u,joint_violation_rad=max_j,
                reactive_reanchor_xnext_max_abs=reanchor_dx,reactive_reanchor_wrench_max_abs=reanchor_df))
        tick+=1
    # Zero transport latency is appropriate to this synchronous in-process test.
    # Disabling wall-time compensation also prevents auditing overhead changing commands.
    cfg=LoopConfig(ctrl_hz=50,latency_comp=False,realtime=None)
    loop=ControlLoop(plant,policy,stance=None,cfg=cfg,on_step=audit)
    start=time.monotonic();log=loop.run(fl.kp,fl.kd,max_seconds=6.)
    wall=time.monotonic()-start
    rows=np.asarray(plant.rows);stem=OUT/('20261016-'+case)
    BVS.write_csv(str(stem)+'.csv',rows,fl,fl.plan['target'],case,0,6.,'pulse=+y90N_4.5to4.8s' if plant.pulse else '')
    arrays={k:np.asarray([r[k] for r in plant.extra]) for k in plant.extra[0]}
    arrays.update({'pred_'+k:np.asarray([r[k] for r in pred]) for k in pred[0]})
    np.savez_compressed(str(stem)+'-raw.npz',**arrays)
    dump(str(stem)+'-predicted-forces.json',forces);dump(str(stem)+'-simulated-contacts.json',plant.contacts)
    dump(str(stem)+'-commands.json',plant.commands)
    result=summarize(fl,plant,rows,arrays,forces,windows)
    result.update(case=case,wall_s=wall,rollout_count=1,periods=len(log),horizon_min=35,horizon_max=35,
        solve_mean_ms=float(np.mean(mp.solve_times)*1000),solve_p95_ms=float(np.percentile(mp.solve_times,95)*1000),
        solve_above_20ms=int(np.sum(np.array(mp.solve_times)>.02)),
        tau_clamp_count=sum(r.get('tau_sat',0) for r in log),q_clamp_count=sum(r.get('q_clip',0) for r in log),
        provenance=provenance(fl),observation=dict(path='MuJoCoPlant.read -> default_sense(seed=17) -> State -> policy',
            base_position_sigma_m=.01,base_rotation_sigma_rad=.01,joint_noise=0,velocity_noise=0,
            description='S17 synthetic estimated-state error model; base pose biased, velocities ideal. Not the deployed EKF.',
            bias_position_m=aslist(np.random.default_rng(17).normal(0,.01,3))),
        runtime=dict(iterations=1,threads=1,latency_comp=False,in_process_free_run=True),
        window_checks=windows)
    dump(str(stem)+'-summary.json',result)
    print(json.dumps({k:result[k] for k in ['case','wall_s','solve_p95_ms','motion','feet','predicted','simulated']}),flush=True)


def summarize(fl,p,rows,a,forces,windows):
    t=a['t'];m=p.m;q=rows[:,2:2+m.nq];v=rows[:,2+m.nq:2+m.nq+m.nv]
    ctrl=rows[:,2+m.nq+m.nv:2+m.nq+m.nv+m.nu];af=rows[:,2+m.nq+m.nv+m.nu:2+m.nq+m.nv+2*m.nu]
    names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,int(j)) for j in p.jids]
    motion={};feet={};dt=np.diff(t)
    for phase,mask in [('startup',t<.5),('approach',t<2.4),('hold',t>=4.),('post_pulse',t>=4.5)]:
        acc=np.diff(v[:,p.va],axis=0)/dt[:,None];amask=mask[1:]&mask[:-1]
        speed=np.abs(v[:,p.va]);ix=np.unravel_index(np.where(mask[:,None],speed,-1).argmax(),speed.shape)
        accel_ix=np.unravel_index(np.where(amask[:,None],abs(acc),-1).argmax(),acc.shape)
        point={}
        for i,name in enumerate(['brace_wrist','reaching_hand']):
            pv=a['point_v'][:,i];pa=np.diff(pv,axis=0)/dt[:,None]
            point[name]=dict(speed_m_s=float(np.linalg.norm(pv[mask],axis=1).max()),acceleration_m_s2=float(np.linalg.norm(pa[amask],axis=1).max()))
        motion[phase]=dict(joint_speed_rad_s=float(speed[ix]),speed_joint=names[ix[1]],speed_time_s=float(t[ix[0]]),
            joint_acceleration_rad_s2=float(abs(acc[accel_ix])),acceleration_joint=names[accel_ix[1]],acceleration_time_s=float((t[accel_ix[0]]+t[accel_ix[0]+1])/2),
            q_tracking_rms_rad=float(np.sqrt(np.mean((a['q_des'][mask]-q[mask][:,p.qa])**2))),
            v_tracking_rms_rad_s=float(np.sqrt(np.mean((a['v_des'][mask]-v[mask][:,p.va])**2))),points=point)
    for i,f in enumerate(FEET):
        pp=a['sole_p'][:,i];rr=a['sole_R'][:,i];anchor=fl.mpc.ocp.foot_ref[f].translation
        idx=np.minimum(np.searchsorted(t,a['pred_t']+.02),len(t)-1);valid=a['pred_t']+.02<=t[-1]+1e-8
        residual=pp[idx]-a['pred_sole_p1'][:,i]
        est_delta=a['pred_sole_p0'][1:,i]-a['pred_sole_p1'][:-1,i]
        feet[f]=dict(displacement_from_start_max_mm=float(1000*np.linalg.norm(pp-pp[0],axis=1).max()),
            tangential_from_start_max_mm=float(1000*np.linalg.norm(pp[:,:2]-pp[0,:2],axis=1).max()),
            rotation_from_start_max_deg=float(max(rot_angle(rr[0],r) for r in rr)*180/np.pi),
            fixed_anchor_residual_max_mm=float(1000*np.linalg.norm(pp-anchor,axis=1).max()),
            estimated_fixed_anchor_residual_max_mm=float(1000*np.linalg.norm(a['pred_sole_p0'][:,i]-anchor,axis=1).max()),
            one_step_truth_position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(residual[valid]**2,axis=1)))),
            one_step_estimated_position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(est_delta**2,axis=1)))),
            loss_sample_seconds=float(np.sum(a['loaded_contacts'][:,i]==0)*m.opt.timestep),
            peak_tangential_speed_m_s=float(np.linalg.norm(a['sole_v'][:,i,:2],axis=1).max()),
            post_pulse_displacement_mm=float(1000*np.linalg.norm(pp[-1]-pp[np.searchsorted(t,4.5)],axis=0)),
            hold_peak_tangential_speed_m_s=float(np.linalg.norm(a['sole_v'][t>=4.,i,:2],axis=1).max()))
    predicted={}
    for name in sorted(set(r['contact'] for r in forces)):
        group=[r for r in forces if r['contact']==name];w=np.array([r['wrench_world'] for r in group]);cv=np.array([r['cone_violation'] for r in group]);loaded=w[:,2]>1.
        out=dict(samples=len(w),normal_min_N=float(w[:,2].min()),normal_max_N=float(w[:,2].max()),negative_normal_count=int(np.sum(w[:,2]<0)),
            tangent_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),cone_force_row_max_N=float(cv[:,:5].max()),
            circular_friction_violation_max_N=float(np.maximum(np.linalg.norm(w[:,:2],axis=1)-fl.plan['mu']*w[:,2],0).max()))
        if name in FEET:
            cop=np.c_[-w[loaded,4]/w[loaded,2],w[loaded,3]/w[loaded,2]]
            out.update(cone_moment_row_max_Nm=float(cv[:,5:].max()),world_axis_cop_margin_min_mm=float(1000*np.min(np.array(cb.FOOT_HALF)-abs(cop))))
        predicted[name]=out
    simulated={}
    for i,f in enumerate(FEET):
        w=a['sole_w'][:,i];R=a['sole_R'][:,i];local=np.concatenate([np.einsum('nji,nj->ni',R,w[:,:3]),np.einsum('nji,nj->ni',R,w[:,3:])],axis=1)
        loaded=local[:,2]>1.;cop=np.c_[-local[loaded,4]/local[loaded,2],local[loaded,3]/local[loaded,2]]
        cv=violation(w@p.cone.A.T,p.cone.lb,p.cone.ub)
        simulated[f]=dict(world_normal_min_N=float(w[:,2].min()),world_normal_max_N=float(w[:,2].max()),world_tangent_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),
            cone_force_row_max_N=float(cv[:,:5].max()),cone_moment_row_max_Nm=float(cv[:,5:].max()),sole_local_cop_margin_min_mm=float(1000*np.min(np.array(cb.FOOT_HALF)-abs(cop))),
            individual_contact_friction_violation_max_N=max(r['circular_friction_violation_N'] for r in p.contacts if r['foot']==f))
    qvi=violation(q[:,p.qa],m.jnt_range[p.jids,0],m.jnt_range[p.jids,1])
    ctrvi=violation(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1])*m.actuator_ctrllimited
    afvi=violation(af,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])*m.actuator_forcelimited
    reach=np.linalg.norm(a['point_p'][:,1]-fl.plan['target'],axis=1)
    return dict(motion=motion,feet=feet,predicted=predicted,simulated=simulated,
        simulation_dt_s=m.opt.timestep,simulation_end_s=float(t[-1]),reach_final_mm=float(1000*reach[-1]),
        reach_hold_rms_mm=float(1000*np.sqrt(np.mean(reach[t>=4.]**2))),pelvis_min_m=float(q[:,2].min()),pelvis_final_m=float(q[-1,2]),
        joint_violation_max_rad=float(qvi.max()),joint_violation_sample_count=int(np.sum(qvi>1e-9)),
        ctrl_bound_violation_max_rad=float(ctrvi.max()),actuator_force_bound_violation_max_Nm=float(afvi.max()),
        actuator_at_force_bound_sample_count=int(np.sum(abs(af)>=np.maximum(abs(m.actuator_forcerange[:,0]),abs(m.actuator_forcerange[:,1]))-1e-5)),
        command_torque_tracking_rms_Nm=float(np.sqrt(np.mean((af-a['tau_ff'])**2))),
        command_torque_tracking_definition='Actuator minus feedforward includes PD response; not a torque-servo tracking error.',
        joint_names=names,control_limited=aslist(m.actuator_ctrllimited),control_bounds=aslist(m.actuator_ctrlrange))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['nominal','pulse']);ap.add_argument('--probe',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.probe: probe()
    else: run(args.case)
