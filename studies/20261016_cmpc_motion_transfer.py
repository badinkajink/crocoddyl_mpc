#!/usr/bin/env python3
"""Opt-in startup/approach experiment; run every mode under resguard.

The 0.1 approach state cost already penalizes velocity. If startup inspection
supports a commanded-motion cause, w_vel=1 adds ten times that velocity weight.
This is a hypothesis test, not a hardware-derived velocity limit.
"""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'studies/runs/20261016-cmpc-motion-transfer'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'
spec = importlib.util.spec_from_file_location('foot_measure', ROOT/'studies/20261016_cmpc_foot_motion.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
np, mj, cs = F.np, F.mujoco, F.cs
BASE = F.CELL
F.OUT = OUT
OriginalRecorder = F.Recorder


class Recorder(OriginalRecorder):
    def sample(self):
        super().sample()
        m,d=self.m,self.d
        table=cs.bid(m,'table'); contacts=[]
        for ci in range(d.ncon):
            con=d.contact[ci]; bs=m.geom_bodyid[con.geom]
            if table not in bs: continue
            other=int(bs[1] if bs[0]==table else bs[0])
            name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,other)
            if name and 'object' in name: continue
            ff=np.zeros(6); mj.mj_contactForce(m,d,ci,ff)
            contacts.append(dict(t=float(d.time),body=name,normal_N=float(ff[0]),
                tangent_N=float(np.linalg.norm(ff[1:3])),geoms=F.aslist(con.geom),
                circular_cone_excess_N=float(max(0,np.linalg.norm(ff[1:3])-con.friction[0]*ff[0]))))
        self.table_contacts.extend(contacts)
        self.extra[-1].update(qacc=d.qacc[self.va].copy(),
            table_normal_N=sum(r['normal_N'] for r in contacts))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table_contacts=[]
        self.__class__.last=self


F.Recorder=Recorder


def load(stem):
    with Path(str(stem)+'.csv').open() as f:
        reader=csv.reader(x for x in f if not x.startswith('#'))
        names=next(reader); rows=np.array(list(reader),float)
    return dict(zip(names,rows.T)), dict(np.load(str(stem)+'-raw.npz'))


def startup_metrics(stem, m):
    c,a=load(stem);t=a['t'];jids=m.actuator_trnid[:,0]
    qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids]
    names=[mj.mj_id2name(m,mj.mjtObj.mjOBJ_JOINT,int(j)) for j in jids]
    q=np.array([c['qpos'+str(i)] for i in qa]).T
    v=np.array([c['qvel'+str(i)] for i in va]).T
    ctrl=np.array([c['ctrl'+str(i)] for i in range(m.nu)]).T
    force=np.array([c['afrc'+str(i)] for i in range(m.nu)]).T
    kp=m.actuator_gainprm[:,0];kd=-m.actuator_biasprm[:,2]
    p=kp*(a['q_des']-q);d=kd*(a['v_des']-v);ff=a['tau_ff']
    requested=p+d+ff
    effective_ctrl=np.where(m.actuator_ctrllimited,np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),ctrl)
    pre_force=kp*(effective_ctrl-q)-kd*v
    realized=np.where(m.actuator_forcelimited,np.clip(pre_force,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]),pre_force)
    acc=np.diff(v,axis=0)/np.diff(t)[:,None]
    mask=t<.5-1e-9;amask=mask[1:]&mask[:-1]
    ix=np.unravel_index(np.where(amask[:,None],abs(acc),-1).argmax(),acc.shape)
    j=ix[1];k=ix[0]
    def mx(x):return float(np.max(np.abs(x[mask])))
    out=dict(startup_acceleration_rad_s2=float(abs(acc[ix])),acceleration_joint=names[j],
        acceleration_time_s=float((t[k]+t[k+1])/2),startup_joint_speed_rad_s=mx(v),
        startup_wrist_peak_m_s=float(np.linalg.norm(a['point_v'][mask,0],axis=1).max()),
        q_tracking_rms_rad=float(np.sqrt(np.mean((a['q_des'][mask]-q[mask])**2))),
        v_tracking_rms_rad_s=float(np.sqrt(np.mean((a['v_des'][mask]-v[mask])**2))),
        startup_position_torque_peak_Nm=mx(p),startup_velocity_torque_peak_Nm=mx(d),
        startup_feedforward_peak_Nm=mx(ff),startup_actuator_peak_Nm=mx(force),
        startup_ctrl_clipping_effect_peak_Nm=mx(requested-pre_force),
        startup_force_saturation_effect_peak_Nm=mx(pre_force-realized),
        startup_ctrl_clipped_joint_samples=int(np.sum(abs(ctrl[mask]-effective_ctrl[mask])>1e-9)),
        startup_force_saturated_joint_samples=int(np.sum(abs(pre_force[mask]-realized[mask])>1e-9)),
        torque_identity_max_error_Nm=float(abs(force-realized).max()),
        initial_q_command_error_max_rad=float(abs(a['q_des'][0]-q[0]).max()),
        initial_v_command_max_rad_s=float(abs(a['v_des'][0]).max()),
        initial_ff_max_Nm=float(abs(ff[0]).max()),initial_requested_torque_max_Nm=float(abs(requested[0]).max()),
        initial_joint_velocity_max_rad_s=float(abs(v[0]).max()),
        initial_joint_limit_margin_rad=float(np.min(np.minimum(q[0]-m.jnt_range[jids,0],m.jnt_range[jids,1]-q[0]))),
        all_joint_limit_margin_min_rad=float(np.min(np.minimum(q-m.jnt_range[jids,0],m.jnt_range[jids,1]-q))),
        peak_acceleration_sample=dict(t=float(t[k]),joint=names[j],q=float(q[k,j]),q_des=float(a['q_des'][k,j]),
            v=float(v[k,j]),v_des=float(a['v_des'][k,j]),position_Nm=float(p[k,j]),velocity_Nm=float(d[k,j]),
            ff_Nm=float(ff[k,j]),actuator_Nm=float(force[k,j])))
    # Retain every command and sampled state, plus decomposition for inspection.
    np.savez_compressed(str(stem)+'-servo.npz',t=t,q=q,v=v,q_des=a['q_des'],v_des=a['v_des'],
        ff=ff,position_torque=p,velocity_torque=d,requested_torque=requested,
        ctrl=ctrl,effective_ctrl=effective_ctrl,actuator_torque=force,joint_names=names)
    F.dump(str(stem)+'-startup.json',out)
    return out


def audit():
    m,d=cs.load(ik_margin=0)
    src=ROOT/'studies/runs/20261016-cmpc-foot-motion/20261016-nominal'
    # Analyze saved traces; never integrate or feed scoring state to control.
    c,a=load(src)
    stem=OUT/'20261016-archived-baseline'
    shutil.copyfile(str(src)+'.csv',str(stem)+'.csv')
    shutil.copyfile(str(src)+'-raw.npz',str(stem)+'-raw.npz')
    out=startup_metrics(stem,m)
    q0=np.array([c['qpos'+str(i)][0] for i in range(m.nq)])
    d.qpos[:]=q0;d.qvel[:]=0;d.ctrl[:]=q0[m.jnt_qposadr[m.actuator_trnid[:,0]]]
    mj.mj_forward(m,d)
    out['initial_fixed_position_qacc_max_rad_s2']=float(abs(d.qacc[m.jnt_dofadr[m.actuator_trnid[:,0]]]).max())
    out['initial_contact_count']=int(d.ncon)
    out['initial_min_contact_dist_m']=float(min(con.dist for con in d.contact)) if d.ncon else None
    out['initial_constraint_vertical_force_N']=float(d.qfrc_constraint[2])
    out['initial_q_matches_plan_max_abs']=float(abs(F.cb.mj_to_pin(q0)-np.load(BASE/'xs_elbow_forearm.npy')[0,:34]).max())
    F.dump(OUT/'20261016-startup-audit.json',out)
    print(json.dumps(out),flush=True)


def run(case):
    F.CELL=BASE if case=='baseline' else OUT/'velocity-cell'
    F.run(case)
    plant=Recorder.last;stem=OUT/('20261016-'+case)
    F.dump(str(stem)+'-table-contacts.json',plant.table_contacts)
    print(json.dumps(startup_metrics(stem,plant.m)),flush=True)


def hold():
    from croco.control.command import Command
    fl=F.build();p=Recorder(fl,False);m,d=p.m,p.d
    d.qpos[:]=fl.start_qpos();d.qvel[:]=0;d.ctrl[:]=d.qpos[7:34];d.time=0
    mj.mj_forward(m,d)
    # Identical q0 and estimated joint observation; no approach, no settling.
    st=p.read();p.write(Command.hold(st.q,fl.kp,fl.kd))
    for _ in range(25):p.step(.02)
    stem=OUT/'20261016-fixed-position'
    F.BVS.write_csv(str(stem)+'.csv',np.asarray(p.rows),fl,fl.plan['target'],'fixed-position',17,.5,'Startup ablation; no MPC motion command')
    np.savez_compressed(str(stem)+'-raw.npz',**{k:np.asarray([r[k] for r in p.extra]) for k in p.extra[0]})
    F.dump(str(stem)+'-table-contacts.json',p.table_contacts)
    print(json.dumps(startup_metrics(stem,m)),flush=True)


def solve():
    import croco_replay as cr
    import croco_plan as cp
    cell=OUT/'velocity-cell';cell.mkdir(exist_ok=True)
    for name in ['modes.json','q_elbow+forearm.txt']:shutil.copyfile(BASE/name,cell/name)
    plan=json.loads((BASE/'plan_elbow_forearm.json').read_text())
    # Measured 2026-10-16: contact wrist speed 0.700 -> 0.083 m/s, but
    # startup acceleration remains 223.6 rad/s² and left sole moves 5.55 mm more.
    # Keep this experiment opt-in; it fails the combined transfer criterion.
    plan['w_vel']=1.0
    ocp,_=cr.build_ocp(plan,str(BASE))
    old_x=np.load(BASE/'xs_elbow_forearm.npy')
    assert np.max(abs(ocp.x0-old_x[0]))<1e-12
    # Re-optimize the complete 200-node plan, then use 35-node MPC online.
    problem=ocp.build(dt=.02,n_approach=120,n_braced=80,cones=True)
    solver=F.crocoddyl.SolverBoxFDDP(problem)
    old_u=np.load(BASE/'us_elbow_forearm.npy')
    import time
    start=time.monotonic();ok=solver.solve(list(old_x),list(old_u),200,False,1e-9)
    elapsed=time.monotonic()-start
    rep=cp.report(solver,ocp)
    plan.update(rep,w_vel=1.0,converged=bool(ok),solve_seconds=elapsed,
        source_plan_sha256=F.sha(BASE/'plan_elbow_forearm.json'),
        experiment='motion_transfer: velocity penalty, same original initial state')
    plan['source_commit']=plan.pop('commit')
    plan['source_stage1']=plan.pop('stage1')
    plan['solve_method']='200 BoxFDDP iterations warm-started from archived xs/us; no new stage1'
    F.dump(cell/'plan_elbow_forearm.json',plan)
    np.save(cell/'xs_elbow_forearm.npy',np.array(solver.xs));np.save(cell/'us_elbow_forearm.npy',np.array(solver.us))
    print(json.dumps(plan),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['audit','baseline','fixed-position','solve','velocity']);args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.mode=='audit':audit()
    elif args.mode=='fixed-position':hold()
    elif args.mode=='solve':solve()
    else:run(args.mode)
