#!/usr/bin/env python3
"""Opt-in estimated-motion dwell before release; run under resguard.

The previous pulse recovery displaced a sole by 8.424 mm and renewed gripper
contact despite feasible first-command force. Delay the same return until
estimated motion passes the prior recovery thresholds for 0.3 s. Preserve
State observations, full 35-node horizon and original site memberships.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('projection',ROOT/'studies/20261017_cmpc_foot_model_transfer.py')
G=importlib.util.module_from_spec(s);s.loader.exec_module(G)
F,np,mj,pin=G.F,G.np,G.mj,G.pin
from croco.control.mpc import MPC
OUT=ROOT/'studies/runs/20261019-cmpc-adaptive-transfer'
MEDIA=G.MEDIA
SHIFT=4.0
RELEASE=5.2
END=9.0

def protocol():
    return json.loads((OUT/'20261019-plan.json').read_text())

class Recorder(F.Recorder):
    def sample(self):
        super().sample()
        table=F.cs.bid(self.m,'table');forces={}
        for i in range(self.d.ncon):
            c=self.d.contact[i];bs=self.m.geom_bodyid[c.geom]
            if table not in bs:continue
            b=int(bs[0] if bs[1]==table else bs[1]);f=np.zeros(6)
            mj.mj_contactForce(self.m,self.d,i,f)
            sign=1 if bs[1]==b else -1
            world=sign*c.frame.reshape(3,3).T@f[:3]
            name=mj.mj_id2name(self.m,mj.mjtObj.mjOBJ_BODY,b)
            forces[name]=forces.get(name,0.)+float(world[2])
        self.extra[-1]['table_force_N']=sum(forces.values())
        self.table.append(dict(t=float(self.d.time),body_vertical_force_N=forces))

def retarget(models,target,state=None):
    for am in models:
        costs=am.differential.costs
        items=costs.costs.todict()
        if 'reach' in items:items['reach'].cost.residual.reference=target
        if state is not None:
            items['stateReg'].cost.residual.reference=state
            costs.changeCostStatus('stateReg',True)
            # Weight 100 is the existing dwell restReg weight, not a fit.
            if 'abortRest' not in items:
                w=np.r_[np.zeros(33),np.ones(33)]
                costs.addCost('abortRest',F.crocoddyl.CostModelResidual(am.state,
                    F.crocoddyl.ActivationModelWeightedQuad(w),
                    F.crocoddyl.ResidualModelState(am.state,state,27)),100.)

def run(case,guarded):
    if not (OUT/'20261019-protocol.json').exists():F.dump(OUT/'20261019-protocol.json',protocol())
    tag=('gated' if guarded else 'baseline')+'-'+case
    fl=F.build();mp=fl.mpc;ocp=mp.ocp
    # Build before the clocked policy; the fixed 35-node return uses the
    # existing feet-only return costs and standing reference.
    problem=ocp.build(dt=.02,n_approach=0,n_braced=0,n_return=35,cones=True)
    ret=MPC(ocp,list(problem.runningModels),problem.terminalModel,horizon=35,iters=1,
        xs_plan=fl.xs[-36:],us_plan=fl.us[-35:],nthreads=1)
    p=Recorder(fl,case=='pulse');p.table=[];m,d=p.m,p.d
    d.qpos[:]=fl.start_qpos();d.qvel[:]=0;d.ctrl[:]=d.qpos[7:34];d.time=0;mj.mj_forward(m,d)
    gate=G.Projector(fl,p);rd=ocp.rmodel.createData();mode='brace';transition=None
    pred=[];records=[];before_rows=[];after_rows=[];forces=[];windows=[];tick=0
    motion=[];quiet_since=None
    motion_data=ocp.rmodel.createData()
    wrist_frame=ocp.rmodel.getFrameId('left_wrist_yaw_link')
    assert wrist_frame<ocp.rmodel.nframes
    initial_target=np.array(fl.plan['target']);target=initial_target.copy()
    def policy(t,st):
        nonlocal mode,transition,target,tick,quiet_since
        start=time.perf_counter()
        q=np.r_[st.base_pos,st.base_quat,st.q];v=np.r_[st.base_linvel,st.base_angvel,st.v]
        x=np.r_[F.cb.mj_to_pin(q),F.cb.mj_to_pin_v(v,F.CT._quat_to_mat(st.base_quat))]
        if t>=SHIFT-1e-9 and target[0]==initial_target[0]:
            target=initial_target+np.array([.02,0,0]);retarget(mp.models+[mp.terminal],target)
        k=min(int(round(t/.02)),165);selected=mp
        # Thresholds are the previous recovery screens; no force sensor input.
        pin.forwardKinematics(ocp.rmodel,motion_data,x[:34],x[34:])
        pin.updateFramePlacements(ocp.rmodel,motion_data)
        speed=lambda fid: pin.getFrameVelocity(ocp.rmodel,motion_data,fid,pin.LOCAL_WORLD_ALIGNED).linear
        sole_speed=[float(np.linalg.norm(speed(ocp.sites[f])[:2])) for f in F.FEET]
        wrist_speed=float(np.linalg.norm(speed(wrist_frame)))
        joint_speed=float(np.max(abs(st.v)))
        quiet=joint_speed<.1 and wrist_speed<.03 and max(sole_speed)<.02
        if quiet:
            if quiet_since is None:quiet_since=float(t)
        else:quiet_since=None
        ready=quiet_since is not None and t-quiet_since>=.3-1e-9
        motion.append(dict(t=float(t),joint_speed_rad_s=joint_speed,wrist_speed_m_s=wrist_speed,
            sole_tangent_speed_m_s=sole_speed,quiet=quiet,quiet_since_s=quiet_since,ready=ready))
        if transition is None and t>=RELEASE-1e-9 and (ready or not guarded):
            ret.xs=list(mp.xs);ret.us=list(mp.us)
            ur,xr=ret(0,x)
            trial=gate.command(ret.models[0],x,st,ur,xr,t,True)
            accepted=trial[3]['status']=='projected'
            transition=dict(t=float(t),release_projection=trial[3],release_accepted=bool(accepted),
                guarded=guarded,motion_dwell=motion[-1],decision='release' if accepted or not guarded else 'abort_braced')
            if accepted or not guarded:mode='return'
            else:
                mode='abort';rest=x.copy();rest[34:]=0
                pin.forwardKinematics(ocp.rmodel,rd,x[:34]);pin.updateFramePlacements(ocp.rmodel,rd)
                abort_target=rd.oMf[ocp.sites['reach']].translation.copy()
                # Cost data must be recreated after adding abortRest.
                retarget(mp.models,abort_target,rest);retarget([mp.terminal],abort_target,rest)
                gate.data.clear()
                braced=mp.models[165:200]
                abort=MPC(ocp,braced,mp.terminal,horizon=35,iters=1,
                    xs_plan=mp.xs,us_plan=mp.us,nthreads=1)
                abort.xs=list(mp.xs);abort.us=list(mp.us)
                policy.abort=abort
                transition['abort_target_estimated_m']=abort_target.tolist()
        if mode=='return':selected=ret;k=0
        elif mode=='abort':selected=policy.abort;k=0
        u,x1=selected(k,x);assert selected.problem.T==35
        cmd,xmodel,total,record,before,after=gate.command(selected.problem.runningModels[0],x,st,u,x1,t,True)
        p0,r0=F.poses(ocp,x,rd);p1,r1=F.poses(ocp,x1,rd)
        record.update(mode=mode,policy_ms=1000*(time.perf_counter()-start),solve_ms=selected.solve_times[-1]*1000)
        pred.append(dict(t=float(t),x_est=x.copy(),x_next=x1.copy(),model_next=xmodel,
            sole_p0=p0,sole_R0=r0,sole_p1=p1,sole_R1=r1,total_torque=total,
            solve_s=selected.solve_times[-1],gate_ms=record['elapsed_ms'],policy_ms=record['policy_ms'],k=k))
        records.append(record);before_rows.extend(before);after_rows.extend(after)
        if tick%5==0:
            for j,(am,xx,uu) in enumerate(zip(selected.problem.runningModels,selected.xs,selected.us)):
                dat=am.createData();am.calc(dat,xx,uu);forces.extend(G.force_rows(am,dat,t,j))
            windows.append(dict(t=float(t),T=selected.problem.T,mode=mode))
        tick+=1;p.phase=0 if mode=='brace' else 1 if mode=='return' else 2;p.cost=selected.solver.cost
        return cmd
    loop=F.ControlLoop(p,policy,stance=None,cfg=F.LoopConfig(ctrl_hz=50,latency_comp=False,realtime=None))
    start=time.monotonic();log=loop.run(fl.kp,fl.kd,max_seconds=END);wall=time.monotonic()-start
    rows=np.asarray(p.rows);stem=OUT/('20261019-'+tag)
    F.BVS.write_csv(str(stem)+'.csv',rows,fl,initial_target,tag,0,END,'90N +y 4.5:4.8' if p.pulse else '')
    a={k:np.asarray([r[k] for r in p.extra]) for k in p.extra[0]}
    a.update({'pred_'+k:np.asarray([r[k] for r in pred]) for k in pred[0]})
    np.savez_compressed(str(stem)+'-raw.npz',**a)
    for suffix,obj in [('predicted-forces',forces),('simulated-contacts',p.contacts),('commands',p.commands),
        ('gate',records),('motion-gate',motion),('executable-before',before_rows),('executable-after',after_rows),('table',p.table)]:
        F.dump(str(stem)+'-'+suffix+'.json',obj)
    summary=F.summarize(fl,p,rows,a,forces,[])
    summary.update(case=tag,wall_s=wall,transition=transition,motion_gate_enabled=guarded,rollout_count=1,periods=len(log),
        window_checks=windows,provenance=F.provenance(fl),standing_reference=fl.start_qpos().tolist(),
        target_original_m=initial_target.tolist(),target_changed_m=(initial_target+[.02,0,0]).tolist(),
        observation='MuJoCoPlant.read -> default_sense(seed=17) -> State; true poses/contacts only in recording and scoring')
    F.dump(str(stem)+'-summary.json',summary)
    print(json.dumps(dict(case=tag,transition=transition,wall_s=wall,pelvis_final_m=summary['pelvis_final_m'])),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['protocol','run']);ap.add_argument('--case',choices=['nominal','pulse'],default='nominal');ap.add_argument('--guarded',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=='protocol':F.dump(OUT/'20261019-protocol.json',protocol())
    else:run(args.case,args.guarded)
