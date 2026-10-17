#!/usr/bin/env python3
"""Opt-in executable-torque cone projection; every invocation uses resguard.

The preceding nominal hold requests tensile forearm force before the pulse.
At t >= 4 s, project the servo's initial total torque into the existing contact
cones, actuator bounds, feedforward bounds and position-input bounds. Retain
the solver's position/velocity command. If the intersection is unavailable,
reject it and use measured joint position, zero velocity and zero feedforward.
That damping fallback has no contact-feasibility guarantee; its violations are
recorded. No contact geometry, friction, horizon or certified default changes.
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'studies/runs/20261017-cmpc-foot-model-transfer'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'
spec = importlib.util.spec_from_file_location('foot_measure', ROOT/'studies/20261016_cmpc_foot_motion.py')
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)
np, mj, pin = F.np, F.mujoco, F.pin
from scipy.optimize import minimize

# The 1e-6 tolerance is numerical, in N or N m for the corresponding cone row.
# The previous nominal violations are 9.13 N and 12.87 N m. No hardware-derived
# compliance/friction/force margin is assumed by this feasibility experiment.
TOL = 1e-6
START = 4.0


def force_rows(am, data, stamp=0., node=0):
    costs=am.differential.costs.costs.todict(); rows=[]
    for name,cd in data.differential.multibody.contacts.contacts.todict().items():
        cone=costs['cone_'+name.removeprefix('brace_')].cost.residual.reference
        w=np.array(cd.f.vector); vals=cone.A@w[:cone.A.shape[1]]
        rows.append(dict(t=float(stamp),node=node,contact=name,wrench_world=w.tolist(),
            cone_values=vals.tolist(),cone_violation=F.violation(vals,cone.lb,cone.ub).tolist()))
    return rows


class Projector:
    def __init__(self, fl, plant):
        self.fl,self.p=fl,plant;self.data={}

    def evaluate(self, am, x, u, stamp):
        if id(am) not in self.data:self.data[id(am)]=am.createData()
        dat=self.data[id(am)];am.calc(dat,x,u)
        return dat,force_rows(am,dat,stamp)

    def project(self, am, x, target, lower, upper, stamp):
        dat,before=self.evaluate(am,x,target,stamp)
        am.calcDiff(dat,x,target)
        mats=[];lo=[];hi=[];offset=[]
        costs=am.differential.costs.costs.todict()
        for name,cd in dat.differential.multibody.contacts.contacts.todict().items():
            cone=costs['cone_'+name.removeprefix('brace_')].cost.residual.reference
            n=cone.A.shape[1];w=np.array(cd.f.vector)[:n]
            J=np.asarray(cd.df_du)[:n]
            A=cone.A@J;b=cone.A@w-A@target
            mats.append(A);offset.append(b);lo.extend(cone.lb-b);hi.extend(cone.ub-b)
        A=np.vstack(mats);lo=np.array(lo);hi=np.array(hi);b=np.concatenate(offset)
        if np.any(lower>upper):
            return None,dict(status='empty_actuator_intersection',iterations=0),before
        # Normalize each inequality by its torque-space gradient. This changes
        # conditioning only; acceptance below uses original N / N m rows.
        scale=np.maximum(np.linalg.norm(A,axis=1),1e-3)
        lowfinite=np.isfinite(lo);hifinite=np.isfinite(hi)
        C=np.vstack((A[lowfinite]/scale[lowfinite,None],-A[hifinite]/scale[hifinite,None]))
        rhs=np.r_[lo[lowfinite]/scale[lowfinite],-hi[hifinite]/scale[hifinite]]
        result=minimize(lambda z:.5*np.sum((z-target)**2),np.clip(target,lower,upper),
            jac=lambda z:z-target,method='SLSQP',bounds=list(zip(lower,upper)),
            constraints=[dict(type='ineq',fun=lambda z:C@z-rhs,jac=lambda z:C)],
            options=dict(maxiter=60,ftol=1e-9,disp=False))
        z=result.x
        actual,after=self.evaluate(am,x,z,stamp)
        cv=max(max(r['cone_violation']) for r in after)
        box=float(F.violation(z,lower,upper).max())
        affine_error=float(np.max(abs(np.concatenate([np.array(r['cone_values']) for r in after])-(A@z+b))))
        accepted=bool(cv<=TOL and box<=TOL and affine_error<=1e-5 and np.isfinite(z).all())
        info=dict(status='projected' if accepted else 'rejected',optimizer_success=bool(result.success),
            optimizer_message=str(result.message),iterations=int(result.nit),cone_row_max=cv,
            box_excess_Nm=box,affine_row_error_max=affine_error,
            torque_change_norm_Nm=float(np.linalg.norm(z-target)))
        return z if accepted else None,info,after

    def command(self, am, x, st, u, x1, stamp, enabled):
        start=time.perf_counter();fl,p=self.fl,self.p;m=p.m
        qcmd=np.clip(x1[7:34],*p.q_range);vcmd=x1[40:].copy()
        # ControlLoop applies its 0.9 feedforward clamp after policy(). The
        # initial four-run audit exposed up to 11.7 N m if this is omitted.
        ff_limit=fl.tau_lim*F.LoopConfig().clamp_ratio
        ff=np.clip(u,-ff_limit,ff_limit)
        pd=fl.kp*(qcmd-st.q)+fl.kd*(vcmd-st.v)
        ctrl=qcmd+(fl.kd*vcmd+ff)/fl.kp
        clipped=np.where(m.actuator_ctrllimited,np.clip(ctrl,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),ctrl)
        raw=fl.kp*(clipped-st.q)-fl.kd*st.v
        target=np.where(m.actuator_forcelimited,np.clip(raw,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1]),raw)
        _,before=self.evaluate(am,x,target,stamp)
        info=dict(t=float(stamp),active=bool(enabled and stamp>=START-1e-9),status='baseline',iterations=0)
        chosen=target.copy()
        if info['active']:
            lower=np.maximum(-ff_limit+pd,m.actuator_forcerange[:,0])
            upper=np.minimum(ff_limit+pd,m.actuator_forcerange[:,1])
            cl=fl.kp*(m.actuator_ctrlrange[:,0]-st.q)-fl.kd*st.v
            cu=fl.kp*(m.actuator_ctrlrange[:,1]-st.q)-fl.kd*st.v
            lower=np.where(m.actuator_ctrllimited,np.maximum(lower,cl),lower)
            upper=np.where(m.actuator_ctrllimited,np.minimum(upper,cu),upper)
            chosen,detail,_=self.project(am,x,target,lower,upper,stamp)
            info.update(detail)
            if chosen is None:
                qcmd=np.clip(st.q,*p.q_range);vcmd=np.zeros(27);ff=np.zeros(27)
                chosen=np.clip(fl.kp*(qcmd-st.q)-fl.kd*st.v,m.actuator_forcerange[:,0],m.actuator_forcerange[:,1])
                info['fallback']='estimated_position_zero_velocity_zero_feedforward'
            else:ff=chosen-pd
        dat,after=self.evaluate(am,x,chosen,stamp)
        info['elapsed_ms']=1000*(time.perf_counter()-start)
        return (qcmd,vcmd,ff),dat.xnext.copy(),chosen,info,before,after


def probe():
    fl=F.build();p=F.Recorder(fl,False);gate=Projector(fl,p);mp=fl.mpc
    result={}
    for case in ['nominal','pulse']:
        a=np.load(ROOT/f'studies/runs/20261016-cmpc-foot-motion/20261016-{case}-raw.npz')
        result[case]=[]
        for stamp in [4.0,4.5,4.8,5.5]:
            idx=int(np.argmin(abs(a['pred_t']-stamp)));x=a['pred_x_est'][idx]
            mp.reset();u,x1=mp(165,x);am=mp.problem.runningModels[0]
            # A saved-state prerequisite check; no simulation time advances.
            dat,rows=gate.evaluate(am,x,u,stamp);am.calcDiff(dat,x,u)
            original={n:(np.array(c.f.vector),np.array(c.df_du)) for n,c in dat.differential.multibody.contacts.contacts.todict().items()}
            du=np.linspace(-.3,.3,27);dat,_=gate.evaluate(am,x,u+du,stamp)
            err=max(float(np.max(abs((np.array(c.f.vector)-original[n][0])[:original[n][1].shape[0]]-original[n][1]@du))) for n,c in dat.differential.multibody.contacts.contacts.todict().items())
            repaired,info,after=gate.project(am,x,u,-fl.tau_lim,fl.tau_lim,stamp)
            info.update(t=stamp,derivative_affine_wrench_error=err,before=rows,after=after)
            result[case].append(info)
            assert err<1e-7
    result['mim_solvers_available']=importlib.util.find_spec('mim_solvers') is not None
    result['provenance']=F.provenance(fl)
    F.dump(OUT/'20261017-probe.json',result)
    print(json.dumps({k:[(r['t'],r['status'],r['cone_row_max']) for r in result[k]] for k in ['nominal','pulse']}))


def run(case, correction):
    tag=('corrected' if correction else 'baseline')+'-'+case
    fl=F.build();mp=fl.mpc;ocp=mp.ocp;p=F.Recorder(fl,case=='pulse');m,d=p.m,p.d
    d.qpos[:]=fl.start_qpos();d.qvel[:]=0;d.ctrl[:]=d.qpos[7:34];d.time=0;mj.mj_forward(m,d)
    mp.reset();rd=ocp.rmodel.createData();gate=Projector(fl,p)
    pred=[];forces=[];windows=[];records=[];before_rows=[];after_rows=[];tick=0;cache={}
    def policy(t,st):
        k=min(int(round(t/.02)),165)
        q=np.r_[st.base_pos,st.base_quat,st.q];v=np.r_[st.base_linvel,st.base_angvel,st.v]
        x=np.r_[F.cb.mj_to_pin(q),F.cb.mj_to_pin_v(v,F.CT._quat_to_mat(st.base_quat))]
        u,x1=mp(k,x);assert mp.problem.T==35
        cmd,xmodel,total,record,before,after=gate.command(mp.problem.runningModels[0],x,st,u,x1,t,correction)
        p0,r0=F.poses(ocp,x,rd);p1,r1=F.poses(ocp,x1,rd);pm,rm=F.poses(ocp,xmodel,rd)
        pred.append(dict(t=float(d.time),x_est=x.copy(),x_next=x1.copy(),model_next=xmodel,
            sole_p0=p0,sole_R0=r0,sole_p1=p1,sole_R1=r1,model_p1=pm,model_R1=rm,
            total_torque=total,solve_s=mp.solve_times[-1],gate_ms=record['elapsed_ms'],k=k))
        records.append(record);before_rows.extend(before);after_rows.extend(after)
        p.phase=k;p.cost=mp.solver.cost
        return cmd
    def audit(row,st,cmd):
        nonlocal tick
        if tick%5==0:
            max_u=max_j=0.
            for j,(am,x,u) in enumerate(zip(mp.problem.runningModels,mp.xs[:-1],mp.us)):
                if id(am) not in cache:cache[id(am)]=am.createData()
                dat=cache[id(am)];am.calc(dat,x,u)
                max_u=max(max_u,float(F.violation(np.asarray(u),am.u_lb,am.u_ub).max()))
                max_j=max(max_j,float(F.violation(np.asarray(x)[7:34],m.jnt_range[p.jids,0],m.jnt_range[p.jids,1]).max()))
                forces.extend(force_rows(am,dat,d.time,j))
            windows.append(dict(t=float(d.time),T=mp.problem.T,control_violation_Nm=max_u,joint_violation_rad=max_j))
        tick+=1
    cfg=F.LoopConfig(ctrl_hz=50,latency_comp=False,realtime=None)
    loop=F.ControlLoop(p,policy,stance=None,cfg=cfg,on_step=audit)
    start=time.monotonic();log=loop.run(fl.kp,fl.kd,max_seconds=6.);wall=time.monotonic()-start
    rows=np.asarray(p.rows);stem=OUT/('20261017-'+tag)
    F.BVS.write_csv(str(stem)+'.csv',rows,fl,fl.plan['target'],tag,0,6.,'90N +y 4.5:4.8' if p.pulse else '')
    a={k:np.asarray([r[k] for r in p.extra]) for k in p.extra[0]}
    a.update({'pred_'+k:np.asarray([r[k] for r in pred]) for k in pred[0]})
    np.savez_compressed(str(stem)+'-raw.npz',**a)
    for suffix,obj in [('predicted-forces',forces),('simulated-contacts',p.contacts),('commands',p.commands),
                       ('gate',records),('executable-before',before_rows),('executable-after',after_rows)]:
        F.dump(str(stem)+'-'+suffix+'.json',obj)
    result=F.summarize(fl,p,rows,a,forces,windows)
    result.update(case=tag,wall_s=wall,rollout_count=1,periods=len(log),horizon_min=35,horizon_max=35,
        solve_mean_ms=float(np.mean(mp.solve_times)*1000),solve_p95_ms=float(np.percentile(mp.solve_times,95)*1000),
        solve_above_20ms=int(np.sum(np.array(mp.solve_times)>.02)),provenance=F.provenance(fl),window_checks=windows,
        observation='MuJoCoPlant.read -> default_sense(seed=17) -> State; same 10 mm/0.01 rad assumptions and ideal joint/velocity channels as foot_motion',
        runtime=dict(iterations=1,threads=1,latency_comp=False),correction=correction)
    F.dump(str(stem)+'-summary.json',result)
    print(json.dumps(dict(case=tag,wall_s=wall,reach_mm=result['reach_final_mm'],gate_status={s:sum(r['status']==s for r in records) for s in set(r['status'] for r in records)})),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['probe','baseline','projected']);ap.add_argument('--case',choices=['nominal','pulse'],default='nominal');a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='probe':probe()
    else:run(a.case,a.mode=='projected')
