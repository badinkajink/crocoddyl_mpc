#!/usr/bin/env python3
"""Score saved trials and reuse the previous physical-contact/video scorer.

No controller or simulation integration is performed. The inherited scorer's
paths and captions are redirected in memory; the original source is unchanged.
Run under resguard, including rendering and final numerical checks.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('experiment',ROOT/'studies/20261017_cmpc_foot_model_transfer.py')
G=importlib.util.module_from_spec(spec);spec.loader.exec_module(G)
np,F,OUT,MEDIA=G.np,G.F,G.OUT,G.MEDIA


def inherited(group,render=False):
    source=ROOT/'studies/20261016_cmpc_foot_analysis.py'
    text=source.read_text().replace("'20261016-'+case",f"'20261017-{group}-'+case")
    text=text.replace("'20261016-foot-'+case",f"'20261017-foot-model-{group}-'+case")
    text=text.replace("f'{case.capitalize()} | Simulation",f"f'{group.capitalize()} {{case}} | Simulation")
    ns=dict(__name__='saved_contact_scorer',__file__=str(source))
    exec(compile(text,str(source), 'exec'),ns)
    dest=OUT/group;dest.mkdir(exist_ok=True)
    ns.update(OUT=OUT,MEDIA=MEDIA if render else dest)
    ns['render' if render else 'analyze']()


def force_summary(rows,lo=4.5):
    result={}
    for name in sorted(set(r['contact'] for r in rows)):
        g=[r for r in rows if r['contact']==name and r['t']>=lo-1e-9]
        w=np.array([r['wrench_world'] for r in g]);cv=np.array([r['cone_violation'] for r in g])
        z=dict(normal_min_N=float(w[:,2].min()),normal_max_N=float(w[:,2].max()),
            tangent_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),
            force_rows_max_N=float(cv[:,:5].max()),negative_normal_samples=int(np.sum(w[:,2]<-G.TOL)),
            violating_samples=int(np.sum(np.max(cv,axis=1)>G.TOL)),samples=len(g))
        if name in F.FEET:
            loaded=w[:,2]>5;cop=np.c_[-w[loaded,4]/w[loaded,2],w[loaded,3]/w[loaded,2]]
            z.update(moment_rows_max_Nm=float(cv[:,5:].max()),cop_margin_min_mm=float(1000*np.min(np.array(F.cb.FOOT_HALF)-abs(cop))) if len(cop) else None)
        result[name]=z
    return result


def finish():
    fl=F.build();results={};arrays={};checks={}
    for group in ['baseline','corrected']:
        inherited_metrics=json.loads((OUT/group/'20261015-foot-motion.json').read_text())
        for case in ['nominal','pulse']:
            tag=group+'-'+case;stem=OUT/('20261017-'+tag)
            a=dict(np.load(str(stem)+'-raw.npz'));arrays[tag]=a
            # Reconstruct issued commands from saved estimated state, including
            # the runtime clamp. Baseline raw predictions retain the v1 audit.
            if group=='baseline':
                saved=json.loads(Path(str(stem)+'-commands.json').read_text())
                repaired=[];total=[];new_forces=[]
                for j,(x,k) in enumerate(zip(a['pred_x_est'],a['pred_k'])):
                    cmd=saved[j];q=x[7:34];v=x[40:]
                    ctrl=np.array(cmd['q'])+(fl.kd*np.array(cmd['v'])+np.array(cmd['tau']))/fl.kp
                    ctrl=np.where(fl.m.actuator_ctrllimited,np.clip(ctrl,fl.m.actuator_ctrlrange[:,0],fl.m.actuator_ctrlrange[:,1]),ctrl)
                    tau=np.clip(fl.kp*(ctrl-q)-fl.kd*v,fl.m.actuator_forcerange[:,0],fl.m.actuator_forcerange[:,1])
                    am=fl.mpc.models[int(k)];dat=am.createData();am.calc(dat,x,tau)
                    repaired.append(dat.xnext.copy());total.append(tau)
                    new_forces.extend(G.force_rows(am,dat,float(a['pred_t'][j])))
                a['pred_model_next']=np.array(repaired);a['pred_total_torque']=np.array(total)
                rd=fl.mpc.ocp.rmodel.createData()
                a['pred_model_p1']=np.array([F.poses(fl.mpc.ocp,x,rd)[0] for x in repaired])
                F.dump(str(stem)+'-executable-recomputed.json',new_forces)
                np.savez_compressed(str(stem)+'-recomputed-model.npz',xnext=a['pred_model_next'],torque=a['pred_total_torque'],sole_p=a['pred_model_p1'])
            gates=json.loads(Path(str(stem)+'-gate.json').read_text())
            after=json.loads(Path(str(stem)+'-executable-after.json').read_text())
            before=json.loads(Path(str(stem)+'-executable-before.json').read_text())
            if group=='baseline':after=new_forces;before=new_forces
            r=inherited_metrics[case];r['executable_force_after_post_pulse']=force_summary(after)
            r['executable_force_before_post_pulse']=force_summary(before)
            r['executable_force_after_hold']=force_summary(after,4.)
            active=[x for x in gates if x['active']];total_ms=1000*a['pred_solve_s']+a['pred_gate_ms']
            r['filter']=dict(status_counts={s:sum(g['status']==s for g in gates) for s in sorted(set(g['status'] for g in gates))},
                active_ticks=len(active),fallback_ticks=sum('fallback' in g for g in gates),
                maximum_affine_row_error=max((g.get('affine_row_error_max',0) for g in gates),default=0),
                gate_p95_ms=float(np.percentile(a['pred_gate_ms'],95)),
                active_gate_p95_ms=float(np.percentile([g['elapsed_ms'] for g in active],95)) if active else None,
                solver_plus_filter_p95_ms=float(np.percentile(total_ms,95)),
                solver_plus_filter_above_20ms=int(np.sum(total_ms>20)))
            residual=np.array([fl.mpc.ocp.state.diff(xp,xn) for xp,xn in zip(a['pred_model_next'][:-1],a['pred_x_est'][1:])])
            for phase,lo,hi in [('hold',4.,7.),('post_pulse',4.5,7.)]:
                mask=(a['pred_t'][:-1]>=lo-1e-9)&(a['pred_t'][1:]<hi)
                rr=residual[mask]
                r['phases'][phase]['state_model_residual']=dict(
                    base_position_rms_mm=float(1000*np.sqrt(np.mean(np.sum(rr[:,:3]**2,axis=1)))),
                    orientation_joint_rms_rad=float(np.sqrt(np.mean(np.sum(rr[:,3:33]**2,axis=1)))),
                    base_linear_velocity_rms_m_s=float(np.sqrt(np.mean(np.sum(rr[:,33:36]**2,axis=1)))),
                    angular_joint_velocity_rms_rad_s=float(np.sqrt(np.mean(np.sum(rr[:,36:]**2,axis=1)))))
                for i,foot in enumerate(F.FEET):
                    er=a['pred_sole_p0'][1:,i]-a['pred_model_p1'][:-1,i]
                    r['phases'][phase][foot]['executable_model_estimated_position_rms_mm']=float(1000*np.sqrt(np.mean(np.sum(er[mask]**2,axis=1))))
                    delta=a['pred_model_p1'][:-1,i]-a['pred_sole_p0'][:-1,i]
                    r['phases'][phase][foot]['predicted_step_motion_max_mm']=float(1000*np.linalg.norm(delta[mask],axis=1).max())
                    ii=np.searchsorted(a['t'],a['pred_t'][:-1]);jj=np.minimum(np.searchsorted(a['t'],a['pred_t'][1:]),len(a['t'])-1)
                    ds=a['sole_p'][jj,i]-a['sole_p'][ii,i]
                    r['phases'][phase][foot]['simulated_step_motion_max_mm']=float(1000*np.linalg.norm(ds[mask],axis=1).max())
            # At each new command, estimated joints/velocities equal plant joints
            # in this observation contract. Thus initial servo realization is
            # independently checked against logged physical actuator force.
            import csv
            with Path(str(stem)+'.csv').open() as f:
                reader=csv.reader(x for x in f if not x.startswith('#'));cols=next(reader);raw=np.array(list(reader),float)
            actual=np.array([raw[:,cols.index('afrc'+str(i))] for i in range(27)]).T
            idx=np.searchsorted(a['t'],a['pred_t']);valid=idx<len(a['t'])
            r['initial_executable_torque_max_error_Nm']=float(np.max(abs(actual[idx[valid]]-a['pred_total_torque'][valid])))
            accepted_times={g['t'] for g in gates if g['status']=='projected'}
            accepted=[x for x in after if x['t'] in accepted_times]
            checks[tag+'_accepted_rows']=max((max(x['cone_violation']) for x in accepted),default=0)<=G.TOL
            checks[tag+'_servo_realization']=r['initial_executable_torque_max_error_Nm']<1e-8
            checks[tag+'_full_horizon']=all(x['T']==35 for x in r['summary']['window_checks'])
            results[tag]=r
    for case in ['nominal','pulse']:
        old=np.load(ROOT/f'studies/runs/20261016-cmpc-foot-motion/20261016-{case}-raw.npz')
        checks[case+'_baseline_reproduction']=bool(np.array_equal(old['pred_x_est'],arrays['baseline-'+case]['pred_x_est']))
        pre=arrays['baseline-'+case]['pred_t']<G.START-1e-9
        checks[case+'_pre_activation_identical']=bool(np.array_equal(arrays['baseline-'+case]['pred_x_est'][pre],arrays['corrected-'+case]['pred_x_est'][pre]))
    base=results['baseline-pulse']['phases']['post_pulse']['sole_left'];proj=results['corrected-pulse']['phases']['post_pulse']['sole_left']
    residual_ratio=proj['executable_model_estimated_position_rms_mm']/base['executable_model_estimated_position_rms_mm']
    reach_delta=results['corrected-nominal']['summary']['reach_final_mm']-results['baseline-nominal']['summary']['reach_final_mm']
    loss_delta=max(results['corrected-nominal']['phases']['hold'][f]['contact_loss_s']-results['baseline-nominal']['phases']['hold'][f]['contact_loss_s'] for f in F.FEET)
    feasible=all(all(x['violating_samples']==0 for x in results['corrected-'+c]['executable_force_after_hold'].values()) for c in ['nominal','pulse'])
    decision=dict(status='checkpoint_complete',adopt_default=False,sliding_model_tested=False,
        pulse_left_residual_ratio=residual_ratio,nominal_final_reach_increase_mm=reach_delta,
        nominal_contact_loss_increase_s=loss_delta,all_issued_hold_model_forces_feasible=feasible,
        hypothesis_passed=bool(residual_ratio<=.9 and reach_delta<=5 and loss_delta<=0 and feasible),
        remaining='Fixed sticking contacts still exclude frictional slip. Executable-torque cone feasibility is a first-node model property. Future horizon forces, unsupported fallback, actuator evolution between ticks and hardware contact geometry remain unresolved.')
    out=dict(checkpoint='foot_model_transfer',date='2026-10-17',plan=json.loads((OUT/'20261017-plan.json').read_text()),
        cases=results,decision=decision,validation=checks,frames=inherited_metrics['frames'],
        collision_sole_geometry=inherited_metrics['collision_sole_geometry'],
        source_hashes={str(p.relative_to(ROOT)):F.sha(p) for p in [Path(__file__),ROOT/'studies/20261017_cmpc_foot_model_transfer.py',ROOT/'studies/20261016_cmpc_foot_motion.py',ROOT/'studies/20261016_cmpc_foot_analysis.py']})
    out['frames']['executable_prediction']='Re-evaluate fixed-contact action model at the estimated state and initial realized servo torque. State residual uses state.diff(model_next, next estimated state); separate translation, orientation/joint, linear-velocity and angular/joint-velocity blocks.'
    F.dump(MEDIA/'20261016-cmpc-foot-model-transfer.json',out)
    print(json.dumps(dict(decision=decision,validation=checks)),flush=True)
    assert all(checks.values()),checks


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['score','render','finish']);ap.add_argument('--group',choices=['baseline','corrected'],default='baseline');a=ap.parse_args()
    if a.mode=='finish':finish()
    else:inherited(a.group,a.mode=='render')
