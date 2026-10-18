#!/usr/bin/env python3
"""Score the saved geometry ablation, without advancing physics."""
import importlib.util,json,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('rollout',ROOT/'studies/20261018_cmpc_brace_geometry_rollout.py');R=importlib.util.module_from_spec(s);s.loader.exec_module(R)
F,np,mj=R.F,R.np,R.mj
out=dict(status='continue',checkpoint='brace_selection_transfer',rollouts_cumulative=1,rollout_retries=0,
 selection_complete=False,preregistered=json.loads((R.OUT/'20261018-preregistered.json').read_text()),
 geometry=json.loads((R.OUT/'20261018-geometry-probe.json').read_text()),conditions={},
 videos=json.loads((R.OUT/'20261018-videos.json').read_text()))
for label,stem,corrected in [('baseline',R.BASE,False),('cad_wrist',R.OUT/'20261018-cad_wrist',True)]:
 summary=json.loads(Path(str(stem)+'-summary.json').read_text());a=dict(np.load(str(stem)+'-raw.npz'))
 with Path(str(stem)+'.csv').open() as f:
  reader=csv.reader(l for l in f if not l.startswith('#'));head=next(reader);rows=np.array(list(reader),float)
 cols=lambda prefix:[i for i,h in enumerate(head) if h.startswith(prefix)]
 m,d=F.cs.load(ik_margin=0)
 if corrected:R.correct(m)
 table=[];body_force={}
 for k in np.flatnonzero((a['t']>=4)&(a['t']<6))[::10]:
  d.qpos[:]=rows[k,cols('qpos')];d.qvel[:]=rows[k,cols('qvel')];d.ctrl[:]=rows[k,cols('ctrl')];mj.mj_forward(m,d)
  support=0.;by={}
  for i in range(d.ncon):
   c=d.contact[i]
   if 99 not in c.geom:continue
   g=int(c.geom1 if c.geom2==99 else c.geom2);name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g]))
   if 'object' in name:continue
   ff=np.zeros(6);mj.mj_contactForce(m,d,i,ff);support+=ff[0];by[name]=by.get(name,0.)+ff[0]
  table.append(support)
  for name in by:body_force.setdefault(name,[]).append(by[name])
 target=np.array(summary['provenance']['plan_fields']['target']);err=np.linalg.norm(a['point_p'][:,1]-target,axis=1)*1000
 hold=a['t']>=4
 slip={f:float(np.linalg.norm(a['sole_v'][hold&(a['sole_w'][:,i,2]>1),i,:2],axis=1).max()) for i,f in enumerate(F.FEET)}
 out['conditions'][label]=dict(summary=summary,source_csv=str(stem)+'.csv',source_sha256=F.sha(str(stem)+'.csv'),
  reach_final_mm=float(err[-1]),reach_last_second_rms_mm=float(np.sqrt(np.mean(err[a['t']>=5]**2))),
  table_support_hold_min_N=float(min(table)),table_support_hold_max_N=float(max(table)),table_support_samples_N=table,
  table_body_loaded_mean_N={k:float(np.mean(v)) for k,v in body_force.items()},loaded_hold_slip_m_s=slip,
  note='Table support reconstructed at 20 ms from saved q/v/ctrl. Body means condition on that body having a contact record.')
b=out['conditions']['baseline'];c=out['conditions']['cad_wrist']
out['comparison']=dict(reach_change_mm=c['reach_final_mm']-b['reach_final_mm'],
 table_support_change_max_N=float(np.max(abs(np.array(c['table_support_samples_N'])-b['table_support_samples_N']))),
 max_sole_displacement_change_mm=max(abs(c['summary']['feet'][f]['displacement_from_start_max_mm']-b['summary']['feet'][f]['displacement_from_start_max_mm']) for f in F.FEET))
out['hypothesis_supported']=any([abs(out['comparison']['reach_change_mm'])>5,out['comparison']['table_support_change_max_N']>10,out['comparison']['max_sole_displacement_change_mm']>5])
out['remaining']='Re-solve two geometry-corrected physical brace strategies with a frozen feasible-reach/support/motion selection score; evaluate a target excluded from selection. Five rollouts remain including retries. Validate an actually loaded corrected wrist witness and CAD region before force credit. Existing static support calculations use saved physical poses for scoring, not estimated controller inputs.'
F.dump(R.MEDIA/'20261016-cmpc-brace-selection-transfer.json',out)
print(json.dumps({'comparison':out['comparison'],'supported':out['hypothesis_supported'],'reach':[b['reach_final_mm'],c['reach_final_mm']],'support':[b['table_support_hold_min_N'],c['table_support_hold_min_N']]}))
