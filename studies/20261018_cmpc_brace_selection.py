#!/usr/bin/env python3
"""Opt-in physical brace-region prerequisite and static support comparison.

Run under resguard in the croco environment. No plant integration or default
changes. Saved poses are design/scoring inputs, never controller observations.
A region is the collision boundary within 0.5 mm of actual CAD triangles whose
normal is within 25 degrees of the collision normal. No finite pad area or
normal-axis moment capacity is inferred. Forces remain at individual witnesses.
"""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('CL_ASSETS_DIR',str(ROOT.parent/'CL_Assets'))
os.environ.setdefault('LEAN_TASK_DIR',str(ROOT/'studies/runs/_stage/mjpc/tasks/humanoid_bench/lean'))
import croco_bridge as cb
import importlib.util,json,time,hashlib,csv
import numpy as np
import mujoco as mj
import contact_select as cs
import quadprog
from PIL import Image,ImageDraw

OUT=ROOT/'studies/runs/20261018-cmpc-brace-selection'
MEDIA=ROOT/'docs/lean/media/20261015-contact-audit'
def module(name,file):
 s=importlib.util.spec_from_file_location(name,ROOT/'studies'/file);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
S=module('surface','20261016_surface_catalogue.py')
def dump(p,o):
 with Path(p).open('w') as f:json.dump(o,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def nearest(point,tri):
 """Exact closest CAD point and incident triangle; batches bounded at 1024."""
 best=(float('inf'),None,None)
 for first in range(0,len(tri),1024):
  t=tri[first:first+1024];a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a
  n=np.cross(ab,ac);nn=np.sum(n*n,1);pa=point-a
  proj=point-n*(np.sum(pa*n,1)/np.maximum(nn,1e-30))[:,None];ap=proj-a
  d00=np.sum(ab*ab,1);d01=np.sum(ab*ac,1);d11=np.sum(ac*ac,1)
  d20=np.sum(ap*ab,1);d21=np.sum(ap*ac,1);den=d00*d11-d01*d01
  v=(d11*d20-d01*d21)/np.maximum(den,1e-30);w=(d00*d21-d01*d20)/np.maximum(den,1e-30)
  dist=np.linalg.norm(proj-point,axis=1);dist[(nn<1e-25)|(v<0)|(w<0)|(v+w>1)]=np.inf
  k=int(dist.argmin())
  if dist[k]<best[0]:best=(float(dist[k]),proj[k],first+k)
  for x,y in [(a,b),(b,c),(c,a)]:
   e=y-x;u=np.clip(np.sum((point-x)*e,1)/np.maximum(np.sum(e*e,1),1e-30),0,1)
   p=x+u[:,None]*e;dist=np.linalg.norm(p-point,axis=1);k=int(dist.argmin())
   if dist[k]<best[0]:best=(float(dist[k]),p[k],first+k)
 return best

# These are collision bodies/geoms, not nearest named sites. F uses the actual
# proximal forearm mesh as well as its capsule only where CAD agreement passes.
STRATEGIES={'elbow_forearm':[41,43,44],'forearm_wrist':[43,44,50,51,52]}

def qp(m,d,contacts):
 """Hard equilibrium, unilateral pyramid and joint torque limits (point forces)."""
 n=3*len(contacts);A=[]
 for c in contacts:
  J=np.zeros((3,m.nv));mj.mj_jac(m,d,J,None,np.array(c['point_world_m']),c['body_id'])
  A.append(J[:,:33].T@np.array(c['frame_rows']).T)
 if not n:return dict(feasible=False,reason='No physical contacts')
 A=np.hstack(A);g=d.qfrc_bias[:33].copy();lim=cs.torque_limits(m)
 H=2*(A[6:].T@np.diag(1/lim**2)@A[6:]+np.eye(n)*1e-5)
 a=2*A[6:].T@(g[6:]/lim**2);C=[x for x in A[:6]];rhs=list(g[:6])
 for i,c in enumerate(contacts):
  e=np.zeros(n);e[3*i]=1;C.append(e);rhs.append(0.)
  mu=min(.6,c['mu'])/np.sqrt(2)
  for j in (1,2):
   for sign in (-1,1):
    e=np.zeros(n);e[3*i]=mu;e[3*i+j]=sign;C.append(e);rhs.append(0.)
 for j in range(27):C.extend([A[6+j],-A[6+j]]);rhs.extend([g[6+j]-lim[j],-g[6+j]-lim[j]])
 try:
  sol=quadprog.solve_qp(H,a,np.array(C).T,np.array(rhs),6)[0]
 except ValueError as e:return dict(feasible=False,reason=str(e),contact_count=len(contacts))
 tau=g[6:]-A[6:]@sol;forces=sol.reshape(-1,3)
 return dict(feasible=True,contact_count=len(contacts),base_residual=float(np.linalg.norm(A[:6]@sol-g[:6])),
  max_torque_ratio=float(np.max(abs(tau)/lim)),effort=float(np.sum((tau/lim)**2)),
  normal_min_N=float(forces[:,0].min()),brace_support_N=float(sum(f[0] for f,c in zip(forces,contacts) if c['is_brace'])),
  forces_contact_N=forces.tolist(),torques_Nm=tau.tolist())

def probe():
 start=time.monotonic();m=mj.MjModel.from_xml_path(str(S.MODELS['staged']));d=mj.MjData(m)
 cat=json.loads((MEDIA/'20261015-surface-catalogue.json').read_text());audit=json.loads(S.AUDIT.read_text())
 source_ok={str(p):sha(p)==h for p,h in cat['required_staged_xml_sha256'].items()};assert all(source_ok.values())
 arrays={k:S.arr_sha(getattr(m,k))==h for k,h in cat['required_staged_array_sha256'].items()};assert all(arrays.values())
 table=S.idx(m,'GEOM','table_top_collision');mesh={}
 for p in cat['patches']:
  g=p['surface_geom_id'];v,f=S.mesh_triangles(m,g);mesh[p['collision_geom']['id']]=(v[f],p['body'],p['old_site'])
 # Additional wrist collision meshes must pass their own body CAD check.
 for g in (50,51):
  if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH:
   v,f=S.mesh_triangles(m,g);mesh[g]=(v[f],S.name(m,'BODY',m.geom_bodyid[g]),'wrist')
 result=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
  hypothesis='Body-specific collision witnesses admit a physical CAD region and feasible point-force support for at least two brace strategies.',
  correction='Require CAD boundary distance <=0.5 mm and normal agreement <=25 degrees; preserve each witness point and lever arm.',
  tolerance_basis='0.5 mm and 25 degrees inherited from validated catalogue; geometry tolerances, not measured hardware compliance.',
  region_definition='Implicit union of collision-boundary points within 0.5 mm of a same-body CAD triangle with outward normal within 25 degrees. No smoothing across holes or other bodies.',
  region_capacity='Point forces only. No area, pressure limit, torsional moment or hardware load rating is credited.',
  fixed_baseline='elbow_forearm',strategies=STRATEGIES,source_checks=source_ok,array_checks=arrays,
  rollout_count=0,physics_steps=0,poses=[])
 poses=list(audit['poses'])
 rollout_module=module('rollout_geometry','20261018_cmpc_brace_geometry_rollout.py')
 fields=['geom_type','geom_dataid','geom_pos','geom_quat','geom_size','geom_rbound','geom_aabb']
 original={k:np.array(getattr(m,k)[52],copy=True) for k in fields}
 for label,path,corrected in [('dynamic_original',rollout_module.BASE,False),('dynamic_cad_wrist',OUT/'20261018-cad_wrist',True)]:
  with Path(str(path)+'.csv').open() as f:
   reader=csv.reader(l for l in f if not l.startswith('#'));head=next(reader);a=np.array(list(reader),float)
  k=int(np.argmin(abs(a[:,head.index('time')]-5.)))
  poses.append(dict(mode=label,qpos=a[k,[i for i,h in enumerate(head) if h.startswith('qpos')]].tolist(),height=.885,corrected=corrected))
 for po in poses:
  for k in fields:getattr(m,k)[52]=original[k]
  cs.set_table_face(m,d,po.get('height',.985))
  if po.get('corrected'):rollout_module.correct(m)
  d.qpos[:]=po['qpos'];d.qvel[:]=0;mj.mj_forward(m,d);records=[];feet=[]
  for i in range(d.ncon):
   c=d.contact[i];gs=list(c.geom);bs=m.geom_bodyid[c.geom]
   if table not in gs:
    floor=[g for g in gs if m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE]
    if not floor or c.dist>0:continue
    g=next(g for g in gs if g!=floor[0]);b=int(m.geom_bodyid[g]);sgn=1 if gs[1]==g else -1
    if 'ankle_roll' not in S.name(m,'BODY',b):continue
    feet.append(dict(body_id=b,point_world_m=c.pos.tolist(),frame_rows=(sgn*c.frame.reshape(3,3)).tolist(),mu=float(c.friction[0]),is_brace=False))
    continue
   first=c.geom2==table;g=int(c.geom1 if first else c.geom2)
   if g not in mesh:continue
   b=int(m.geom_bodyid[g]);R=d.xmat[b].reshape(3,3);outward=c.frame[:3]*(1 if first else -1)
   witness=c.pos-.5*c.dist*outward;local=R.T@(witness-d.xpos[b]);normal=R.T@outward
   tri,body,site=mesh[g];dist,p,k=nearest(local,tri)
   # At a corner several CAD triangles can share the closest point. Take the
   # largest outward-normal agreement among triangles within 1 micrometre of it.
   near=[]
   centers=tri.mean(1);radii=np.linalg.norm(tri-centers[:,None],axis=2).max(1)
   for j in np.flatnonzero(np.linalg.norm(centers-local,axis=1)<=radii+dist+1e-6):
    dd,_,_=nearest(local,tri[j:j+1])
    if dd<=dist+1e-6:near.append(j)
   ns=np.cross(tri[near,1]-tri[near,0],tri[near,2]-tri[near,0]);ns/=np.maximum(np.linalg.norm(ns,axis=1)[:,None],1e-30)
   j=int(np.argmax(ns@normal));angle=float(np.degrees(np.arccos(np.clip(ns[j]@normal,-1,1))))
   old=np.asarray(cs.SITES[site][1]);old_world=d.xpos[b]+R@old
   force=-100*outward
   rec=dict(contact_index=i,body=body,body_id=b,geom_id=g,geom_name=S.name(m,'GEOM',g),signed_gap_mm=float(c.dist*1000),
    point_world_m=witness.tolist(),point_body_m=local.tolist(),normal_body=normal.tolist(),cad_point_body_m=p.tolist(),
    nearest_triangle=int(k),cad_distance_mm=dist*1000,normal_difference_deg=angle,
    region_member=bool(dist<=.0005 and angle<=25),closed=bool(c.dist<=0),
    old_site=site,old_point_body_m=old.tolist(),old_point_distance_mm=float(np.linalg.norm(local-old)*1000),
    lever_arm_body_m=local.tolist(),old_site_100N_moment_error_Nm=float(np.linalg.norm(np.cross(old_world-witness,force))),
    frame_rows=((-1 if first else 1)*c.frame.reshape(3,3)).tolist(),mu=float(c.friction[0]),is_brace=True,
    table_height_error_mm=float(1000*((c.pos+.5*c.dist*outward)[2]-d.geom_xpos[table,2]-m.geom_size[table,2])))
   records.append(rec)
  strategies={}
  for strategy,geoms in STRATEGIES.items():
   # Baseline support solve uses the same actual contact positions but ignores
   # CAD agreement. The correction removes physically unsupported proxies.
   raw=[r for r in records if r['geom_id'] in geoms and r['closed']]
   valid=[r for r in raw if r['region_member']]
   strategies[strategy]=dict(ungated=qp(m,d,feet+raw),geometry_gated=qp(m,d,feet+valid),
    raw_brace_witnesses=len(raw),usable_brace_witnesses=len(valid),rejected_geom_ids=[r['geom_id'] for r in raw if not r['region_member']])
  result['poses'].append(dict(mode=po['mode'],table_height_m=po.get('height',.985),geometry_corrected=po.get('corrected',False),qpos=po['qpos'],contacts=records,physical_floor_witnesses=len(feet),strategies=strategies))
 result['elapsed_s']=time.monotonic()-start
 dump(OUT/'20261018-geometry-probe.json',result)
 for po in result['poses']:
  print(po['mode'],[(r['geom_id'],round(r['cad_distance_mm'],3),round(r['normal_difference_deg'],2),r['region_member']) for r in po['contacts']],flush=True)
  print(po['strategies'],flush=True)

if __name__=='__main__':OUT.mkdir(parents=True,exist_ok=True);probe()
