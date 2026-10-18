#!/usr/bin/env python3
"""Opt-in physical-point strategy comparison. Run every mode under resguard.

Geometry and controller changes are process-local. No catalogue facet is given
an area or moment capacity. A point force needs same-body CAD/witness agreement.
"""
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def module(n, p):
    s = importlib.util.spec_from_file_location(n, ROOT/'studies'/p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
G = module('brace_regions', '20261018_cmpc_brace_selection.py')
R = module('brace_geometry', '20261018_cmpc_brace_geometry_rollout.py')
F, np, mj, cs = R.F, R.np, R.mj, R.F.cs
import croco_replay as cr
import croco_plan as cp

OUT = ROOT/'studies/runs/20261018-cmpc-brace-planning'
MEDIA = R.MEDIA
BASE = F.CELL
SUBSETS = {'elbow_forearm':['elbow','forearm'], 'forearm_wrist':['forearm','wrist'], 'elbow':['elbow']}
GEOMS = {'elbow':[41], 'forearm':[43,44], 'wrist':[50,51,52]}
SURFACES = {41:40, 43:42, 44:42, 50:49, 51:49, 52:49}
OLD_SITES = {s:(b,np.array(o).copy()) for s,(b,o) in cs.SITES.items()}
ORIGINAL_LOAD = cs.load
cs.TABLE_H = .885

def load(*a, **kw):
    m,d = ORIGINAL_LOAD(*a, **kw); R.correct(m); mj.mj_forward(m,d); return m,d
cs.load = load
F.cb._MJ_CACHE = None

def dump(p, o): F.dump(p, o)
def apply_sites(sites):
    for name,v in sites.items(): cs.SITES[name] = (v['body'], np.array(v['point_body_m']))
    assert set(cs.SITES) == set(OLD_SITES)

def close_points(m,d,subset,target):
    """Remove submillimetre soft-IK gaps before assigning a brace force.

    Eight Newton corrections preserve feet and reach to first order. No dynamic
    simulation or controller observation is involved. Final collision checks
    decide availability; this is not a load-capacity certificate.
    """
    before=d.qpos.copy();steps=[]
    for _ in range(8):
        mj.mj_forward(m,d);J=[];e=[]
        for foot in cs.FEET:
            b=cs.bid(m,foot);jp=np.zeros((3,m.nv));jr=np.zeros((3,m.nv))
            mj.mj_jac(m,d,jp,jr,d.xpos[b],b);J.extend([jp[:,:33],jr[:,:33]]);e.extend([np.zeros(3),np.zeros(3)])
        J.append(cs.point_jac(m,d,cs.REACH_BODY,cs.REACH_OFF)[:,:33]);e.append(target-cs.point_world(m,d,cs.REACH_BODY,cs.REACH_OFF))
        for s in subset:
            b,o=cs.SITES[s];J.append(cs.point_jac(m,d,b,o)[2:3,:33]);e.append(np.array([cs.table_top_z(m,d)-.0002-cs.point_world(m,d,b,o)[2]]))
        jac=np.vstack(J)
        dq=np.linalg.solve(jac.T@jac+1e-6*np.eye(33),jac.T@np.concatenate(e))
        scale=min(1.,.02/max(np.max(abs(dq)),1e-12));full=np.zeros(m.nv);full[:33]=dq*scale
        mj.mj_integratePos(m,d.qpos,full,1.);steps.append(float(np.max(abs(dq*scale))))
    mj.mj_forward(m,d)
    return dict(max_joint_change_rad=float(np.max(abs(d.qpos[7:34]-before[7:34]))),step_max=steps,
                site_height_mm={s:float(1000*(cs.point_world(m,d,*cs.SITES[s])[2]-cs.table_top_z(m,d))) for s in subset})

class Geometry:
    def __init__(self, m):
        self.m=m; self.mesh={}
        for g,src in SURFACES.items():
            v,f=G.S.mesh_triangles(m,src); self.mesh[g]=v[f]
        self.cache={}
    def cad(self,g,p,n):
        # Exact CAD distance, plus best normal among equally near triangles at
        # edges/corners. Cache only identical local witnesses and normal bins.
        key=(g,tuple(np.round(p,9)),tuple(np.round(n,6)))
        if key in self.cache:return self.cache[key]
        tri=self.mesh[g];dist,near,k=G.nearest(p,tri)
        centers=tri.mean(1);radii=np.linalg.norm(tri-centers[:,None],axis=2).max(1)
        ids=np.flatnonzero(np.linalg.norm(centers-p,axis=1)<=radii+dist+1e-6)
        candidates=[j for j in ids if G.nearest(p,tri[j:j+1])[0]<=dist+1e-6]
        ns=np.cross(tri[candidates,1]-tri[candidates,0],tri[candidates,2]-tri[candidates,0])
        ns/=np.maximum(np.linalg.norm(ns,axis=1)[:,None],1e-30)
        dot=ns@n;angle=float(np.degrees(np.arccos(np.clip(max(dot),-1,1))))
        out=dict(cad_distance_mm=float(1000*dist),normal_difference_deg=angle,
                 cad_triangle=int(k),cad_point_body_m=near.tolist(),
                 region_member=bool(dist<=.0005 and angle<=25))
        self.cache[key]=out;return out
    def contacts(self,d,sites=None):
        m=self.m;table=G.S.idx(m,'GEOM','table_top_collision');records=[]
        for i in range(d.ncon):
            c=d.contact[i];gs=list(c.geom);is_brace=table in gs
            floor=[g for g in gs if m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE]
            if not is_brace and not floor:continue
            env=table if is_brace else floor[0];g=int(gs[0] if gs[1]==env else gs[1]);b=int(m.geom_bodyid[g])
            if is_brace and g not in SURFACES:continue
            name=G.S.name(m,'BODY',b)
            if not is_brace and 'ankle_roll' not in name:continue
            first=gs[0]==g;outward=c.frame[:3]*(1 if first else -1)
            rotation=d.xmat[b].reshape(3,3);witness=c.pos-.5*c.dist*outward
            p=rotation.T@(witness-d.xpos[b]);n=rotation.T@outward
            ff=np.zeros(6);mj.mj_contactForce(m,d,i,ff)
            r=dict(geom_id=g,body_id=b,body=name,geom_name=G.S.name(m,'GEOM',g),is_brace=is_brace,
                   point_world_m=witness.tolist(),point_body_m=p.tolist(),normal_body=n.tolist(),
                   lever_arm_body_m=p.tolist(),frame_rows=((-1 if first else 1)*c.frame.reshape(3,3)).tolist(),
                   mu=float(c.friction[0]),signed_gap_mm=float(c.dist*1000),closed=bool(c.dist<=0),
                   normal_N=float(ff[0]),tangent_N=float(np.linalg.norm(ff[1:3])),
                   physical_cone_excess_N=float(max(0,np.linalg.norm(ff[1:3])-c.friction[0]*ff[0])))
            if is_brace:
                r.update(self.cad(g,p,n))
                r['plane_error_mm']=float(1000*((c.pos+.5*c.dist*outward)[2]-d.geom_xpos[table,2]-m.geom_size[table,2]))
                r['sites']={}
                for s,v in (sites or {}).items():
                    if v['body']!=name:continue
                    delta=np.array(v['point_body_m'])-p
                    r['sites'][s]=dict(point_error_mm=float(1000*np.linalg.norm(delta)),
                        moment_error_per_100N_Nm=float(np.linalg.norm(np.cross(delta,-100*n))),
                        correct_geom=g in GEOMS[s])
            else:r['region_member']=True
            records.append(r)
        return records

def design():
    OUT.mkdir(parents=True,exist_ok=True)
    cat=json.loads((MEDIA/'20261015-surface-catalogue.json').read_text())
    # Proximal forearm and upper-arm end points almost coincide at the elbow:
    # an undamped closure moved joints 0.16 rad while leaving a 0.39 mm gap.
    # The catalogue's distal forearm candidate separates these force points.
    seeds={'elbow':'upper_arm_elbow_end','forearm':'forearm_distal','wrist':'wrist_housing_underside'}
    points={s:dict(body=next(p for p in cat['patches'] if p['id']==k)['body'],
                   point_body_m=next(p for p in cat['patches'] if p['id']==k)['local_surface_point_m']) for s,k in seeds.items()}
    prereg=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        failure='Seven wrist-box witnesses miss CAD by 8.637–23.515 mm; corrected saved pose has zero closed wrist witnesses.',
        hypothesis='Selecting a feasible physical brace strategy reduces the frozen reach/support/motion score by at least 10% versus fixed elbow_forearm on the reserved target.',
        fixed_baseline='elbow_forearm',selection_target_m=[1.06,-.2348,.9982],reserved_target_m=[.98,-.2348,.9982],
        target_basis='1.06 m is the preceding measured condition; 0.98 m is a previously studied target with different historical mode preference. Neither is a hardware uncertainty range.',
        score='Reject when hold reach RMS >30 mm, CAD-valid closed selected-body support minimum <20 N, lost loaded floor contact >20 ms in hold, or any sampled hold predicted cone force violation >1 N / moment violation >0.1 N m. Otherwise score = reach RMS/30 mm + 20 N/minimum support + integral(sum(joint_speed^2) dt)/10 rad^2/s. Lower is better. Ties retain baseline.',
        thresholds='Task-screen assumptions; no measured hardware loading or speed limits. Hold is 4–6 s. Motion integral is whole episode.',
        selection_rule='Choose on 1.06 m only, freeze winner before running 0.98 m. If no candidate passes, reject selection and report raw tradeoffs; never credit a rejected force.',
        strategy_rule='Try forearm_wrist design against fixed elbow_forearm. If either requested wrist/forearm body lacks a closed CAD-valid design witness, use elbow-only as alternate. No tuning from reserved results.',
        surface_region='Same-body collision boundary within 0.5 mm of CAD and 25 degrees of a closest outward triangle normal; individual point forces, no finite pad area or torsional capacity.',
        geometry='Named wrist box replaced in memory by existing CAD convex collision mesh, both planner geometry queries and plants.',
        observation='default_sense(seed=17); only archived design q_star seeds offline geometry IK; rollout truth used for scoring/rendering.',
        horizon=35,dt_s=.02,rollouts_before_session=1,planned_rollouts=4,max_total_rollouts=6)
    dump(OUT/'20261018-protocol.json',prereg)
    m,d=cs.load();geo=Geometry(m)
    qseed=np.loadtxt(BASE/'q_elbow+forearm.txt')
    results={}
    for strategy in ['elbow_forearm','forearm_wrist','elbow']:
        subset=SUBSETS[strategy];sites={s:dict(v) for s,v in points.items()};apply_sites(sites)
        d.qpos[:]=qseed;d.qpos[2]-=.06;mj.mj_forward(m,d)
        floor=[c.dist for c in d.contact if any(m.geom_type[g]==mj.mjtGeom.mjGEOM_PLANE for g in c.geom)]
        assert floor
        d.qpos[2]-=min(floor)+.0002;mj.mj_forward(m,d)
        ik=cs.solve_ik(m,d,np.array(prereg['selection_target_m']),subset,iters=120)
        closure=close_points(m,d,subset,np.array(prereg['selection_target_m']))
        rec=geo.contacts(d,sites)
        # Freeze each force frame at a same-body collision witness, after CAD
        # validation. The union region permits other witnesses at later poses.
        for s in subset:
            valid=[r for r in rec if r['geom_id'] in GEOMS[s] and r['closed'] and r['region_member']]
            if valid:
                r=min(valid,key=lambda r:np.linalg.norm(np.array(r['point_body_m'])-sites[s]['point_body_m']))
                sites[s]=dict(body=r['body'],point_body_m=r['point_body_m'],normal_body=r['normal_body'],geom_id=r['geom_id'])
        apply_sites(sites);rec=geo.contacts(d,sites)
        closed=[r for r in rec if r['closed'] and r['region_member'] and (not r['is_brace'] or any(r['geom_id'] in GEOMS[s] for s in subset))]
        available=all(any(r['geom_id'] in GEOMS[s] and r['closed'] and r['region_member'] for r in rec) for s in subset)
        qp=G.qp(m,d,closed)
        out=dict(strategy=strategy,subset=subset,sites=sites,q_star=d.qpos.tolist(),ik=ik,closure=closure,contacts=rec,qp=qp,geometry_available=available)
        results[strategy]=out
        print(strategy,'available',available,'reach',ik['reach'],'QP',qp.get('feasible'),'support',qp.get('brace_support_N'),flush=True)
    alternate='forearm_wrist' if results['forearm_wrist']['geometry_available'] else 'elbow'
    result=dict(strategies=results,compared=['elbow_forearm',alternate],protocol_sha256=F.sha(OUT/'20261018-protocol.json'))
    dump(OUT/'20261018-design.json',result)

def solve(strategy, target):
    design=json.loads((OUT/'20261018-design.json').read_text());assert strategy in design['compared']
    spec=design['strategies'][strategy];apply_sites(spec['sites'])
    protocol=json.loads((OUT/'20261018-protocol.json').read_text());xyz=protocol[target+'_target_m']
    cell=OUT/(strategy+'-'+target);cell.mkdir(exist_ok=True)
    np.savetxt(cell/'q_elbow+forearm.txt',spec['q_star'])
    modes=json.loads((BASE/'modes.json').read_text());modes['target']=xyz
    dump(cell/'modes.json',modes)
    plan=json.loads((BASE/'plan_elbow_forearm.json').read_text())
    plan.update(target=xyz,subset=spec['subset'],physical_sites=spec['sites'],strategy=strategy,
                experiment='brace_selection_transfer physical-point correction',geometry_corrected=True)
    ocp,_=cr.build_ocp(plan,str(cell));problem=ocp.build(dt=.02,n_approach=120,n_braced=80,cones=True)
    solver=F.crocoddyl.SolverBoxFDDP(problem);xs=np.load(BASE/'xs_elbow_forearm.npy');us=np.load(BASE/'us_elbow_forearm.npy')
    start=time.monotonic();ok=solver.solve(list(xs),list(us),200,False,1e-9);elapsed=time.monotonic()-start
    report=cp.report(solver,ocp);plan.update(report,converged=bool(ok),solve_seconds=elapsed)
    dump(cell/'plan_elbow_forearm.json',plan);np.save(cell/'xs_elbow_forearm.npy',np.array(solver.xs));np.save(cell/'us_elbow_forearm.npy',np.array(solver.us))
    # Verify the force frame agrees with the intended MuJoCo body offset.
    m,d=cs.load(ik_margin=0);d.qpos[:]=spec['q_star'];mj.mj_forward(m,d)
    rd=ocp.rmodel.createData();F.pin.forwardKinematics(ocp.rmodel,rd,ocp.q_star);F.pin.updateFramePlacements(ocp.rmodel,rd)
    parity={s:float(np.linalg.norm(rd.oMf[ocp.sites[s]].translation-cs.point_world(m,d,v['body'],v['point_body_m']))) for s,v in spec['sites'].items()}
    dump(cell/'20261018-frame-parity.json',parity)
    # Measured seed/model frame disagreement is 0.318 um, below a 1 um frame
    # check and the inherited 500 um CAD-region tolerance.
    assert max(parity.values())<1e-6, parity
    print(strategy,target,'solved',ok,'seconds',elapsed,'reach',report['reach_err'],'frame parity',max(parity.values()),flush=True)

def run(strategy,target):
    design=json.loads((OUT/'20261018-design.json').read_text());apply_sites(design['strategies'][strategy]['sites'])
    F.CELL=OUT/(strategy+'-'+target);F.OUT=OUT
    F.run(strategy+'-'+target)
    for p in OUT.glob('20261016-'+strategy+'-'+target+'*'):p.rename(p.with_name(p.name.replace('20261016','20261018',1)))

def verify(strategy,target):
    design=json.loads((OUT/'20261018-design.json').read_text());spec=design['strategies'][strategy];apply_sites(spec['sites'])
    cell=OUT/(strategy+'-'+target);plan=json.loads((cell/'plan_elbow_forearm.json').read_text());ocp,_=cr.build_ocp(plan,str(cell))
    m,d=cs.load(ik_margin=0);d.qpos[:]=spec['q_star'];mj.mj_forward(m,d)
    rd=ocp.rmodel.createData();F.pin.forwardKinematics(ocp.rmodel,rd,ocp.q_star);F.pin.updateFramePlacements(ocp.rmodel,rd)
    parity={s:dict(pin=rd.oMf[ocp.sites[s]].translation.tolist(),mujoco=cs.point_world(m,d,v['body'],v['point_body_m']).tolist(),error_m=float(np.linalg.norm(rd.oMf[ocp.sites[s]].translation-cs.point_world(m,d,v['body'],v['point_body_m'])))) for s,v in spec['sites'].items()}
    dump(cell/'20261018-frame-parity.json',parity)
    assert max(v['error_m'] for v in parity.values())<1e-6
    print(parity,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['design','solve','run','verify']);p.add_argument('--strategy');p.add_argument('--target',choices=['selection','reserved']);a=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if a.mode=='design':design()
    elif a.mode=='solve':solve(a.strategy,a.target)
    elif a.mode=='verify':verify(a.strategy,a.target)
    else:run(a.strategy,a.target)
