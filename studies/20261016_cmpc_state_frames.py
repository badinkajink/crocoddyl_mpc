#!/usr/bin/env python3
"""Opt-in frame check and plan-frame adapter; no plant integration or defaults changed.

Run under resguard. The input is the saved estimated observation at 4.50 s
from the nominal foot-motion experiment. A known change of world coordinates
is converted back to the existing plan frame, including the scene contract.
This adapter requires a known frame transform; it does not estimate base bias.
"""
import importlib.util
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('CL_ASSETS_DIR', str(ROOT.parent/'CL_Assets'))
os.environ.setdefault('LEAN_TASK_DIR', str(ROOT/'studies/runs/_stage/mjpc/tasks/humanoid_bench/lean'))
os.environ['TABLE_H'] = '0.885'
import croco_bridge as cb
import numpy as np
import pinocchio as pin
import mujoco
import crocoddyl as cro
import croco_geom as cg

spec = importlib.util.spec_from_file_location('foot', ROOT/'studies/20261016_cmpc_foot_motion.py')
foot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(foot)
OUT = ROOT/'studies/runs/20261016-cmpc-state-frames'
MEDIA = ROOT/'docs/lean/media/20261015-contact-audit'


class WorldFrame:
    """World <- plan rigid transform; Pinocchio base tangent stays body-local."""
    def __init__(self, yaw=0., translation=(0., 0., 0.)):
        self.R = pin.rpy.rpyToMatrix(0., 0., yaw)
        self.t = np.asarray(translation, float)
        if self.t[2] != 0:
            raise ValueError('This check preserves the horizontal floor and gravity.')

    def point(self, p):
        return self.R @ p + self.t

    def pose(self, p):
        return pin.SE3(self.R @ p.rotation, self.point(p.translation))

    def state(self, x, inverse=False):
        y = np.asarray(x).copy()
        r = pin.Quaternion(x[3:7]).matrix()
        y[:3] = self.R.T @ (x[:3]-self.t) if inverse else self.point(x[:3])
        y[3:7] = pin.Quaternion(self.R.T @ r if inverse else self.R @ r).coeffs()
        return y

    def vector(self, v, inverse=False):
        R = self.R.T if inverse else self.R
        return np.concatenate([R @ v[i:i+3] for i in range(0, len(v), 3)])


def scene(ocp):
    return dict(table=pin.SE3(np.eye(3), ocp.table_c),
                target=np.asarray(ocp.reach_target),
                feet={k: v.copy() for k, v in ocp.foot_ref.items()},
                brace={k: v.copy() for k, v in ocp.site_ref.items()})


def transform_scene(s, g):
    return dict(table=g.pose(s['table']), target=g.point(s['target']),
                feet={k: g.pose(v) for k, v in s['feet'].items()},
                brace={k: g.point(v) for k, v in s['brace'].items()})


def scene_error(reference, supplied, g):
    expected = transform_scene(reference, g)
    errors = [np.max(abs(expected['table'].homogeneous-supplied['table'].homogeneous)),
              np.max(abs(expected['target']-supplied['target']))]
    errors += [np.max(abs(expected['feet'][k].homogeneous-supplied['feet'][k].homogeneous)) for k in expected['feet']]
    errors += [np.max(abs(expected['brace'][k]-supplied['brace'][k])) for k in expected['brace']]
    return float(max(errors))


def plan_frame_control(mp, x_world, supplied_scene, g):
    """Opt-in entry point, preserving all original costs and the full horizon.

    The archived plan defines the canonical world. Validate the external scene
    before converting the estimate. Joint commands require no world rotation.
    The caller must retain the MPC warm start in the canonical frame.
    """
    error = scene_error(scene(mp.ocp), supplied_scene, g)
    if error > 1e-9:
        raise ValueError('State/table/target/contact references use different frames')
    u, x1 = mp(len(mp.models)-35, g.state(x_world, inverse=True))
    assert mp.problem.T == 35
    return u, g.state(x1)


def contact_model(original, g):
    dm = original.differential
    contacts = cro.ContactModelMultiple(dm.state, dm.nu)
    for name, item in dm.contacts.contacts.todict().items():
        c = item.contact
        if c.nc == 6:
            new = cro.ContactModel6D(dm.state, c.id, g.pose(c.reference), pin.LOCAL_WORLD_ALIGNED, dm.nu, c.gains)
        else:
            new = cro.ContactModel3D(dm.state, c.id, g.point(c.reference), pin.LOCAL_WORLD_ALIGNED, dm.nu, c.gains)
        contacts.addContact(name, new)
    # Costs do not enter calc's constrained forward dynamics. The complete
    # objective is checked separately through plan_frame_control, including box
    # orientation and cone facets, whose axes belong to the plan frame.
    dynamics = cro.DifferentialActionModelContactFwdDynamics(
        dm.state, dm.actuation, contacts, cro.CostModelSum(dm.state, dm.nu), 1e-4, True)
    return cro.IntegratedActionModelEuler(dynamics, .02)


def evaluate(model, x, u, ocp, s, g):
    d = model.createData()
    model.calc(d, x, u)
    pd = d.differential.multibody.pinocchio
    pin.updateFramePlacements(ocp.rmodel, pd)
    contacts = {}
    for name, c in d.differential.multibody.contacts.contacts.todict().items():
        cm = model.differential.contacts.contacts[name].contact
        p = pd.oMf[cm.id]
        ref = cm.reference
        pos = p.translation-(ref.translation if cm.nc == 6 else ref)
        rot = pin.log3(ref.rotation.T @ p.rotation) if cm.nc == 6 else np.zeros(3)
        contacts[name] = dict(position_plan_m=g.R.T @ pos, rotation_local_rad=rot,
                              wrench_plan=g.vector(np.array(c.f.vector), inverse=True),
                              acceleration_constraint_plan=g.vector(np.array(c.a0), inverse=True))
    reach = g.R.T @ (pd.oMf[ocp.sites['reach']].translation-s['target'])
    table_R = s['table'].rotation
    distances = [cg.sdf_box(table_R.T @ (pd.oMf[p['fid']].translation-s['table'].translation), ocp.table_half)
                 for p in ocp.keepout]
    return dict(acceleration=np.array(d.differential.xout), next_state=np.array(d.xnext),
                reach_residual_m=reach, table_distances_m=np.array(distances), contacts=contacts)


def maxabs(x):
    return float(np.max(np.abs(x)))


def compare(a, b, g, state):
    out = dict(base_linear_acceleration_delta_m_s2=maxabs(a['acceleration'][:3]-b['acceleration'][:3]),
               angular_joint_acceleration_delta_rad_s2=maxabs(a['acceleration'][3:]-b['acceleration'][3:]),
               next_state_tangent_max_abs=maxabs(state.diff(a['next_state'], g.state(b['next_state'], inverse=True))),
               reach_residual_delta_mm=1000*maxabs(a['reach_residual_m']-b['reach_residual_m']),
               table_signed_distance_delta_mm=1000*maxabs(a['table_distances_m']-b['table_distances_m']))
    for key, scale in [('position_plan_m', 1000), ('rotation_local_rad', 1), ('wrench_plan', 1), ('acceleration_constraint_plan', 1)]:
        out[key+'_delta_max'] = max(scale*maxabs(a['contacts'][n][key]-b['contacts'][n][key]) for n in a['contacts'])
    return out


def velocity_check(ocp, x, g):
    # Nonzero six-axis probe avoids the degeneracy of a nearly static hold.
    xp = x.copy(); xp[34:40] = [.12, -.08, .04, .07, -.11, .13]
    q = cb.pin_to_mj(xp[:34]); q[34:] = [1.5, 0., .935, 1., 0., 0., 0.]
    v = np.zeros(39); v[:33] = xp[34:]; R = pin.Quaternion(xp[3:7]).matrix(); v[:3] = R @ v[:3]
    y = g.state(xp); qg = cb.pin_to_mj(y[:34], q); qg[34:37] = g.point(q[34:37])
    qg[37:41] = pin.Quaternion(g.R).coeffs()[[3,0,1,2]]
    vg = v.copy(); vg[:3] = g.R @ v[:3]
    converted = cb.mj_to_pin_v(vg, pin.Quaternion(y[3:7]).matrix())
    dt = 1e-6; qnext = qg.copy()
    mujoco.mj_integratePos(ocp_mj, qnext, vg, dt)
    pin_next = pin.integrate(ocp.rmodel, y[:34], converted*dt)
    return dict(bridge_delta_max=maxabs(converted-xp[34:]),
                integrate_configuration_tangent_error=maxabs(pin.difference(ocp.rmodel, pin_next, cb.mj_to_pin(qnext))),
                probe_base_body_velocity=xp[34:40].tolist(), dt_s=dt,
                transformed_table_pose_m_wxyz=qg[34:].tolist())


def serializable(v):
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, dict): return {k: serializable(x) for k,x in v.items()}
    if isinstance(v, list): return [serializable(x) for x in v]
    return v


def main():
    global ocp_mj
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    fl = foot.build(); mp = fl.mpc; ocp = mp.ocp; ocp_mj = fl.m
    source = ROOT/'studies/runs/20261016-cmpc-foot-motion/20261016-nominal-raw.npz'
    saved = np.load(source); idx = int(np.argmin(abs(saved['pred_t']-4.50)))
    x = saved['pred_x_est'][idx].copy(); k = len(fl.us)-35
    assert k == 165 and fl.dt == .02
    u, x1 = mp(k, x); assert mp.problem.T == 35
    xs, us = [v.copy() for v in mp.xs], [v.copy() for v in mp.us]
    original_scene = scene(ocp)
    identity = WorldFrame(); translate = WorldFrame(0., (.10, -.07, 0.)); yaw = WorldFrame(.47, (.10, -.07, 0.))
    baseline = [evaluate(m, z, c, ocp, original_scene, identity) for m,z,c in zip(mp.problem.runningModels,xs,us)]
    rows = {}; summaries = {}; raw = {'baseline': baseline}; controls = {}
    for name, g, bias in [('translation',translate,0.), ('translation_yaw',yaw,0.), ('base_only_10mm',identity,.01)]:
        transformed_scene = transform_scene(original_scene, g)
        vals = []
        for m,z,c in zip(mp.problem.runningModels,xs,us):
            zm = g.state(z); zm[0] += bias
            vals.append(evaluate(contact_model(m,g), zm, c, ocp, transformed_scene, g))
        raw[name] = vals
        rows[name] = [compare(a,b,g,ocp.state) for a,b in zip(baseline,vals)]
        summaries[name] = {key:max(r[key] for r in rows[name]) for key in rows[name][0]}
        mp.reset(); zm = g.state(x); zm[0] += bias
        t0 = time.monotonic(); un, xn = plan_frame_control(mp, zm, transformed_scene, g)
        controls[name] = dict(torque_delta_max_Nm=maxabs(un-u),
                              joint_command_delta_max_rad=maxabs(xn[7:34]-x1[7:34]),
                              solve_s=mp.solve_times[-1], call_s=time.monotonic()-t0,
                              torque_Nm=un.tolist(), horizon=mp.problem.T,
                              cost=float(mp.solver.cost))
    bad_scene_rejected = False
    try: plan_frame_control(mp, yaw.state(x), original_scene, yaw)
    except ValueError: bad_scene_rejected = True
    feasibility = {}
    costs = mp.problem.runningModels[0].differential.costs.costs.todict()
    for n in baseline[0]['contacts']:
        w = np.array([b['contacts'][n]['wrench_plan'] for b in baseline])
        cone = costs['cone_'+n.removeprefix('brace_')].cost.residual.reference
        cv = w[:,:cone.A.shape[1]] @ cone.A.T
        vio = np.maximum(np.maximum(cone.lb-cv,cv-cone.ub),0)
        feasibility[n] = dict(normal_force_min_N=float(w[:,2].min()), normal_force_max_N=float(w[:,2].max()),
                              tangential_force_max_N=float(np.linalg.norm(w[:,:2],axis=1).max()),
                              cone_row_violation_max=vio.max(axis=0).tolist(),
                              cone_row_units=['N']*5+(['N m']*12 if w.shape[1]==6 and cone.A.shape[1]==6 else []))
    result = dict(checkpoint='state_frames', rollouts=0, horizon=35, dt_s=.02, hold_index=k,
                  hypothesis='A consistent horizontal frame change preserves contact residuals and dynamics; the plan-frame adapter preserves the first joint command. A base-only 10 mm +x error changes relative references and feedback.',
                  thresholds=dict(acceleration=1e-7, force=1e-7, residual_m=1e-9, torque_Nm=1e-6),
                  observation=dict(path='Saved MuJoCoPlant.read -> default_sense(seed=17) -> State -> policy -> pred_x_est',
                                   source=str(source.relative_to(ROOT)), source_sha256=foot.sha(source), index=idx,
                                   simulation_time_s=float(saved['pred_t'][idx]), synthetic_bias=True, ground_truth_input=False),
                  provenance=foot.provenance(fl), transformations=dict(translation_m=translate.t.tolist(), yaw_rad=.47, base_error_m=[.01,0,0]),
                  metrics=summaries, first_control=controls, scene_mismatch_rejected=bad_scene_rejected,
                  velocity_convention_check=velocity_check(ocp,x,yaw), baseline_feasibility=feasibility,
                  numerical_wall_s=time.monotonic()-start,
                  limitations=['No plant integration, slip, recovery or hardware test.',
                               'Raw table keep-out requires axis-aligned world geometry. The adapter uses the archived plan frame for the entire objective, including cone facet axes.',
                               'Frame transform is supplied and checked, not estimated. Base-only error is additional to the saved synthetic estimate.',
                               'Native transformed contact test evaluates the same 35 state/control samples; only canonical-frame solves use the complete objective.'])
    foot.dump(OUT/'20261016-state-frames-raw.json', serializable(dict(input_estimate=x, input_controls=us, input_states=xs, results=raw, comparisons=rows)))
    foot.dump(MEDIA/'20261015-state-frames.json', result)
    print(json.dumps(dict(metrics=summaries,first_control=controls,velocity=result['velocity_convention_check'],wall_s=result['numerical_wall_s']),indent=2),flush=True)


if __name__ == '__main__': main()
