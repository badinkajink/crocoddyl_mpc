#!/usr/bin/env python3
"""Run and score `Lean Simple H12 Magpie` rollouts.

The question this harness exists to answer is narrow on purpose: given a
REQUESTED contact mode -- a subset of {elbow, forearm, palm} -- does MJPC drive
the robot from a stand into a lean that establishes exactly that mode, keep both
feet planted, and get the reaching hand closer to the target than standing still
would?

Three things it does differently from the S12 harness, each because that page
had to retract a claim built on the missing version:

  1. REACH IS SCORED AS A GAIN, never as a residual. The baseline is the same
     hand's distance to the same target at t = 0, standing. S12 §2 reported
     0.348-0.551 m of "reach" for eight rollouts whose true gain was -0.041 m.
  2. FRAMES ARE RE-SOLVED WITH THE LOGGED COMMAND. S12 §1 re-solved every frame
     with `d.ctrl` left at zero against position servos -- i.e. commanding every
     joint to angle 0 -- and every contact-force number on the page was that
     robot's, not the rollout's.
  3. CONTACT IS REPORTED AS A DUTY OVER A WINDOW, not as "what was touching on
     the last frame". A seed that grazes the table 7% of the time and one that
     holds a brace look identical in the second measure.

usage:
  simple_lean.py run   --out DIR [--modes elbow+forearm,...] [--seeds N] ...
  simple_lean.py score CSV [CSV ...]
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build/bin/testspeed")
TASK = "Lean Simple H12 Magpie"
MODEL = os.path.join(ROOT, "build/mjpc/tasks/humanoid_bench/lean/"
                           "Lean_Simple_H12_Magpie.xml")

# Candidate brace links: cost-term name -> the body whose contacts count.
LINKS = {
    "elbow": "left_shoulder_yaw_link",
    "forearm": "left_elbow_link",
    "palm": "left_magpie_gripper",
}
TRUNK = "torso_link"
WEIGHT = {"elbow": "Brace Elbow", "forearm": "Brace Forearm",
          "palm": "Brace Palm"}

# Seat-surface geometry, identical to lean_simple.cc's kBraceLinks (and to
# seat_calib.py, which fitted it). Kept in sync by hand; seat_calib.py is the
# check that it still reads zero where MuJoCo makes contact.
#
# `prims=True` means "every collidable primitive geom on the body", which is
# what the gripper needs: its jaws hang 80 mm below the palm box and touch the
# slab first.
SEAT = {
    "elbow": dict(body="left_shoulder_yaw_link",
                  seg=[(0.002, -0.007, -0.030), (0.002, -0.007, -0.182)],
                  r=0.045),
    "forearm": dict(body="left_elbow_link",
                    seg=[(0.020, -0.010, -0.015), (0.110, -0.030, -0.015)],
                    r=0.035),
    "palm": dict(body="left_magpie_gripper", prims=True),
}
TRUNK_BODIES = ["torso_link", "pelvis"]

FALL_PELVIS_Z = 0.60      # pelvis this low is on the floor, not leaning


# --------------------------------------------------------------------------- #
# model helpers
# --------------------------------------------------------------------------- #
def load():
    m = mujoco.MjModel.from_xml_path(MODEL)
    return m, mujoco.MjData(m)


def slab(m, d):
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    c, s = d.geom_xpos[g], m.geom_size[g]
    return dict(top=c[2] + s[2], cx=c[0], cy=c[1], hx=s[0], hy=s[1])


def clearance(sl, xy, z, inset=0.0):
    ex = max(0.0, abs(xy[0] - sl["cx"]) - (sl["hx"] - inset))
    ey = max(0.0, abs(xy[1] - sl["cy"]) - (sl["hy"] - inset))
    e = np.hypot(ex, ey)
    gap = z - sl["top"]
    return np.hypot(e, max(0.0, gap)) if e > 0 else gap


def geom_low(m, d, g):
    """World z of a primitive geom's lowest point, and its xy."""
    p, R, s = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
    t = m.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return p[:2], p[2] - s[0]
    if t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        e = [p + R @ np.array([0, 0, sg * s[1]]) for sg in (1, -1)]
        q = min(e, key=lambda v: v[2])
        return q[:2], q[2] - s[0]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        c = [p + R @ (np.array(v) * s)
             for v in [(i, j, k) for i in (-1, 1) for j in (-1, 1)
                       for k in (-1, 1)]]
        q = min(c, key=lambda v: v[2])
        return q[:2], q[2]
    return p[:2], p[2]


def body_prim_clearance(m, d, sl, body, inset=0.0):
    """min clearance over a body's collidable primitives; None if it has none."""
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    best = None
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b:
            continue
        if not m.geom_contype[g] and not m.geom_conaffinity[g]:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "keepaway" in name:
            continue
        xy, z = geom_low(m, d, g)
        c = clearance(sl, xy, z, inset)
        best = c if best is None else min(best, c)
    return best


def seat_gap(m, d, sl, spec, inset=0.06):
    if spec.get("prims"):
        c = body_prim_clearance(m, d, sl, spec["body"], inset)
        return 0.6 if c is None else c
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec["body"])
    R, o = d.xmat[b].reshape(3, 3), d.xpos[b]
    pts = [o + R @ np.array(q) for q in spec["seg"]]
    p = min(pts, key=lambda q: q[2])
    return clearance(sl, p[:2], p[2] - spec["r"], inset)


def body_geoms(m, body):
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    return set(g for g in range(m.ngeom) if m.geom_bodyid[g] == b)


def touching(m, d, geoms, table):
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist > 0:
            continue
        if (c.geom1 == table and c.geom2 in geoms) or \
           (c.geom2 == table and c.geom1 in geoms):
            return True
    return False


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def load_traj(path):
    """Returns (column index, rows, header dict).

    The header dict carries the run's PROVENANCE -- the --weights and --numeric
    strings testspeed was given. Scoring reads the reach target from there, not
    from the XML: a scorer that takes the target from the model while the run
    moved it with --numeric measures the distance to a target nobody used, and
    reports a rollout that finished 4 mm from its actual target as +0.068 m of
    gain. Older CSVs without these lines fall back to the XML default.
    """
    meta, body = {}, []
    with open(path) as f:
        for l in f:
            if l.startswith("#"):
                for tok in l[1:].strip().split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        meta.setdefault(k, v)
                if l.startswith("# weights=") or l.startswith("# numerics="):
                    k, v = l[2:].rstrip("\n").split("=", 1)
                    meta[k] = v
            else:
                body.append(l)
    r = csv.reader(body)
    hdr = next(r)
    rows = np.array([[float(v) for v in row] for row in r])
    return {n: i for i, n in enumerate(hdr)}, rows, meta


def target_from_meta(meta, default):
    """The reach target the run actually used."""
    for item in (meta.get("numerics") or "").split(","):
        if "=" not in item:
            continue
        k, v = item.split("=", 1)
        if k.strip() == "reach_target":
            vals = [float(x) for x in v.split("|")]
            out = np.array(default, dtype=float)
            out[:len(vals)] = vals
            return out
    return np.asarray(default, dtype=float)


def score(path, stride=5, settle_frac=0.25):
    col, rows, meta = load_traj(path)
    if len(rows) < 10:
        return dict(path=path, error="trajectory too short (%d rows)" % len(rows))
    m, d = load()
    nq, nv, nu = m.nq, m.nv, m.nu
    qi = [col["qpos%d" % i] for i in range(nq)]
    vi = [col["qvel%d" % i] for i in range(nv)]
    ui = [col["ctrl%d" % i] for i in range(nu)]
    ai = [col["afrc%d" % i] for i in range(nu)]

    table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top_collision")
    geoms = {k: body_geoms(m, b) for k, b in LINKS.items()}
    geoms["trunk"] = body_geoms(m, TRUNK)
    hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_hand")
    wrist = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
    n = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "reach_target")
    target = target_from_meta(
        meta, m.numeric_data[m.numeric_adr[n]:m.numeric_adr[n] + 3])
    taumax = np.abs(m.actuator_forcerange[:, 1])
    taumax[taumax == 0] = 1.0

    # standing baseline: the SAME hand, the SAME target, at the home key.
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    base_err = float(np.linalg.norm(d.site_xpos[hand] - target))
    base_wrist = float(np.linalg.norm(
        d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ [0.13, 0, 0] - target))

    idx = range(0, len(rows), stride)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    ser = {k: [] for k in ("t", "reach", "reach_wrist", "pelvis_z", "pelvis_x",
                           "cost", "gap_elbow", "gap_forearm", "gap_palm",
                           "trunk_gap", "tau", "pitch", "hand_x", "tau_who")}
    con = {k: [] for k in list(LINKS) + ["trunk"]}
    for k in idx:
        d.qpos[:] = rows[k, qi]
        d.qvel[:] = rows[k, vi]
        d.ctrl[:] = rows[k, ui]           # position servos: ctrl == commanded angle
        mujoco.mj_forward(m, d)
        sl = slab(m, d)
        ser["t"].append(rows[k, col["time"]])
        ser["cost"].append(rows[k, col["cost"]])
        ser["reach"].append(float(np.linalg.norm(d.site_xpos[hand] - target)))
        ser["reach_wrist"].append(float(np.linalg.norm(
            d.xpos[wrist] + d.xmat[wrist].reshape(3, 3) @ [0.13, 0, 0] - target)))
        ser["pelvis_z"].append(float(d.qpos[2]))
        ser["pelvis_x"].append(float(d.qpos[0]))
        R = d.xmat[torso].reshape(3, 3)
        ser["pitch"].append(float(np.degrees(np.arctan2(R[0, 2], R[2, 2]))))
        ser["hand_x"].append(float(d.site_xpos[hand][0]))
        for name in LINKS:
            ser["gap_%s" % name].append(float(seat_gap(m, d, sl, SEAT[name])))
        ser["trunk_gap"].append(float(min(
            body_prim_clearance(m, d, sl, b, 0.0) for b in TRUNK_BODIES)))
        ser["tau"].append(float(np.max(np.abs(rows[k, ai]) / taumax)))
        ser["tau_who"].append(int(np.argmax(np.abs(rows[k, ai]) / taumax)))
        for name, gs in geoms.items():
            con[name].append(bool(touching(m, d, gs, table)))

    ser = {k: np.asarray(v) for k, v in ser.items()}
    con = {k: np.asarray(v) for k, v in con.items()}
    ns = max(1, int(len(ser["t"]) * settle_frac))
    sl_ = slice(-ns, None)

    # command churn: path length of the commanded pose against its net move.
    u = rows[:, ui]
    path_len = float(np.abs(np.diff(u, axis=0)).sum())
    net = float(np.abs(u[-1] - u[0]).sum())

    fell = bool((ser["pelvis_z"] < FALL_PELVIS_Z).any())
    achieved = sorted(k for k in LINKS if con[k][sl_].mean() >= 0.5)
    out = dict(
        path=os.path.basename(path),
        t_end=float(ser["t"][-1]),
        fell=fell,
        fall_time=(float(ser["t"][np.argmax(ser["pelvis_z"] < FALL_PELVIS_Z)])
                   if fell else None),
        duty={k: float(con[k][sl_].mean()) for k in con},
        duty_all={k: float(con[k].mean()) for k in con},
        achieved_mode=achieved,
        reach_base=base_err,
        reach_settled=float(ser["reach"][sl_].mean()),
        reach_best=float(ser["reach"].min()),
        reach_gain=float(base_err - ser["reach"][sl_].mean()),
        reach_gain_best=float(base_err - ser["reach"].min()),
        reach_base_wrist=base_wrist,
        reach_gain_wrist=float(base_wrist - ser["reach_wrist"][sl_].mean()),
        gap_settled={k: float(ser["gap_%s" % k][sl_].mean()) for k in LINKS},
        trunk_gap_min=float(ser["trunk_gap"].min()),
        pelvis_z_settled=float(ser["pelvis_z"][sl_].mean()),
        pelvis_x_settled=float(ser["pelvis_x"][sl_].mean()),
        pelvis_x_start=float(ser["pelvis_x"][0]),
        torso_pitch_settled=float(ser["pitch"][sl_].mean()),
        torso_pitch_max=float(np.abs(ser["pitch"]).max()),
        hand_x_settled=float(ser["hand_x"][sl_].mean()),
        peak_tau_ratio=float(ser["tau"].max()),
        peak_tau_joint=str(mujoco.mj_id2name(
            m, mujoco.mjtObj.mjOBJ_ACTUATOR,
            int(ser["tau_who"][int(np.argmax(ser["tau"]))]))),
        saturated_frac=float((ser["tau"] > 0.99).mean()),
        churn=float(path_len / net) if net > 1e-9 else None,
        cost_settled=float(ser["cost"][sl_].mean()),
    )
    return out, ser, con


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def mode_weights(mode, seat=300.0):
    """'elbow+forearm' -> the --weights string that requests exactly that mode."""
    want = set(mode.split("+")) if mode != "none" else set()
    bad = want - set(LINKS)
    if bad:
        raise SystemExit("unknown link(s) in mode '%s': %s" % (mode, bad))
    return ",".join("%s=%g" % (WEIGHT[k], seat if k in want else 0.0)
                    for k in LINKS)


def run_one(out_csv, mode, seconds, threads, seat=300.0, extra_weights="",
            numerics="", start_key="", log=None, stride=5):
    w = mode_weights(mode, seat)
    if extra_weights:
        w += "," + extra_weights
    cmd = [BIN, "--task=%s" % TASK, "--total_time=%g" % seconds,
           "--planner_thread=%d" % threads, "--dump_traj=%s" % out_csv,
           "--dump_stride=%d" % stride, "--weights=%s" % w]
    if numerics:
        cmd.append("--numeric=%s" % numerics)
    if start_key:
        cmd.append("--start_key=%s" % start_key)
    with open(log or os.devnull, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return r.returncode, " ".join(cmd)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--modes", default="elbow+forearm")
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--seconds", type=float, default=20.0)
    r.add_argument("--threads", type=int, default=12)
    r.add_argument("--seat", type=float, default=300.0)
    r.add_argument("--weights", default="", help="extra Name=val,... overrides")
    r.add_argument("--numeric", default="", help="planner numerics to override")
    r.add_argument("--start-key", default="")
    r.add_argument("--tag", default="")

    s = sub.add_parser("score")
    s.add_argument("csv", nargs="+")
    s.add_argument("--json", default="")

    a = ap.parse_args()

    if a.cmd == "run":
        os.makedirs(a.out, exist_ok=True)
        results = []
        for mode in a.modes.split(","):
            for seed in range(a.seeds):
                tag = "%s%s_s%d" % (a.tag + "_" if a.tag else "",
                                    mode.replace("+", "-"), seed)
                csv_path = os.path.join(a.out, tag + ".csv")
                rc, cmd = run_one(csv_path, mode, a.seconds, a.threads, a.seat,
                                  a.weights, a.numeric, a.start_key,
                                  log=os.path.join(a.out, tag + ".log"))
                print("[%s] rc=%d" % (tag, rc), flush=True)
                if rc != 0 or not os.path.exists(csv_path):
                    results.append(dict(tag=tag, mode=mode, seed=seed,
                                        error="run failed"))
                    continue
                out, _, _ = score(csv_path)
                out.update(tag=tag, mode=mode, seed=seed, cmd=cmd)
                results.append(out)
                print("   mode=%-16s achieved=%-22s gain=%+.3f m fell=%s"
                      % (mode, "+".join(out["achieved_mode"]) or "none",
                         out["reach_gain"], out["fell"]), flush=True)
        with open(os.path.join(a.out, "summary%s.json"
                               % ("_" + a.tag if a.tag else "")), "w") as f:
            json.dump(results, f, indent=1)
    else:
        allout = []
        for p in a.csv:
            out, _, _ = score(p)
            allout.append(out)
            print(json.dumps(out, indent=1))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(allout, f, indent=1)


if __name__ == "__main__":
    main()
