#!/usr/bin/env python3
"""Generate the 31-35 lean pipeline JSONs from the <30 pre-lean twins.
Single source of truth: each pipeline phase = strat-6 leg-core + the stage's
arm/lean overrides. Run by the build (mjpc/tasks/CMakeLists.txt) and by hand.
"""
import json, os, copy
HERE = os.path.dirname(os.path.abspath(__file__))

def load(name): return json.load(open(os.path.join(HERE, name + ".json")))
def dump(name, phases):
    json.dump(phases, open(os.path.join(HERE, name + ".json"), "w"), indent=4)

CORE = {"Foot Left Up":2000.0,"Foot Right Up":2000.0,"Foot Stability":15.0,
        "Hip Yaw L":20.0,"Hip Roll L":20.0,"Hip Yaw R":20.0,"Hip Roll R":20.0,
        "Lateral Center":150.0,"Symmetry":200.0,"Angular Momentum":5.0,"Balance":2.5,
        "CoM Vel.":10.0,"Velocity":0.625,"Joint Vel.":0.01,"Joint Vel. Limit":5.0,
        "Waist Yaw":30.0,"Control":0.05,"Posture":12.0}
NO_CONTACT = [{"body1":-1,"body2":-1,"geom1":-1,"geom2":-1,
               "local_pos1":[0,0,0],"local_pos2":[0,0,0]} for _ in range(5)]

def phase(src_json, idx, name, sustain, ramp):
    """Build one ladder phase: src's terminal weight block, leg-core re-anchored."""
    w = copy.deepcopy(src_json[-1]["weight"])
    w.update(CORE)                      # enforce the standing-core anchor
    p = copy.deepcopy(src_json[-1])
    p["weight"] = w
    p["name"] = name
    p["success_sustain_time"] = sustain
    p["target_ramp_sec"] = ramp
    p.setdefault("contacts", NO_CONTACT)
    return p

def build(prefix):
    # prefix is "h12_" (base/magpie) or "h12_hands_" (hands). Pre-lean files are
    # "<prefix>simple_*"; generated ladder files are "<prefix>lean_*".
    stand  = load(prefix + "simple_stand")
    reach  = load(prefix + "simple_reach")
    cbal   = load(prefix + "simple_counterbalance")
    brace  = load(prefix + "simple_forearm_brace")
    # ladder phase definitions: (src, phase-name, lead-in sustain, ramp)
    P_stand = phase(stand, 0, "stand_up",                5.0, 1.5)
    P_reach = phase(reach, 1, "reach_to_target",         4.0, 2.0)
    P_cbal  = phase(cbal,  2, "counterbalance_standing", 4.0, 2.0)
    P_brace = phase(brace, 3, "forearm_brace_lean",      4.0, 15.0)
    held = lambda p: {**copy.deepcopy(p), "success_sustain_time": 9999.0}
    out = prefix + "lean_"   # "h12_lean_" or "h12_hands_lean_"
    # cumulative-prefix ladder; last phase of each truncation is held
    dump(out + "stand",          [held(P_stand)])
    dump(out + "reach",          [P_stand, held(P_reach)])
    dump(out + "counterbalance", [P_stand, P_reach, held(P_cbal)])
    dump(out + "brace",          [P_stand, P_reach, P_cbal, held(P_brace)])
    dump(out + "full",           [P_stand, P_reach, P_cbal, P_brace])  # 35: all finite

build("h12_")          # base / magpie names: h12_lean_*
build("h12_hands_")    # hands names: h12_hands_lean_*
print("generated 31-35 ladder JSONs")
