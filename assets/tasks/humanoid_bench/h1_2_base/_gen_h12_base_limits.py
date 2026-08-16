"""Import the H1-2 joint limits from CL_Assets into the MJPC planner base.

CL_Assets IS THE SOURCE OF TRUTH for the robot (2026-08-03). Its mujoco_assets/
h1_2_handless.xml matches h12_safety_layer/core/joint_limits.py exactly on all 27
motorised joints -- both URDF_POSITION_LIMITS (joint range) and URDF_TORQUE_LIMITS
(actuatorfrcrange). The old base, vendored from correlllab/h1_mujoco@ef83c84, did not:
it predates the arm service that changed the physical limits, so it carried

    left_elbow          -2.53 .. 1.6    vs the real -0.95 .. 3.18   (~pi/2 offset)
    left_shoulder_yaw   -3.01 .. 2.66   vs the real -2.66 .. 3.01   (L/R swapped)
    torso               -3.14 .. 1.57   vs the real -2.35 .. 2.35
    knee                -0.12 .. 2.05   vs the real -0.12 .. 2.19
    arm actuatorfrcrange 120/120/75/25  vs the real 40/18/18/19

...i.e. the planner could solve poses the machine cannot reach. Running this at BUILD
time, against the fetched CL_Assets tree, is what keeps CL the source of truth: there is
no second copy of the numbers to drift, and therefore no parity test to maintain -- the
import IS the check, and it is a HARD FAILURE (see die() below) rather than a warning.

WHAT THIS DOES *NOT* TOUCH, and why the base is not simply replaced by CL's file:
the MJPC base carries planning/sim2real work that has no upstream home --
  - <position> actuators + kp/kv (CL ships torque <motor>s with no ctrlrange; MJPC's
    ctrl vector IS the planner's decision variable and must be in joint-target space),
  - back_equipment (0.67 kg, real mass on the torso),
  - ankle condim=4 friction="1 0.06 0.0001",
  - the torso/shoulder <exclude> pairs.
So the split is: CL owns kinematics, inertials and limits; MJPC owns actuation and
contact. This script moves exactly the first group across.

ctrlrange is rewritten to equal the imported joint range. It has to move in lockstep --
otherwise the planner keeps commanding into the retired envelope -- and pinning it to the
joint range also fixes a pre-existing inconsistency where right_shoulder_roll had
ctrlrange="-3.4 0.19" against a joint range of "-3.4 0.38".

FORCERANGE (2026-08-16, icra2026 merge). Actuator <position forcerange> is the
torque the planner is allowed to ASK for, which is a different question from
actuatorfrcrange (what the joint can take) and is NOT derivable from CL_Assets --
it is a mission budget. icra2026 carried it as a hand-written hunk in
h1_2_modified_magpie.xml.patch; that hunk cannot survive this import, because its
context lines quote the ctrlranges this script rewrites, so it REJECTED -- and the
magpie patch is applied through apply_patch_tolerant.cmake, which discards the
.rej. The build therefore succeeded while silently shipping a model with no
forcerange at all. Moving the budget here makes it derived, attributed and
hard-failing like every other number in this file.

  --forcerange none   (default) leave <position> forcerange alone.
  --forcerange urdf   forcerange = CL_Assets actuatorfrcrange. What the hardware
                      can take; the planner may ask for all of it.
  --forcerange real   the budget validated on the REAL robot for the braced lean
                      (icra2026 49e3591 "model: explicit per-joint actuator
                      forceranges", reproduced byte-for-byte by REAL_FORCERANGE
                      below). NOT a ratio of anything shipped: the implied
                      ratio/URDF-torque set matches no safety-layer config
                      (checked against default/tight/relax_safety_full.yaml), so
                      the torques themselves are the primary record.

SCOPING. Pass --forcerange only for the magpie/lean model. The shared base feeds
stabilize, stand, upper and walk, all tuned against an unbudgeted actuator, and
icra2026's hunk landed after those derivations for exactly that reason.

usage: _gen_h12_base_limits.py <CL h1_2_handless.xml> <MJPC base in> <out>
                               [--forcerange none|urdf|real]
"""
import re
import sys

# Joints that must be present in BOTH files. Mirrors h12_safety_layer MOTOR_COUNT=27.
EXPECTED_MOTORS = 27

# --forcerange real. Verbatim from icra2026 49e3591; keyed by joint name so it
# cannot be silently mis-applied by index the way a patch hunk could.
REAL_FORCERANGE = {
    "left_hip_yaw_joint": 54.0,      "right_hip_yaw_joint": 54.0,
    "left_hip_pitch_joint": 117.0,   "right_hip_pitch_joint": 117.0,
    "left_hip_roll_joint": 180.0,    "right_hip_roll_joint": 180.0,
    "left_knee_joint": 270.0,        "right_knee_joint": 270.0,
    "left_ankle_pitch_joint": 48.6,  "right_ankle_pitch_joint": 48.6,
    "left_ankle_roll_joint": 32.4,   "right_ankle_roll_joint": 32.4,
    "torso_joint": 36.0,
    "left_shoulder_pitch_joint": 28.8, "right_shoulder_pitch_joint": 28.8,
    "left_shoulder_roll_joint": 28.8,  "right_shoulder_roll_joint": 28.8,
    "left_shoulder_yaw_joint": 13.0,   "right_shoulder_yaw_joint": 13.0,
    "left_elbow_joint": 13.0,          "right_elbow_joint": 13.0,
    "left_wrist_roll_joint": 8.6,      "right_wrist_roll_joint": 8.6,
    "left_wrist_pitch_joint": 8.6,     "right_wrist_pitch_joint": 8.6,
    "left_wrist_yaw_joint": 8.6,       "right_wrist_yaw_joint": 8.6,
}


def die(msg):
    sys.stderr.write("_gen_h12_base_limits: FATAL: %s\n" % msg)
    sys.exit(1)


def read_cl_limits(path):
    """name -> (range, actuatorfrcrange) for every actuated joint in the CL model."""
    try:
        with open(path) as f:
            src = f.read()
    except IOError as exc:
        die("cannot read CL_Assets model %s (%s). The h12 FetchContent must point at "
            "correlllab/CL_Assets." % (path, exc))

    limits = {}
    for tag in re.findall(r"<joint\b[^>]*/?>", src):
        name = re.search(r'\bname="([^"]+)"', tag)
        rng = re.search(r'\brange="([^"]+)"', tag)
        frc = re.search(r'\bactuatorfrcrange="([^"]+)"', tag)
        if name and rng and frc:
            limits[name.group(1)] = (rng.group(1), frc.group(1))

    if len(limits) != EXPECTED_MOTORS:
        die("CL_Assets model %s yielded %d limited joints, expected %d. The upstream "
            "model changed shape -- reconcile before bumping MUJOCO_MPC_CL_ASSETS_GIT_TAG."
            % (path, len(limits), EXPECTED_MOTORS))
    return limits


def forcerange_for(name, basis, cl_frc):
    """Torque budget for one actuator, or None to leave the attribute alone."""
    if basis == "none":
        return None
    if basis == "urdf":
        return cl_frc                      # already a "lo hi" pair from CL_Assets
    if name not in REAL_FORCERANGE:
        die("--forcerange real has no budget for %r. REAL_FORCERANGE must cover "
            "every actuated joint; reconcile it against the model before building."
            % name)
    tau = REAL_FORCERANGE[name]
    return "%.6g %.6g" % (-tau, tau)


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    basis = "none"
    for i, a in enumerate(sys.argv):
        if a == "--forcerange":
            if i + 1 >= len(sys.argv):
                die("--forcerange needs a value: none|urdf|real")
            basis = sys.argv[i + 1]
            argv = [x for x in argv if x != basis]
    if basis not in ("none", "urdf", "real"):
        die("unknown --forcerange %r (want none|urdf|real)" % basis)
    if len(argv) != 3:
        die("usage: %s <CL h1_2_handless.xml> <base in> <out> "
            "[--forcerange none|urdf|real]" % sys.argv[0])
    cl_path, in_path, out_path = argv

    limits = read_cl_limits(cl_path)
    with open(in_path) as f:
        xml = f.read()

    seen_joints = set()
    seen_ctrl = set()

    def sub_attr(tag, attr, value):
        if re.search(r'\b%s="[^"]*"' % attr, tag):
            return re.sub(r'\b%s="[^"]*"' % attr, '%s="%s"' % (attr, value), tag, count=1)
        # attribute absent -> insert it right after the name, keeping tags readable
        return re.sub(r'(\bname="[^"]+")', r'\1 %s="%s"' % (attr, value), tag, count=1)

    def fix_joint(m):
        tag = m.group(0)
        name = re.search(r'\bname="([^"]+)"', tag)
        if not name or name.group(1) not in limits:
            return tag  # free joint / unlimited helper joint: left alone
        rng, frc = limits[name.group(1)]
        seen_joints.add(name.group(1))
        tag = sub_attr(tag, "range", rng)
        return sub_attr(tag, "actuatorfrcrange", frc)

    def fix_ctrl(m):
        tag = m.group(0)
        joint = re.search(r'\bjoint="([^"]+)"', tag)
        if not joint or joint.group(1) not in limits:
            return tag
        jname = joint.group(1)
        seen_ctrl.add(jname)
        # ctrlrange tracks the joint range: the planner commands joint targets.
        tag = sub_attr(tag, "ctrlrange", limits[jname][0])
        frc = forcerange_for(jname, basis, limits[jname][1])
        return tag if frc is None else sub_attr(tag, "forcerange", frc)

    xml = re.sub(r"<joint\b[^>]*/?>", fix_joint, xml)
    xml = re.sub(r"<position\b[^>]*/?>", fix_ctrl, xml)

    missing_j = sorted(set(limits) - seen_joints)
    missing_c = sorted(set(limits) - seen_ctrl)
    if missing_j:
        die("%d joint(s) in CL_Assets have no counterpart in %s: %s"
            % (len(missing_j), in_path, ", ".join(missing_j)))
    if missing_c:
        die("%d joint(s) have no <position> actuator in %s: %s"
            % (len(missing_c), in_path, ", ".join(missing_c)))

    with open(out_path, "w") as f:
        f.write(xml)
    sys.stderr.write("_gen_h12_base_limits: imported %d joint limits from CL_Assets"
                     "; forcerange basis=%s\n" % (len(limits), basis))


if __name__ == "__main__":
    main()
