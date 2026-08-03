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

usage: _gen_h12_base_limits.py <CL h1_2_handless.xml> <MJPC base in> <out>
"""
import re
import sys

# Joints that must be present in BOTH files. Mirrors h12_safety_layer MOTOR_COUNT=27.
EXPECTED_MOTORS = 27


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


def main():
    if len(sys.argv) != 4:
        die("usage: %s <CL h1_2_handless.xml> <base in> <out>" % sys.argv[0])
    cl_path, in_path, out_path = sys.argv[1:4]

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
        seen_ctrl.add(joint.group(1))
        # ctrlrange tracks the joint range: the planner commands joint targets.
        return sub_attr(tag, "ctrlrange", limits[joint.group(1)][0])

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
    sys.stderr.write("_gen_h12_base_limits: imported %d joint limits from CL_Assets\n"
                     % len(limits))


if __name__ == "__main__":
    main()
