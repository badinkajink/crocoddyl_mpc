#!/usr/bin/env python3
"""Clamp task keyframes into the joint ranges the BUILT model actually carries.

A maintenance tool, NOT a build step -- it rewrites source task XMLs in place, and
that has to be a reviewed diff rather than something the build does behind you.

Why it exists: the CL_Assets limit import (_gen_h12_base_limits.py) tightens joint
ranges under keyframes that were posed against the retired envelope, which strands
them outside their own limits. MuJoCo does not complain -- it silently clamps at
mj_resetDataKeyframe -- so the pose the planner starts from is quietly not the pose
the XML says. Two rounds of this have happened so far:

  2026-08-03  elbow floor -2.53 -> -0.95 stranded 21 jab_guard/jab_extend entries
  2026-08-04  wrist_pitch -0.471 -> -0.4625 stranded 3 of Allen's forearm_brace poses

Both were sub-degree overshoots of a limit the keyframe was posed AT, which is the
signature of a pose tuned against the old number rather than a real design intent.

Values are matched by JOINT NAME against each model's own qposadr, never by a fixed
index: the Hands variants put the same joint at a different qpos offset, so an index
keyed off the gripperless layout silently misses them.

Keyframes inside XML comments are skipped. The lean models carry several superseded
<key> blocks commented out with the SAME name as a live one, and rewriting those
would be both pointless and confusing in review.

usage: _clamp_keyframes.py <built model> <source xml> [--write]
       (without --write it reports and changes nothing)
"""
import os
import re
import sys

import mujoco

TOL = 1e-9


def comment_spans(src):
    return [(m.start(), m.end()) for m in re.finditer(r"<!--.*?-->", src, flags=re.S)]


def in_comment(pos, spans):
    return any(a <= pos < b for a, b in spans)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    write = "--write" in sys.argv
    if len(args) != 2:
        sys.exit(__doc__.strip().splitlines()[-2].strip())
    built, source = args

    # ABSOLUTE path required: the staged models reach their meshes through a
    # relative meshdir plus a symlink, and MuJoCo resolves that against the model
    # path as given -- hand it a relative one and it builds a nonsense mesh path.
    m = mujoco.MjModel.from_xml_path(os.path.abspath(built))
    # joint name -> (qpos index within a key's qpos vector, lo, hi)
    limits = {}
    for j in range(m.njnt):
        if not m.jnt_limited[j] or m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        limits[name] = (int(m.jnt_qposadr[j]), float(m.jnt_range[j][0]),
                        float(m.jnt_range[j][1]))
    by_adr = {adr: (n, lo, hi) for n, (adr, lo, hi) in limits.items()}

    src = open(source).read()
    spans = comment_spans(src)
    hits = []

    def fix_key(mo):
        if in_comment(mo.start(), spans):
            return mo.group(0)
        tag = mo.group(0)
        kname = re.search(r'name="([^"]+)"', tag)
        kname = kname.group(1) if kname else "?"
        qm = re.search(r'qpos="([^"]*)"', tag)
        if not qm:
            return tag
        vals = qm.group(1).split()
        if len(vals) != m.nq:
            sys.stderr.write("  skip key %-24s qpos has %d entries, model nq=%d\n"
                             % (kname, len(vals), m.nq))
            return tag
        changed = False
        for i, v in enumerate(vals):
            if i not in by_adr:
                continue
            jn, lo, hi = by_adr[i]
            x = float(v)
            if lo - TOL <= x <= hi + TOL:
                continue
            c = min(max(x, lo), hi)
            hits.append((kname, jn, x, c, lo, hi))
            # keep the source's own formatting style: plain decimal, trimmed
            vals[i] = ("%.6f" % c).rstrip("0").rstrip(".")
            changed = True
        if not changed:
            return tag
        return tag[:qm.start(1)] + " ".join(vals) + tag[qm.end(1):]

    out = re.sub(r"<key\b[^>]*?/>", fix_key, src, flags=re.S)

    if not hits:
        print("%s: 0 out-of-range keyframe entries" % source)
        return 0
    print("%s: %d out-of-range keyframe entries" % (source, len(hits)))
    for kname, jn, x, c, lo, hi in hits:
        print("  %-24s %-26s %+.6f -> %+.6f   (range %+.4f .. %+.4f, over by %.4f rad"
              " = %.2f deg)" % (kname, jn, x, c, lo, hi, abs(x - c),
                                abs(x - c) * 180 / 3.141592653589793))
    if write:
        open(source, "w").write(out)
        print("  written")
    else:
        print("  (dry run -- pass --write to apply)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
