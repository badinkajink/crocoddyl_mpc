#!/usr/bin/env python3
"""Find every media file the docpages reference, and put it where they look.

WHY THIS EXISTS. The docpages reference `media/<name>` relative to
docs/lean/. That directory did not survive the move out of the pre-rename
checkout and the studies/ reorg, so EVERY video in EVERY page was a broken
element -- silently, because a missing <video> renders as blank space rather
than as an error, and the pages otherwise look finished.

It is a resolver rather than a copy command on purpose: the media has lived in
at least four places (docs/lean/media, docs/lean/media_old, the old checkout,
and each session's runs/<date>/media), and the next reorg will move it again.
Given a basename it searches all of them, newest mtime wins, and it says what it
could not find instead of leaving a blank rectangle.

usage:
    fix_media.py            # report what is referenced, present and missing
    fix_media.py --copy     # actually populate docs/lean/media/
    fix_media.py --link     # symlink instead (no duplication; breaks if the
                            # source run directory is pruned)
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DEST = os.path.join(HERE, "media")

# Every place media has lived, newest-first is decided by mtime not by order.
SEARCH = [
    os.path.join(HERE, "media"),
    os.path.join(HERE, "media_old"),
    os.path.join(REPO, "studies", "runs"),
    os.path.join(REPO, "lean_analysis", "runs"),
    "/home/humanoid/Programs/mjpc_icra2026/docs/lean/media",
    "/home/humanoid/Programs/mjpc_icra2026/docs/lean/media_old",
    "/home/humanoid/Programs/mjpc_icra2026/lean_analysis/runs",
]
EXT = (".mp4", ".webm", ".png", ".svg", ".gif", ".jpg")
REF = re.compile(r'(?:src|href)="(media/[^"]+)"')


def referenced():
    """{basename: [pages that want it]} across every docpage."""
    out = {}
    for name in sorted(os.listdir(HERE)):
        if not name.endswith(".html"):
            continue
        text = open(os.path.join(HERE, name), encoding="utf-8",
                    errors="replace").read()
        for rel in REF.findall(text):
            out.setdefault(os.path.basename(rel), []).append(name)
    return out


def index():
    """{basename: newest path} over every search root."""
    found = {}
    for root in SEARCH:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for f in files:
                if not f.lower().endswith(EXT):
                    continue
                p = os.path.join(dirpath, f)
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                cur = found.get(f)
                if cur is None or mt > os.path.getmtime(cur):
                    found[f] = p
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--link", action="store_true")
    a = ap.parse_args()

    want, have = referenced(), index()
    present = set(os.listdir(DEST)) if os.path.isdir(DEST) else set()
    todo, missing = [], []
    for name in sorted(want):
        if name in present:
            continue
        (todo if name in have else missing).append(name)

    print("referenced: %d   already in media/: %d   resolvable: %d   MISSING: %d"
          % (len(want), len(want) - len(todo) - len(missing), len(todo),
             len(missing)))
    if a.copy or a.link:
        os.makedirs(DEST, exist_ok=True)
        for name in todo:
            dst = os.path.join(DEST, name)
            if a.link:
                os.symlink(have[name], dst)
            else:
                shutil.copy2(have[name], dst)
        print("%s %d file(s) into %s" % ("linked" if a.link else "copied",
                                         len(todo), DEST))
    elif todo:
        print("\nresolvable (re-run with --copy):")
        for name in todo[:8]:
            print("  %-46s <- %s" % (name, have[name].replace(REPO + "/", "")))
        if len(todo) > 8:
            print("  ... and %d more" % (len(todo) - 8))
    if missing:
        print("\nNOT FOUND ANYWHERE -- these render as blank space:")
        for name in missing:
            print("  %-46s wanted by %s" % (name, ", ".join(sorted(set(
                want[name])))))
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
