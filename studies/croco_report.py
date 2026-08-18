#!/usr/bin/env python3
"""Assemble a lean docpage: series stylesheet + body fragment + injected figures.

The pages in docs/lean/ share one hand-written stylesheet and inline every figure
as SVG so both themes work.  Copying that stylesheet by hand into each new page is
how a series drifts, so this lifts it from an existing page instead, wraps the
body fragment, and substitutes the <!--FIG:name--> placeholders from
croco_figs.py's JSON.

usage: simple_report.py --body _body_2026-08-06.html --out 2026-08-06_croco_loop.html
"""

import argparse
import json
import os
import re

DOCS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "lean")


def style_from(path):
    s = open(path).read()
    i, j = s.index("<style>"), s.index("</style>") + len("</style>")
    return s[i:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--style-from", default="2026-08-05_crocoddyl.html")
    ap.add_argument("--figs", default="media/figs_2026-08-06.json")
    args = ap.parse_args()

    body = open(os.path.join(DOCS, args.body)).read()
    figs_path = os.path.join(DOCS, args.figs)
    figs = json.load(open(figs_path)) if os.path.exists(figs_path) else {}
    for name, svg in figs.items():
        body = body.replace(f"<!--FIG:{name}-->", svg)
    left = re.findall(r"<!--FIG:([a-z_0-9]+)-->", body)
    if left:
        print(f"  WARNING: no figure for {sorted(set(left))}")

    html = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            f'<title>{args.title}</title>\n'
            + style_from(os.path.join(DOCS, args.style_from))
            + '\n</head>\n<body><div class="wrap">\n\n'
            + body
            + '\n</div></body>\n</html>\n')
    dst = os.path.join(DOCS, args.out)
    with open(dst, "w") as fh:
        fh.write(html)
    print(f"wrote {dst}  ({len(html)} bytes, {len(figs)} figures)")


if __name__ == "__main__":
    main()
