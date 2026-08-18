#!/usr/bin/env python3
"""Assemble the Lean Simple docpage: shared stylesheet + body + generated parts.

The docs/lean/ pages share one hand-written stylesheet and inline every figure as
SVG so both themes work and each page stays a single file. Copying the stylesheet
by hand into a new page is how a series drifts, so this lifts it from an existing
page, wraps the body fragment, and substitutes the <!--NAME--> placeholders with
the files simple_metrics.py / simple_region_svg.py wrote. Every number and every
figure in the page therefore comes from the run directory, not from a transcript.

usage: simple_page.py --body _body_simple_2026-08-06.html \
                      --run ../studies/runs/2026-08-06_simple/matrix \
                      --out 2026-08-06_lean_simple.html --title "..."
"""
import argparse
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "..", "docs", "lean")

# placeholder -> file in the run directory
PARTS = {
    "TABLE_RESULTS": "table_results.html",
    "FIG_REACH": "fig_reach.svg",
    "FIG_GAPS": "fig_gaps.svg",
    "FIG_MODES": "fig_modes.svg",
    "REGION_PANELS": "region_panels.html",
}


def style_from(path):
    s = open(path).read()
    i, j = s.index("<style>"), s.index("</style>") + len("</style>")
    return s[i:j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--style-from", default="2026-08-05_mjpc_chain.html")
    a = ap.parse_args()

    body = open(os.path.join(DOCS, a.body)).read()
    for name, fn in PARTS.items():
        p = os.path.join(a.run, fn)
        if not os.path.exists(p):
            print("  WARNING: missing %s for <!--%s-->" % (p, name))
            continue
        body = body.replace("<!--%s-->" % name, open(p).read())
    left = re.findall(r"<!--([A-Z_0-9]+)-->", body)
    if left:
        print("  WARNING: unfilled placeholders", sorted(set(left)))

    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            '<title>%s</title>\n' % a.title
            + style_from(os.path.join(DOCS, a.style_from))
            # the region panels reuse the handoff page's `.dot` legend swatch,
            # which lives in that page's stylesheet rather than the shared one
            + '\n<style>.key .dot{width:10px;height:10px;display:inline-block;'
              'flex:0 0 auto}</style>\n'
            + '</head>\n<body><div class="wrap">\n\n' + body
            + '\n</div></body>\n</html>\n')
    dst = os.path.join(DOCS, a.out)
    open(dst, "w").write(html)
    print("wrote %s (%d bytes)" % (dst, len(html)))


if __name__ == "__main__":
    main()
