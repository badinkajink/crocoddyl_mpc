#!/usr/bin/env python3
"""Render the static-equilibrium CoM regions from regions.json as inline SVG.

Inline rather than a PNG so the docpage stays a single self-contained file and
the plot restyles itself in dark mode with the rest of the page (strokes use
currentColor-adjacent CSS variables, not baked-in hex).

Top view, world x to the RIGHT (away from the robot, toward the table) and world
y UP the page. Drawn per contact set: the actuation-limited region, the pose's
CoM, the contact points, and -- for reference in every panel -- the legs-only
region, so the growth is visible rather than asserted.

2026-08-04 (S11): contacts are COLOURED BY IDENTITY and sized by normal load,
per the user's request. The S10 plot drew every contact as the same grey dot, so
it could show that a region grew without showing which contact grew it -- and
with the elbow and forearm sites only 57 mm apart, the two dots that mattered
most were also the two that were hardest to tell apart. The table outline is
drawn too, since "is the arm even over the table" turned out to be a live
question (measured: at the centred stance a placed site sits 5 mm OUTSIDE the
table's side edge).
"""
import json
import sys

import numpy as np

W, H = 300, 250
PAD = 26
ORDER = ["legs", "palm", "elbow+forearm", "elbow+forearm+palm", "elbow+forearm+hip"]
LABEL = {"legs": "legs only", "palm": "palm",
         "elbow+forearm": "elbow + forearm",
         "elbow+forearm+palm": "elbow + forearm + palm",
         "elbow+forearm+hip": "elbow + forearm + hip"}
COLOR = {"legs": "var(--ink3)", "palm": "var(--s3)",
         "elbow+forearm": "var(--s1)", "elbow+forearm+palm": "var(--s2)",
         "elbow+forearm+hip": "var(--s5)"}
# per-CONTACT colours; deliberately not the same ramp as the region colours so a
# dot is never confused with the region it sits in
CCOLOR = {"foot": "var(--ink3)", "elbow": "var(--s1)", "forearm": "var(--s4)",
          "palm": "var(--s3)", "hip": "var(--s5)", "torso": "var(--s2)"}
CMARK = {"foot": "square", "elbow": "circle", "forearm": "circle",
         "palm": "circle", "hip": "diamond", "torso": "diamond"}


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "regions.json"
    R = json.load(open(src))
    meta = R.pop("_meta", {})
    order = [k for k in ORDER if k in R]

    allpts = []
    for k in order:
        allpts += R[k]["actuated"] + [R[k]["com"]] + [p["p"][:2] for p in R[k]["pts"]]
    A = np.array(allpts)
    x0, x1 = A[:, 0].min(), A[:, 0].max()
    y0, y1 = A[:, 1].min(), A[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.12          # square the aspect
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    sx = lambda x: PAD + (x - x0) / (x1 - x0) * (W - 2 * PAD)
    sy = lambda y: H - PAD - (y - y0) / (y1 - y0) * (H - 2 * PAD)

    def poly(pts, **kw):
        if not pts:
            return ""
        s = " ".join("%.1f,%.1f" % (sx(p[0]), sy(p[1])) for p in pts)
        att = " ".join('%s="%s"' % (k.replace("_", "-"), v) for k, v in kw.items())
        return '<polygon points="%s" %s/>' % (s, att)

    def mark(kind, X, Y, r, fill):
        if kind == "square":
            return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="%s" '
                    'opacity="0.75"/>' % (X - r, Y - r, 2 * r, 2 * r, fill))
        if kind == "diamond":
            return ('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" '
                    'fill="%s" opacity="0.9"/>'
                    % (X, Y - r * 1.3, X + r * 1.3, Y, X, Y + r * 1.3,
                       X - r * 1.3, Y, fill))
        return ('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" opacity="0.9" '
                'stroke="var(--bg)" stroke-width="0.8"/>' % (X, Y, r, fill))

    out = []
    seen_labels = []
    for k in order:
        r = R[k]
        g = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="equilibrium region, {LABEL[k]}">']
        # table outline, so "is the arm over the table" is answerable by looking
        tx, ty = r.get("table_x"), r.get("table_y")
        if tx and ty:
            g.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="none" stroke="var(--line)" stroke-width="1" '
                     'stroke-dasharray="2 2"/>'
                     % (sx(tx[0]), sy(ty[1]),
                        max(sx(tx[1]) - sx(tx[0]), 0), max(sy(ty[0]) - sy(ty[1]), 0)))
        # 10 cm scale bar
        bx, by = PAD, H - 10
        blen = 0.10 / (x1 - x0) * (W - 2 * PAD)
        g.append(f'<line x1="{bx}" y1="{by}" x2="{bx+blen:.1f}" y2="{by}" '
                 f'stroke="var(--ink3)" stroke-width="1.5"/>')
        g.append(f'<text x="{bx+blen+5:.1f}" y="{by+3.5}" font-size="9" '
                 f'fill="var(--ink3)">10 cm</text>')
        if k != "legs" and "legs" in R:
            g.append(poly(R["legs"]["actuated"], fill="none", stroke="var(--ink3)",
                          stroke_width="1", stroke_dasharray="3 3", opacity="0.75"))
        g.append(poly(r["actuated"], fill=COLOR[k], fill_opacity="0.16",
                      stroke=COLOR[k], stroke_width="1.8"))
        # contacts, coloured by identity and sized by normal load
        fmax = max([abs(p.get("fn", 0.0)) for p in r["pts"]] or [1.0])
        for p in r["pts"]:
            lab = p["label"]
            if lab not in seen_labels:
                seen_labels.append(lab)
            rad = 1.6 + 2.6 * np.sqrt(max(abs(p.get("fn", 0.0)), 0.0) / max(fmax, 1e-9))
            g.append(mark(CMARK.get(lab, "circle"), sx(p["p"][0]), sy(p["p"][1]),
                          rad, CCOLOR.get(lab, "var(--ink3)")))
        # CoM
        g.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="none" stroke="var(--ink)" '
                 'stroke-width="1.8"/>' % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('<circle cx="%.1f" cy="%.1f" r="1.6" fill="var(--ink)"/>'
                 % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('</svg>')

        slip = r.get("slip", {})
        worst = max([(v, kk) for kk, v in slip.items() if kk != "feet_worst"] or
                    [(0.0, "")])
        extra = ("" if not worst[1] else
                 ' Worst brace slip <b style="color:var(--ink)">%.2f</b> (%s).'
                 % (worst[0], worst[1]))
        out.append(
            '<figure><div class="plot">%s</div>'
            '<figcaption><b style="color:var(--ink)">%s</b> — margin '
            '<b style="color:var(--ink)">%.0f mm</b> actuated, %.0f mm contact-only.%s'
            ' Dashed polygon = legs-only region; dashed rectangle = the table.</figcaption></figure>'
            % ("".join(g), LABEL[k], 1000 * r["margin_a"], 1000 * r["margin_c"], extra))

    # legend, in the order the contacts were first drawn
    keyparts = []
    for lab in seen_labels:
        shape = CMARK.get(lab, "circle")
        css = ("border-radius:50%" if shape == "circle" else
               ("transform:rotate(45deg)" if shape == "diamond" else ""))
        keyparts.append('<span><i class="dot" style="background:%s;%s"></i>%s</span>'
                        % (CCOLOR.get(lab, "var(--ink3)"), css, lab))
    legend = ('<div class="key">%s<span><i class="dot" style="background:none;'
              'border:2px solid var(--ink)"></i>CoM</span>'
              '<span style="color:var(--ink3)">dot area &prop; normal force</span></div>'
              % "".join(keyparts))

    dst = sys.argv[2] if len(sys.argv) > 2 else "region_svg.html"
    open(dst, "w").write(legend + '\n<div class="grid2">\n' +
                         "\n".join(out) + "\n</div>")
    print("wrote", dst, "from", src, meta)


if __name__ == "__main__":
    main()
