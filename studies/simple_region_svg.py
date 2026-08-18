#!/usr/bin/env python3
"""Render regions_simple.json as inline SVG panels, one per rollout.

Same picture as docs/lean/2026-08-04_mjpc_handoff.html §5 -- top view, world x to
the right (toward the table), world y up the page, the actuation-limited region
filled, contacts coloured by identity and sized by the min-effort normal force,
the CoM as a ring -- with one difference that is the whole point: the dashed
reference polygon is the LEGS-ONLY region AT THE SAME POSE, not at a different
solve. So each panel reads "here is where this robot's CoM could be if the arm
carried nothing, and here is where it can be because the arm does".

Inline SVG rather than a PNG so the page stays one file and restyles in dark
mode with everything else.

usage: simple_region_svg.py regions_simple.json out.html [cell,cell,...]
"""
import json
import sys

import numpy as np

W, H = 300, 250
PAD = 26
CCOLOR = {"foot": "var(--ink3)", "elbow": "var(--s1)", "forearm": "var(--s4)",
          "palm": "var(--s3)", "hip": "var(--s5)", "torso": "var(--s2)"}
CMARK = {"foot": "square", "elbow": "circle", "forearm": "circle",
         "palm": "circle", "hip": "diamond", "torso": "diamond"}
LABEL = {
    "near_none_s0": "NEAR, no mode requested",
    "near_brace_s0": "NEAR, {elbow, forearm}",
    "far_none_s0": "FAR, no mode requested",
    "far_brace_s0": "FAR, {elbow, forearm}",
    "far_palm_s0": "FAR, {palm}",
    "far_all_s0": "FAR, {elbow, forearm, palm}",
    "vfar_none_s0": "VFAR, no mode requested",
    "vfar_brace_s0": "VFAR, {elbow, forearm}",
}


def main():
    src = sys.argv[1]
    dst = sys.argv[2]
    R = json.load(open(src))
    cells = (sys.argv[3].split(",") if len(sys.argv) > 3
             else [k for k in R if not k.endswith("__legs")])
    cells = [c for c in cells if c in R]

    allpts = []
    for c in cells:
        allpts += R[c]["actuated"] + [R[c]["com"]] + \
            [p["p"][:2] for p in R[c]["pts"]] + \
            R.get(c + "__legs", {}).get("actuated", [])
    A = np.array(allpts)
    x0, x1 = A[:, 0].min(), A[:, 0].max()
    y0, y1 = A[:, 1].min(), A[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) / 2 * 1.10
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
            return ('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                    'fill="%s" opacity="0.75"/>' % (X - r, Y - r, 2 * r, 2 * r, fill))
        if kind == "diamond":
            return ('<polygon points="%.1f,%.1f %.1f,%.1f %.1f,%.1f %.1f,%.1f" '
                    'fill="%s" opacity="0.9"/>'
                    % (X, Y - r * 1.3, X + r * 1.3, Y, X, Y + r * 1.3, X - r * 1.3, Y,
                       fill))
        return ('<circle cx="%.1f" cy="%.1f" r="%.1f" fill="%s" opacity="0.9" '
                'stroke="var(--bg)" stroke-width="0.8"/>' % (X, Y, r, fill))

    out, seen = [], []
    for c in cells:
        r = R[c]
        legs = R.get(c + "__legs")
        braced = bool(r["subset"])
        col = "var(--s1)" if braced else "var(--s4)"
        g = ['<svg viewBox="0 0 %d %d" role="img" aria-label="equilibrium '
             'region for %s">' % (W, H, LABEL.get(c, c))]
        tx, ty = r.get("table_x"), r.get("table_y")
        if tx and ty:
            g.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" '
                     'fill="none" stroke="var(--line)" stroke-width="1" '
                     'stroke-dasharray="2 2"/>'
                     % (sx(tx[0]), sy(ty[1]), max(sx(tx[1]) - sx(tx[0]), 0),
                        max(sy(ty[0]) - sy(ty[1]), 0)))
        bx, by = PAD, H - 10
        blen = 0.10 / (x1 - x0) * (W - 2 * PAD)
        g.append('<line x1="%d" y1="%d" x2="%.1f" y2="%d" stroke="var(--ink3)" '
                 'stroke-width="1.5"/>' % (bx, by, bx + blen, by))
        g.append('<text x="%.1f" y="%.1f" font-size="9" fill="var(--ink3)">'
                 '10 cm</text>' % (bx + blen + 5, by + 3.5))
        if legs:
            g.append(poly(legs["actuated"], fill="none", stroke="var(--ink3)",
                          stroke_width="1", stroke_dasharray="3 3", opacity="0.8"))
        g.append(poly(r["actuated"], fill=col, fill_opacity="0.16", stroke=col,
                      stroke_width="1.8"))
        fmax = max([abs(p.get("fn", 0.0)) for p in r["pts"]] or [1.0])
        for p in r["pts"]:
            if p["label"] not in seen:
                seen.append(p["label"])
            rad = 1.6 + 2.6 * np.sqrt(max(abs(p.get("fn", 0.0)), 0.0)
                                      / max(fmax, 1e-9))
            g.append(mark(CMARK.get(p["label"], "circle"), sx(p["p"][0]),
                          sy(p["p"][1]), rad, CCOLOR.get(p["label"], "var(--ink3)")))
        g.append('<circle cx="%.1f" cy="%.1f" r="4.5" fill="none" '
                 'stroke="var(--ink)" stroke-width="1.8"/>'
                 % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('<circle cx="%.1f" cy="%.1f" r="1.6" fill="var(--ink)"/>'
                 % (sx(r["com"][0]), sy(r["com"][1])))
        g.append('</svg>')

        ma, ml = 1000 * r["margin_a"], 1000 * (legs or r)["margin_a"]
        cls = "good" if ma > 0 else "bad"
        out.append(
            '<figure><div class="plot">%s</div>'
            '<figcaption><b style="color:var(--ink)">%s</b> — contacts '
            '<b style="color:var(--ink)">%s</b>. Margin '
            '<b style="color:var(--%s)">%+.0f mm</b> actuated, %+.0f mm '
            'contact-only; legs alone at this same pose: '
            '<b style="color:var(--%s)">%+.0f mm</b>. Min-effort peak torque '
            '%.2f&times;. Dashed polygon = legs-only region; dashed rectangle = '
            'the table.</figcaption></figure>'
            % ("".join(g), LABEL.get(c, c), "+".join(r["subset"]) or "none",
               cls, ma, 1000 * r["margin_c"],
               "good" if ml > 0 else "bad", ml, r["peak"]))

    keyparts = []
    for lab in seen:
        shape = CMARK.get(lab, "circle")
        css = ("border-radius:50%" if shape == "circle" else
               ("transform:rotate(45deg)" if shape == "diamond" else ""))
        keyparts.append('<span><i class="dot" style="background:%s;%s"></i>%s</span>'
                        % (CCOLOR.get(lab, "var(--ink3)"), css, lab))
    legend = ('<div class="key">%s<span><i class="dot" style="background:none;'
              'border:2px solid var(--ink)"></i>CoM</span>'
              '<span style="color:var(--ink3)">dot area &prop; normal force</span>'
              '</div>' % "".join(keyparts))
    open(dst, "w").write(legend + '\n<div class="grid2">\n' + "\n".join(out) +
                         "\n</div>\n")
    print("wrote", dst, "with", len(cells), "panels")


if __name__ == "__main__":
    main()
