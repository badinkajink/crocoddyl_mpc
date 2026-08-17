#!/usr/bin/env python3
"""Repair hand-written MathML in the lean docpages.

Two passes, both confined to <math> elements.

PASS 1 -- inter-tag whitespace.
Symptom: equations render with stray line breaks and gaps after variables.
Cause: <math> in an HTML document is FOREIGN CONTENT. Everything inside it is
parsed in the MathML namespace, where a whitespace-only run between two tags is
not "source formatting" -- it is a text node. MathML Core renders text found
directly inside an <mrow> as an anonymous token box, so pretty-printing

    <mrow>
      <mi>a</mi>
      <mo>+</mo>

puts a real, rendered, line-breakable space after every single token. The
equation then wraps at each one. HTML's whitespace rules do not apply here,
which is why this looks like a CSS problem and is not one.
Fix: `>\\s+<` -> `><`. Whitespace inside a token element (<mtext>a b</mtext>) is
untouched because it is not between tags, and deliberate spacing already uses
<mspace>, which this cannot affect.

PASS 2 -- operator spacing and accents.
Symptom that survives pass 1: superscripts sit too far out and too high, dotted
derivatives float above the variable, |x| renders with gaps inside the bars.
Cause: an <mo> carries the operator dictionary's spacing wherever it appears,
INCLUDING inside a script position, and an operator with no dictionary entry
(such as U+22A4 transpose) falls back to infix spacing of 0.2777em a side. So
`<msup><mi>J</mi><mo>&#x22A4;</mo></msup>` typesets the tack with a quarter-em
of padding on both sides, inside a superscript that is already scaled down.
Likewise <mover> is a generic script constructor: without accent="true" the
overscript is positioned as a script, not tucked down as a diacritic, so q-dot
renders with the dot floating clear of the q.
Fix, structurally rather than by pattern -- the fragments are well-formed XML,
so this parses them and walks the tree:
  * script-position <mo> (msup/msub/msubsup/munder/mover) gets zero lspace and
    rspace;
  * <mover> with a diacritic overscript gets accent="true", and the diacritic
    gets stretchy="false";
  * a large operator carrying a script (<msub><mo>&#x2211;</mo>...) becomes
    <munder>/<mover>, matching the other sums on the page and letting
    movablelimits do the inline-vs-display placement;
  * fence glyphs used as delimiters (| and the double bar) get zero spacing and
    stretchy="false", so |x| closes up.

usage: fix_mathml.py FILE...   (rewrites in place, reports what it changed)
"""
import re
import sys
import xml.etree.ElementTree as ET

MATH = re.compile(r"<math\b[^>]*>.*?</math>", re.S)
GAP = re.compile(r">\s+<")

# Combining-free diacritics used as <mover> overscripts.
ACCENTS = {"˙", "¨", "¯", "ˆ", "˜", "̇", "̈",
           "⃗", "→", "^", "~"}
# Operators that take limits above/below rather than to the side.
BIG_OPS = {"∑", "∏", "∫", "∬", "∭", "⋃", "⋂",
           "⨀", "⨁", "⨂"}
# Delimiters that should hug their content.
FENCES = {"|", "‖", "∣", "∥", "⎥"}
SCRIPT_PARENTS = {"msup", "msub", "msubsup", "munder", "mover", "munderover"}


def _zero_space(el, counts, key):
    if el.get("lspace") == "0em" and el.get("rspace") == "0em":
        return
    el.set("lspace", "0em")
    el.set("rspace", "0em")
    counts[key] = counts.get(key, 0) + 1


def _walk(el, counts):
    kids = list(el)
    tag = el.tag

    # A large operator carrying a sub/superscript wants under/over placement.
    if tag in ("msub", "msup", "msubsup") and kids and kids[0].tag == "mo" \
            and (kids[0].text or "").strip() in BIG_OPS:
        el.tag = {"msub": "munder", "msup": "mover",
                  "msubsup": "munderover"}[tag]
        counts["big operator moved to under/over"] = \
            counts.get("big operator moved to under/over", 0) + 1
        tag = el.tag

    # Diacritic overscript: tuck it onto the base instead of scripting it.
    if tag == "mover" and len(kids) == 2 and kids[1].tag == "mo" \
            and (kids[1].text or "").strip() in ACCENTS:
        if el.get("accent") != "true":
            el.set("accent", "true")
            kids[1].set("stretchy", "false")
            counts["accent marked on <mover>"] = \
                counts.get("accent marked on <mover>", 0) + 1

    # Any <mo> sitting in a script slot keeps its dictionary spacing; kill it.
    if tag in SCRIPT_PARENTS:
        for child in kids[1:]:
            if child.tag == "mo":
                _zero_space(child, counts, "script-position <mo> de-spaced")

    # Delimiters used as absolute-value / norm bars.
    if tag == "mo" and (el.text or "").strip() in FENCES:
        _zero_space(el, counts, "fence bar de-spaced")
        if el.get("stretchy") != "false":
            el.set("stretchy", "false")

    for k in kids:
        _walk(k, counts)


def fix(html):
    counts = {}

    def one(mo):
        body = GAP.sub("><", mo.group(0))
        n = len(GAP.findall(mo.group(0)))
        if n:
            counts["stray text node removed"] = \
                counts.get("stray text node removed", 0) + n
        root = ET.fromstring(body)
        _walk(root, counts)
        out = ET.tostring(root, encoding="unicode", short_empty_elements=True)
        # ElementTree writes `<mspace width="1em" />`; close it up so re-running
        # the whitespace pass over the result is a no-op.
        return out.replace(" />", "/>")

    return MATH.sub(one, html), counts


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for path in sys.argv[1:]:
        out, counts = fix(open(path).read())
        if counts:
            open(path, "w").write(out)
        print(path)
        for k in sorted(counts):
            print("    %4d  %s" % (counts[k], k))
        if not counts:
            print("    nothing to do")


if __name__ == "__main__":
    main()
