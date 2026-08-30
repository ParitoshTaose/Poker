#!/usr/bin/env python3
"""
report_fit.py -- check that no page of the report overflows its sheet.

deliverables/report/ecosystem-engine-report.html lays every page out as an explicit
794x1123 px box with overflow:hidden, so content that does not fit is *silently
clipped* in the PDF rather than reflowing. That is the one failure mode this document
has, and it is invisible unless something measures it.

This renders the report in headless Chrome, measures the lowest content pixel on each
sheet (ignoring the absolutely-positioned footer), and reports every page that runs
past the safe limit.

Run:  .venv/bin/python src/report_fit.py          # report, exit 1 if anything overflows
      .venv/bin/python src/report_fit.py -v       # also list the pages that fit
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "deliverables" / "report" / "ecosystem-engine-report.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The sheet is 1123 px tall; the footer rule sits at bottom:6.5mm. Content must stop
# clear of it, so this is the last pixel a content element may occupy.
SAFE_BOTTOM = 1082

PROBE = """
<script>
(function(){
  var out=[];
  document.querySelectorAll('.pg').forEach(function(pg,i){
    var pr = pg.getBoundingClientRect(), maxb = 0, worst = '', maxr = 0, wide = '';
    var right = pg.clientWidth - parseFloat(getComputedStyle(pg).paddingRight);
    pg.querySelectorAll('*').forEach(function(el){
      if (el.closest('.ft') || el.classList.contains('fill')) return;   // positioned or full-height
      var r = el.getBoundingClientRect();
      var name = (el.className && el.className.baseVal === undefined ? el.className : el.tagName);
      if (r.height > 0 && r.bottom - pr.top > maxb) { maxb = r.bottom - pr.top; worst = name; }
      // SVG charts are allowed to bleed a little; flow content is not
      if (r.width > 0 && !el.closest('svg') && r.right - pr.left > maxr) { maxr = r.right - pr.left; wide = name; }
    });
    out.push([i+1, pg.id, Math.round(maxb), String(worst).slice(0,34),
              Math.round(maxr), Math.round(right + parseFloat(getComputedStyle(pg).paddingLeft)),
              String(wide).slice(0,34)].join('|'));
  });
  var d = document.createElement('pre');
  d.textContent = 'FITSTART\\n' + out.join('\\n') + '\\nFITEND';
  document.body.appendChild(d);
})();
</script>
</body>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="list pages that fit too")
    args = ap.parse_args()

    html = REPORT.read_text()
    if "</body>" not in html:
        sys.exit("report has no </body>")
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "fit.html"
        probe.write_text(html.replace("</body>", PROBE, 1))
        dom = subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--dump-dom",
             "--virtual-time-budget=6000", f"file://{probe}"],
            capture_output=True, text=True,
        ).stdout
    block = re.search(r"FITSTART\n(.*?)\nFITEND", dom, re.S)
    if not block:
        sys.exit("could not measure -- Chrome returned no probe output")

    over, fits, wide = [], [], []
    for row in block.group(1).splitlines():
        n, page_id, bottom, worst, right, limit, widest = row.split("|")
        record = (int(n), page_id, int(bottom), worst)
        (over if int(bottom) > SAFE_BOTTOM else fits).append(record)
        if int(right) > int(limit) + 1:
            wide.append((int(n), page_id, int(right), int(limit), widest))

    total = len(over) + len(fits)
    if args.verbose:
        for n, page_id, bottom, worst in fits:
            print(f"  [ok  ] p{n:<3} {page_id:<10} bottom {bottom:>5}   "
                  f"{SAFE_BOTTOM - bottom:>4} px spare")
    for n, page_id, bottom, worst in over:
        print(f"  [OVER] p{n:<3} {page_id:<10} bottom {bottom:>5}   "
              f"{bottom - SAFE_BOTTOM:>4} px CLIPPED   last: {worst}")

    for n, page_id, right, limit, widest in wide:
        print(f"  [WIDE] p{n:<3} {page_id:<10} right  {right:>5}   "
              f"{right - limit:>4} px past the column   last: {widest}")

    print(f"\n{total} pages · {len(fits)} fit · {len(over)} overflow · {len(wide)} too wide "
          f"(safe bottom {SAFE_BOTTOM} px)")
    return 1 if (over or wide) else 0


if __name__ == "__main__":
    raise SystemExit(main())
