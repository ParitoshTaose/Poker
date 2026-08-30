# Consulting report — The Ecosystem Engine

**`ecosystem-engine-report.html`** is the report. **`ecosystem-engine-report.pdf`** is the same
file printed for submission — **exactly 35 A4 pages, one page per sheet**.

The brief names nine sections; the report uses those words as its headings so an examiner can find
each one without hunting:

| § | Section | Carries |
|---|---|---|
| 1 | Executive Summary | Written to be the only page anyone reads |
| 2 | Business Context | The industry, the KPIs, the ethics, and the 2009 date raised by us |
| 3 | Data Understanding (5 Vs) | Each V answered with a measurement, not a definition |
| 4 | Enterprise Architecture | The medallion lake, the compute, and what breaks at 10× |
| 5 | Data Engineering | Parsing, defects, missing values, outliers, **feature engineering (5.6)**, rake derivation (5.7) |
| 6 | Machine Learning | The bake-off, three MLlib approaches, evaluation and final selection |
| 7 | Business Insights | Five findings, each with a number attached |
| 8 | Strategic Recommendations | Four costed decisions, risks, a 90-day roadmap |
| 9 | Appendix | **References · AI-usage disclosure · team contributions** + outcome map, reproducibility, glossary |

The three named appendix requirements — references, AI-usage disclosure and team contributions —
are Appendices A, B and C.

## Before submitting

- **Add the team names.** Three amber `TEAM:` chips mark where: the cover, Appendix C's table, and
  the closing block on the last page.
- Re-export the PDF after any edit (below), and re-run both checks.

## Nothing in this report is typed by hand

Every figure sits in a span tagged with the JSON path it came from —
`<b class="fig" data-fig="rake.headline.players" data-fmt="int">77,268</b>` — and every table and
chart sits between `<!--CHART:name:START-->` markers. **240 figures and 22 generated blocks.**

```bash
.venv/bin/python src/pipeline_facts.py      # re-measure the lake      -> docs/pipeline-facts.json
.venv/bin/python src/report_figures.py      # rewrite figures + blocks, print what changed
.venv/bin/python src/report_figures.py --check   # verify only; exit 1 if the report has drifted
```

`report_figures.py` reads `docs/pipeline-facts.json`, `rake-at-risk-results.json`,
`fork-a-results.json`, `segments-results.json`, `bakeoff-results.json` and the shipped
`data/gold/rake_at_risk` table, re-derives the three ranking curves itself, and **refuses to write
unless its own numbers reconcile against the model runs' JSON**. A run that changes nothing is the
proof the report still matches the analysis. It also stamps the page numbers from sheet order, so
splitting a page can never desynchronise them.

`pipeline_facts.py` is the same discipline for the engineering numbers: it recomputes the row
counts, sizes, per-venue splits and both correctness identities from `data/bronze|silver|gold` and
**aborts if the 21,605,687 reconciliation does not close exactly**.

## Every page must fit its sheet

Each page is an explicit 794×1123 px box with `overflow:hidden`, so content that does not fit is
**silently clipped in the PDF** rather than reflowing. That is the one failure mode this document
has, and it is invisible unless something measures it:

```bash
.venv/bin/python src/report_fit.py          # exit 1 if any page overflows or runs past the column
.venv/bin/python src/report_fit.py -v       # also list the pages that fit, with spare room
```

It renders the report in headless Chrome and measures the lowest and right-most content pixel on
every sheet, ignoring the absolutely-positioned footer. **Run it after any edit.**

## Re-export the PDF

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="deliverables/report/ecosystem-engine-report.pdf" \
  --virtual-time-budget=9000 \
  "file://$PWD/deliverables/report/ecosystem-engine-report.html"
```

Then check the page count, because a clipped page would not change it and a broken `@page` rule
would:

```bash
python3 -c "d=open('deliverables/report/ecosystem-engine-report.pdf','rb').read(); \
print(d.count(b'/Type /Page') - d.count(b'/Type /Pages'))"     # expect 35
```

## Layout rules that were paid for — do not regress them

- **`@page{size:A4;margin:0}` with each sheet an explicit `794×1123 px` box.** Declaring the page
  in `mm` *and* letting content flow gave Chrome control of the breaks and lost the per-page
  footers; this way one `.pg` is exactly one printed page.
- **Scope stacking media queries to `@media screen`.** The print layout box is 696 px wide, which
  is under the usual 900 px breakpoint — an unscoped query collapses every two-column grid.
- **Set `print-color-adjust:exact`**, or every tinted panel, chip and callout prints white.
- **`.chip` is `white-space:nowrap`** by design; a long one inside a table forces the table wider
  than the page. Use `class="chip wrap"` there — `report_fit.py` catches it either way.
- The cover's full-height flex wrapper carries `class="fill"` so the fit checker does not read its
  bottom edge as content overflow.
