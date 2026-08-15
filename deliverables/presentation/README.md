# Executive deck — The Ecosystem Engine

**`ecosystem-engine-deck.html`** is the deck. **`ecosystem-engine-deck.pdf`** is the same thing
exported for submission — **exactly 10 pages, one slide per page**, which is the brief's hard limit
(max 10 slides *including* title and thank-you).

## Presenting it

Open the HTML in any browser and press **F** (or the browser's full-screen key) — it is
self-contained, so it works from a USB stick, from `file://`, with no server and no internet.

| Key | Does |
|---|---|
| `→` · `Space` · `PageDown` | next slide |
| `←` · `PageUp` | previous slide |
| `Home` / `End` | first / last slide |

The slide counter and progress rail sit at the bottom; `#s7` in the URL deep-links to slide 7.

## The ten slides

| # | Title states the conclusion | Earns |
|---|---|---|
| 1 | Next week, $604,163 of rake walks out — and we have the names | the one number |
| 2 | The house never wins a hand. It rents the seat | business problem, before any technology |
| 3 | 116,619,267 seat-rows in. 18 MB out | data engineering & architecture, one diagram |
| 4 | We lose the players we earn from, and keep the ones we don't | unsupervised model + the ecosystem thesis |
| 5 | Going quiet is predictable a week ahead | supervised model + feature engineering |
| 6 | A four-minute spreadsheet sort scores 0.800 | **the failure slide** — baselines, threshold, limits |
| 7 | The model ranks players. It does not price them | model evaluation — calibration + the rake estimator |
| 8 | A retention list sorted by churn risk is the wrong list | business recommendation |
| 9 | Four decisions, each with a number and an owner | recommendations with owners + the dashboard |
| 10 | One number, and the five things we will not hide behind it | thank you + caveats |

## Before submitting

- **Add the team names.** Two amber `TEAM: add names & roll numbers` chips (slides 1 and 10) mark
  where. They are deliberately impossible to miss.
- Re-export the PDF after any edit (below).

## Nothing on these slides is typed by hand

Every figure sits in a span tagged with the JSON path it came from, and every chart sits between
`<!--CHART:name:START-->` markers. Rebuild them from the measured results with:

```bash
.venv/bin/python src/deck_charts.py           # rewrite figures + charts, print what changed
.venv/bin/python src/deck_charts.py --check   # verify only; exits 1 if the deck has drifted
```

It reads `docs/rake-at-risk-results.json`, `docs/fork-a-results.json`,
`docs/segments-results.json` and the shipped `data/gold/rake_at_risk` table, re-derives the three
ranking curves itself, and **refuses to write unless its own numbers reconcile against the model
run's JSON**. A run that changes nothing is the proof the deck still matches the analysis.

Re-export the PDF (needs Chrome; the `@page` rule in the deck does the 16:9 paging):

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="deliverables/presentation/ecosystem-engine-deck.pdf" \
  --virtual-time-budget=6000 \
  "file://$PWD/deliverables/presentation/ecosystem-engine-deck.html"
```

Then check it is still ten pages, because eleven is a rubric violation:

```bash
python3 -c "d=open('deliverables/presentation/ecosystem-engine-deck.pdf','rb').read(); \
print(d.count(b'/Type /Page') - d.count(b'/Type /Pages'))"
```
