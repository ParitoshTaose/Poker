# Next session — the task, in full

**Hand this file to a new session.** Everything it needs to start is either here or named here.

---

## Read these first (in this order)

1. `Project/CLAUDE.md` — especially the three measured sections **FORK A**, **K-MEANS
   SEGMENTATION** and **RAKE AT RISK**, and the whole *Data & code gotchas* block. The gotchas
   cost days to find; do not rediscover them.
2. `docs/rake-at-risk-results.md` — the headline the dashboard, the deck and the report all open
   on, plus the four sensitivities behind it.
3. `docs/fork-a-results.md` — the lapse model, its baselines, and its known defects.
4. `docs/segments-results.md` — the segments. (Its closing calibration warning is now **closed**;
   see §1 of the rake write-up.)

## Where the project stands

**The analysis is complete, the dashboard is built, and the deck is built.** Bronze → Silver →
Gold, then Fork A (short-horizon lapse classification, three MLlib algorithms against three
baselines), K-Means segmentation, and the money-weighted "rake at risk" headline. All three
Phase-3 model types exist and are measured, every model output has a dollar figure attached,
**`docs/dashboard.html` (15 Aug)** closes Gate A bar A7 — which needs a teammate rather than more
code — and **`deliverables/presentation/ecosystem-engine-deck.html` (16 Aug)** closes Gate B.

The topic is **cleared with the professor** — that blocker is closed.

What does *not* exist yet: the **report PDF** and the **notebook**. `notebooks/` and
`deliverables/report/` are empty. Submission is **8 Sept 2026**; presentations 11–12 Sept.
Everything remaining is writing and packaging.

**Databricks is deliberately deferred to later in the project** (user's decision, 14 Aug). Note the
notebook itself does not have to wait for it — the logic is already written as local PySpark, so
the notebook is a re-hosting job whenever it happens.

## The line everything else opens on

> **$604,163 of next week's rake sits at risk across 77,268 players on five venues — 30% of it in
> the recreational segment — and contacting the top 10% by expected loss (7,727 players) puts
> $421,064 of it in reach.**

Quote it with its small print attached: weekly rake $2,114,626 of which **34.4% is measured** and
the rest estimated; five venues (iPoker excluded, it reconciles on none of its 6.0 M hands);
**2009 US dollars**, no conversion, no inflation. And the finding that earns the recommendation
marks: **a retention list sorted by churn risk shares 0.2% of its names with a list sorted by
expected loss, and reaches 0.9% of the money.**

---

## DONE — the dashboard (15 Aug)

`docs/dashboard.html`, built from `src/dashboard_data.py`. **Do not hand-edit its numbers**;
re-run the script, which rewrites the block between the `/*DATA:START*/` markers and refuses to
write unless the payload reconciles against Gold and the run log.

Five surfaces: the headline band (the sentence, number large, filter-aware) · a sticky
venue/segment/stake filter bar · the contact-budget panel with the three ranking curves · the
segment × venue money matrix you click to drill · the call list with CSV export · the trust strip
(measured-vs-modelled coverage, the reliability curve, model-vs-recency-baseline, each venue's own
clock, the iPoker exclusion, the four sensitivities).

Gate A: **A1–A6 and A8–A10 are met. A7 is the one left, and it needs a person** — hand the file to
a teammate on their own machine, say nothing, and write down their first sentence. If it is a
question about a chart rather than a statement about the business, fix the dashboard.

### The one thing to re-run after any rebuild

Held-out fold, 10% budget: **1,936 contacts · $107,598 reached (70%) · 1,013 true lapsers ·
$104,523 realised · risk-only finds 1,876 lapsers but reaches 0.9% of the money · 0.2% overlap.**
Every figure matches `rake-at-risk-results.md` §8. If those move, something broke.

---

## DONE — the deck (16 Aug)

`deliverables/presentation/ecosystem-engine-deck.html`, exported to `.pdf` at exactly ten pages,
with `deliverables/presentation/README.md` covering how to present it and how to re-export.
Gate B: **B1–B8 all met.** Ten slides including title and thank-you; the business problem lands
before any technology; the architecture is one diagram carrying *116,619,267 seat-rows in, 18 MB
out*; the failure has its own slide with the baseline comparison on it; the closing slide is four
decisions each with a number and an owner; every title states its conclusion; the dashboard appears
as a captioned screenshot.

**Do not hand-edit its numbers.** `src/deck_charts.py` re-derives the three ranking curves from
`gold/rake_at_risk`, checks them against `rake-at-risk-results.json`, refuses to write if they
disagree, and rewrites all 46 tagged figures plus five chart/table blocks in place.
`--check` verifies without writing and exits 1 on drift — run it before submitting.

**Two things left on the deck itself, both needing a person, not code:**

1. **Team names.** Two amber `TEAM: add names & roll numbers` chips (slides 1 and 10) mark where.
2. **Gate A7 / a first-reader test.** Hand the deck (or the dashboard) to a teammate, say nothing,
   and write down their first sentence. A question about a chart means the slide needs fixing; a
   statement about the business means it works.

---

## THE TASK — the report PDF

The consulting report, into `deliverables/report/`. Named sections from the brief: Exec Summary ·
Business Context · Data Understanding (**the 5 Vs**) · Enterprise Architecture · Data Engineering ·
ML · Business Insights · Strategic Recommendations · **Appendix**. `docs/delivery-gate.html`
**Gate D** is the spec.

- **The appendix has three named requirements that are easy to forget and cheap to lose marks on:
  references, an AI-usage disclosure, and team contributions.**
- Everything is already written down somewhere — `docs/rake-at-risk-results.md`,
  `fork-a-results.md`, `segments-results.md` and `bakeoff-decision.md` between them carry every
  number, every caveat and most of the prose. The report is assembly and narrative, not new work.
- The two paragraphs that show the analysis was *understood* rather than executed: **who paid the
  rake versus whose money funded it** (rake write-up §6), and **why the population filter was a
  selection leak** (Fork A §1).
- Reuse the deck's structure for the spine, then go deeper — the report is where the rejected
  options belong (the bake-off's five candidate questions, why Fork A won, why EXTENDED lost).

## Traps already paid for — do not rediscover these

- **`os.environ.setdefault("PYSPARK_PYTHON", sys.executable)` before importing pyspark**, or Spark
  runs its workers under Apple's Python 3.9 and dies inside a Java stack trace.
- **MLlib's `Imputer` only accepts double/float.** Cast numeric features up front.
- **Never `.collect()` or `.toPandas()` on anything but Gold grain.** The driver is your 16 GB.
- **`spark.local.dir` must stay on the project disk** (`data/_work/spark-scratch`), never `/tmp`.
- **GBT is not bit-reproducible** even with a fixed seed. Quote 3 decimals; don't chase the wobble.
- **bb/100 has a wild tail** — a small-sample artifact from thin money coverage. Filter on
  `money_hands`, never `hands_played`.
- **Currency is USD** ($0.25 big blinds). Converting to ₹ would mean inventing a 2009 exchange
  rate — keep dollars and say why.
- The uncalled-bet correction in `parse_phh.py` is load-bearing and already correct. Leave it alone.
- **`gold/rake_at_risk` stores `risk_cal` rounded to 4 dp, but `expected_rake_at_risk` was computed
  from the unrounded probability.** Never recompute `risk × rake` yourself — read the column. Same
  rounding means "players at risk" re-derived from the table is **58,837**, one more than the
  58,836 in the run log; both are stated in `rake-at-risk-results.md` §11.
- **Printing HTML to PDF: Chrome lays a page out at 3/4 of its declared pixel size.** The deck
  uses `@page{size:1707px 960px}` to get a 1280x720 layout box and ten exact pages. Declare the
  page in mm and the layout drops under the responsive breakpoint, every grid collapses to one
  column, and content is clipped. Also set `print-color-adjust:exact`, or tinted panels print white,
  and scope stacking media queries to `@media screen`.
- **`player_id` is unique only within a venue.** Any set/join on it alone silently de-duplicates
  across sites — it made the deck's first overlap figure 99.8% where it had to be 100%. Use
  `site + ":" + player_id`.
- **`gold/rake_at_risk` has no stake column.** The stake filter uses each player's *modal* stake,
  computed over their venue's own prior window and cached at `data/_work/player_stake.parquet`.
  `p_max_stake` is the wrong join — it differs for a quarter of players.

---

## After this task, in order

1. **The notebook** — Gate C. "Fully executable" is the brief's word: it must ship **with outputs
   saved**, run top-to-bottom on a fresh kernel, and print both correctness tests (the zero-sum
   identity and big-blind VPIP ≈ 31.3%) plus the 21,605,687 reconciliation. Then Databricks when
   the team gets to it.

Two small things that are pure marks and take minutes: the **team table in `README.md` is still a
`TODO`**, and Gate 0.2 — **which Databricks tier** the course uses — is still unanswered. The
README is also drifting: it still describes `src/` as "no Spark needed" and lists only
`parse_phh.py`, and it says the window is 23 days (it is 26, per venue). Worth ten minutes before
submission, since Professionalism & Documentation is 2 marks.

## Not run, and worth knowing about

- **A time-shifted backtest** (train at `site_last_day − 14`, test at `− 7`). Fork A is split by
  player, never by period, so temporal generalisation is not claimed anywhere.
- **A "funded-by" rake attribution** weighted by net losses rather than contributions. Contributed
  rake says who *paid*; the ecosystem thesis is about whose money *funded* it. Both belong in the
  story and only the first is measured.
- **`features.py` still writes `+inf` into `avg_stack_bb`** for all iPoker players. The guard
  exists in `features_lapse.py` (`finite()`); porting it back and rebuilding Gold has not been done.
