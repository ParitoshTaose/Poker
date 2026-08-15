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

**The analysis is complete, and the dashboard is built.** Bronze → Silver → Gold, then Fork A
(short-horizon lapse classification, three MLlib algorithms against three baselines), K-Means
segmentation, and the money-weighted "rake at risk" headline. All three Phase-3 model types exist
and are measured, every model output has a dollar figure attached, and **`docs/dashboard.html`
(15 Aug) is the first graded artifact to exist** — Gate A, all seven blockers met bar A7, which
needs a teammate rather than more code.

The topic is **cleared with the professor** — that blocker is closed.

What does *not* exist yet: the **deck**, the **report PDF** and the **notebook**. `notebooks/` and
both `deliverables/` folders are empty. Submission is **8 Sept 2026**; presentations 11–12 Sept.
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

## THE TASK — the deck

**Max 10 slides including title and thank-you** (B1 — a named constraint, and the cheapest mark
anyone ever loses). That is eight working slides. `docs/delivery-gate.html` **Gate B** is the spec.

- **Slide 1 is the headline sentence**, with its small print attached: 34% measured, five venues,
  2009 USD.
- **The business problem lands before any technology** (B2). Rake, recreational churn, ecosystem
  collapse — in the operator's language. No Spark, no Parquet, no medallion until they care.
- **The architecture is one slide, one diagram** (B3), carrying the compression story:
  **116,619,267 seat-rows in, 18 MB out. ML never touches the big data.**
- **The failure gets its own slide** (B4), with the baseline comparison on it — the recency sort
  at ROC 0.800 against GBT's 0.857, and the fact that at the default threshold every model *loses*
  on F1 to "quiet ≥ 1 day".
- **The recommendation slide is the 0.2% overlap finding**: a retention list sorted by churn risk
  is almost exactly the wrong list. Screenshot the budget panel — the two curves make the argument
  faster than any sentence.
- **The closing slide is decisions, each with its number and its owner** (B5).

The dashboard is the source for every figure; screenshot it rather than retyping numbers.

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
- **`gold/rake_at_risk` has no stake column.** The stake filter uses each player's *modal* stake,
  computed over their venue's own prior window and cached at `data/_work/player_stake.parquet`.
  `p_max_stake` is the wrong join — it differs for a quarter of players.

---

## After this task, in order

1. **The report PDF** — the appendix needs three named items that are easy to forget: references,
   **AI-usage disclosure**, and **team contributions**.
2. **The notebook**, then Databricks when the team gets to it.

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
