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

**The analysis is complete.** Bronze → Silver → Gold, then Fork A (short-horizon lapse
classification, three MLlib algorithms against three baselines), K-Means segmentation, and the
money-weighted "rake at risk" headline. All three Phase-3 model types exist and are measured, and
every model output now has a dollar figure attached to it.

The topic is **cleared with the professor** — that blocker is closed.

What does *not* exist yet: **any graded artifact.** `notebooks/` and both `deliverables/` folders
are empty. Submission is **8 Sept 2026**; presentations 11–12 Sept. Everything remaining is
writing and packaging.

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

## THE TASK — the dashboard

**The money-weighted headline is DONE** (`src/rake_at_risk.py`, 14 Aug — see
`docs/rake-at-risk-results.md`). The analysis is finished; the dashboard is the terminal artifact
and the faculty's explicit steer.

`docs/delivery-gate.html` **Gate A** is the checklist, and it is the spec — 7 blockers, 3 lifts.
Build it the way the four guides in `docs/` are built: **one self-contained HTML file, no server,
no CDN**, cloning the `<style>` block from `docs/poker-project-plan.html` verbatim.

**A1 — it opens on a decision, not a chart.** The top of the screen is the headline sentence
above, in words, with the number large. Charts justify it; they never replace it.

**A3 — the filters are venue, segment and stake, and venue is mandatory.** The venues do not share
an observation window, a lapse rate (40.4%–77.9%), or money coverage (11.1%–96.8%), so a global
view mixes populations that do not belong in one average.

**A4 — iPoker is excluded or visibly flagged wherever money or table count appears.** It has no
rake and no table identity. Never render `max_tables = 1` for it.

**A5 — every number traces to a Gold column.** `gold/rake_at_risk` (77,268 × 23) is now the
primary source: `weekly_rake_usd`, `weekly_rake_usd_measured`, `measured_share`, `risk_cal`,
`expected_rake_at_risk`, `segment_id`. Join `gold/player_segments` and `gold/player_lapse` for
behaviour. Export what the page needs to a small JSON and inline it — do not hand-type figures.

**A9 (lift) — drill from segment → player list → the action.** `gold/rake_at_risk` sorted by
`expected_rake_at_risk` *is* the call list; the top 12 rows are in the run log.

**Show the measured share wherever a dollar figure appears.** Two thirds of the weekly rake is
estimated, and the page must not pretend otherwise — a small "34% measured" chip beside the
headline does the whole job.

### Done when

- A teammate opens the file on their own machine and states a business decision unaided (A7).
- Every panel survives "so what do I do?" said out loud (A2).
- Venue, segment and stake filters all change the decision, not the decoration (A3).
- No number on screen is hand-typed (A5).

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

---

## After this task, in order

1. **The deck** — max 10 slides including title and thank-you. Slide 1 is the headline sentence;
   the risk-only-vs-expected-loss finding is the recommendation slide.
2. **The report PDF** — the appendix needs three named items that are easy to forget: references,
   **AI-usage disclosure**, and **team contributions**.
3. **The notebook**, then Databricks when the team gets to it.

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
