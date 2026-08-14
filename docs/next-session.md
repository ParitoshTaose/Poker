# Next session — the task, in full

**Hand this file to a new session.** Everything it needs to start is either here or named here.

---

## Read these first (in this order)

1. `Project/CLAUDE.md` — especially the two measured sections **FORK A** and **K-MEANS
   SEGMENTATION**, and the whole *Data & code gotchas* block. The gotchas cost days to find; do
   not rediscover them.
2. `docs/fork-a-results.md` — the lapse model, its baselines, and its known defects.
3. `docs/segments-results.md` — the segments, and the calibration warning at the end of it.

## Where the project stands

The pipeline and the models are **done**: Bronze → Silver → Gold, then Fork A (short-horizon lapse
classification, three MLlib algorithms against three baselines) and K-Means segmentation. All three
Phase-3 model types exist and are measured.

The topic is **cleared with the professor** — that blocker is closed.

What does *not* exist yet: **any graded artifact.** `notebooks/` and both `deliverables/` folders
are empty. Submission is **8 Sept 2026**; presentations 11–12 Sept.

**Databricks is deliberately deferred to later in the project** (user's decision, 14 Aug). Note the
notebook itself does not have to wait for it — the logic is already written as local PySpark, so
the notebook is a re-hosting job whenever it happens.

---

## THE TASK — the money-weighted headline ("rake at risk")

**Build `src/rake_at_risk.py`.** This is the last piece of *analysis* in the project. Everything
after it is writing and packaging.

**Why it comes first:** the dashboard headline, slide 1, the report's executive summary and the
recommendations section all resolve to the same single sentence. Write them before this number
exists and they get written twice.

**The sentence it has to produce:**

> "$X of weekly rake sits with N at-risk players — Y% of it in the recreational segment — and
> contacting the top 10% by expected loss puts $Z of it in reach."

### Step 1 — Recalibrate the lapse probabilities (do this first, it gates the rest)

Fork A's GBT was fitted with balanced class weights, which pulls probabilities toward 0.5. Measured:
mean predicted risk per segment (0.52 · 0.53 · 0.64 · 0.67) orders the segments exactly right but
sits **~9 points below** the observed lapse rates (0.616 · 0.613 · 0.746 · 0.771).

**Multiplying dollars by an uncalibrated probability produces a confidently wrong number.** So:

- Refit the CORE GBT **without `weightCol`**, on the same hash split (seed 42, 25% test) from
  `src/fork_a.py`. At 68/32 there was never a real imbalance to correct, so weighting was probably
  unnecessary in the first place.
- **Verify with a reliability curve:** bin the test set into 10 deciles of predicted probability,
  compare mean predicted against observed lapse rate in each. Report the largest deviation.
- Confirm ranking is unharmed — ROC-AUC should stay near **0.857**. Only calibration should move.
- If it is still off, fit `pyspark.ml.regression.IsotonicRegression` on the *train* fold's
  (score → label) and apply it to test. Never fit calibration on the test fold.

The before/after reliability curve is itself Model Evaluation material — keep it for the report.

### Step 2 — Attribute rake to players

Use the **contributed-rake** method, which is the industry-standard attribution:

```
player_rake_bb = hand.rake_bb × (player.invested_bb / Σ invested_bb over that hand)
```

- Inputs: `data/_work/hp_enriched` (seat rows: `hand_uid`, `player_id`, `invested_bb`, `site`,
  `day`, `money_ok`) joined to `data/gold/hand_features` for `rake_bb` and `big_blind`.
- **The money filter is `rake_bb IS NOT NULL`, never "winnings present."** This is a recorded
  gotcha — 13.8% of hands carry a full set of winnings that sums to ~1.5× the pot and are
  unusable.
- Aggregate **over the prior window only** (`day <= cutoff`), so it lines up with Fork A's
  features. Per player: `p_rake_usd_observed`, `p_rake_hands`.
- Dollars, not big blinds: `rake_bb × big_blind`. Money is comparable across stakes only after
  that conversion.

**The coverage hole, and how to handle it honestly.** Rake reconciles on only ~25% of hands and
it is a venue property, not random: ONG 99.1% · ABS 82.0% · PS 27.3% · FTP 27.2% · PTY 14% ·
**IPN 0.0%**.

- **Exclude iPoker entirely.** It reconciles on none of its 6.0M hands. Never impute it.
- For the other five, compute an observed **rake per 100 hands** from the covered hands, then scale
  to the player's full prior-window hand count. **Label this an estimate everywhere it appears**
  and report the observed total beside the estimated one, so the size of the extrapolation is visible.
- Say "five venues, N% of players" out loud every time the money number is quoted.

### Step 3 — The headline

- **Weekly rake** = rake attributable in the **last 7 days of the prior window** (the `p_hands_w1`
  window). This matches the 7-day lapse horizon, so the two halves of the multiplication describe
  the same period.
- `expected_rake_at_risk = Σ [ calibrated P(lapse) × weekly_rake_usd ]` over players active at the
  cutoff.
- Slice by **segment** (join `data/gold/player_segments`) and by **venue**. Both are mandatory —
  the venue spread on lapse alone runs 30.9%–87.2%.
- **The budget version:** rank by `P(lapse) × weekly_rake` (expected loss), take the top 10%, and
  report the dollars in reach.
- **Then compare that ranking against ranking by `P(lapse)` alone.** Does weighting by money change
  who you would actually call? Either answer is a finding worth a paragraph — and if it barely
  changes the list, say so.

### Step 4 — Write it up

- `docs/rake-at-risk-results.md` + `docs/rake-at-risk-results.json`, following the shape of
  `docs/fork-a-results.md`: what was run, the numbers, then a section on where it is weak.
- Update `Project/CLAUDE.md` with a measured section, as was done for Fork A and the segments.

### Done when

- The headline sentence exists with real numbers in it.
- The reliability curve is reported and the max calibration deviation stated.
- Coverage caveats are explicit: five venues, ~25% of hands measured, the rest estimated and labelled.
- Every number traces to a named Gold column — no hand-typed figures.

---

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

1. **The dashboard** — the faculty's explicit steer, and the terminal artifact. Gate A in
   `docs/delivery-gate.html` has the full checklist; A1 (opens on a decision, not a chart) and A8
   (money-weighted headline) are what this task unlocks.
2. **The deck** — max 10 slides including title and thank-you.
3. **The report PDF** — the appendix needs three named items that are easy to forget: references,
   **AI-usage disclosure**, and **team contributions**.
4. **The notebook**, then Databricks when the team gets to it.

Two small things that are pure marks and take minutes: the **team table in `README.md` is still a
`TODO`**, and Gate 0.2 — **which Databricks tier** the course uses — is still unanswered.
