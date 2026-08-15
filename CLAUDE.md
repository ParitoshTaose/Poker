# CLAUDE.md — Project/ (BDA MBA group project, 20 marks)

This folder holds the **graded group project**, which is separate from the session study notes in
`../notes/`. Read the parent `../CLAUDE.md` for course-wide context. Read `../notes/CLAUDE.md` for the
design system — **the HTML files here clone it and must not diverge**.

The user is an MBA student (not a CS major), is a **total beginner at poker**, and has **never used
Spark or Databricks** (only Polars from sessions 3/7/8 and the MapReduce/HDFS sims from 1/2/4).
Explain jargon in plain language and always tie technical ideas to a business consequence.

## What is in this folder

> **Reorganised 2026-08-05** into a graded-repo layout. The files below moved out of the root into
> `docs/` and `src/`; nothing was renamed. Root now holds only `README.md`, `CLAUDE.md`, `.gitignore`.

| Path | What it is |
|---|---|
| `README.md` | **Repo front door.** Business problem, dataset + citation, layout, architecture, how to run, timeline, team table (TODO). Graded under Professionalism. |
| `docs/poker-project-plan.html` | **The plan.** 15 sections. Business problem → data → architecture → 3 phases → rubric map → what to submit. Interactive player-segment explorer. |
| `docs/poker-explained.html` | **Companion 01.** Poker from absolute zero → terminology → strategy → how each game concept becomes a data column. 12 sections. |
| `docs/data-pipeline-guide.html` | **Companion 02.** The 20 GB pipeline. Concept *and* runnable commands. Teaches Spark from scratch. 14 sections. |
| `docs/execution-roadmap.html` | **The checklist (v2, 6 Aug 2026).** Steps 0–9 in order: commands, "done when", and chips mapping each step to the phase/rubric/deliverable it earns. **v2 added a line-by-line plain-English `.walk` walkthrough under every command and script**, plus measurements taken while actually running Steps 1–3. Also holds the "one base, three forks" strategy. |
| `src/parse_phh.py` | **A tested, working parser.** Turns one `.phhs` file into 3 flat tables. Run it from the repo root: `python3.13 src/parse_phh.py <file.phhs>` |
| `src/build_silver.py` | **Bronze → Silver, done.** Parses all 21,782 files on 8 cores into batched Parquet parts. Run: `python3.13 src/build_silver.py`. **Must stay batched** — see gotchas. |
| `src/features.py` | **Silver → Gold (Spark), done.** Writes `gold/player_features` (one row per **site + player**) and `gold/hand_features` (one row per hand, Fork C's target `pot_bb`). Run: `.venv/bin/python src/features.py`; smoke-test on a subset with `--parts 3` (writes to `data/gold_sample/`, never clobbers real Gold). |
| `src/bakeoff.py` | **Step 5, done.** Five candidate ML questions prototyped in scikit-learn, each against its own baseline, to *choose* the fork. Throwaway by design — nothing here is graded. Writes `docs/bakeoff-results.json`. |
| `src/features_lapse.py` | **Fork A part 1, done.** Silver → `gold/player_lapse`: every feature recomputed over `day <= cutoff` only, plus recency/trend columns. This is the leakage fix the bake-off made Fork A conditional on. Run: `.venv/bin/python src/features_lapse.py` (3.8 min; `--rebuild` re-does the 116 M-row join from Silver instead of reusing `data/_work/hp_enriched`). |
| `src/fork_a.py` | **Fork A part 2, done.** The graded models in **PySpark MLlib**: logistic regression · GBT · random forest, on two feature sets, against three baselines, with the bake-off's leakage ablation re-run. Writes `docs/fork-a-results.json` and `gold/lapse_scores`. Run: `.venv/bin/python src/fork_a.py` (5.0 min; `--quick` for a 1.3-min smoke test). |
| `docs/fork-specs.html` | The three forks specced on paper *before* any model ran — exact target, feature list, honest risk each. |
| `docs/bakeoff-decision.md` · `bakeoff-results.html` | Step 5's scorecard and decision note: **Fork A wins, conditional on closing the leakage.** |
| `docs/fork-a-results.md` | **Fork A's results.** Population, label, baselines, metrics, the leakage re-check, the contact-budget table, and what is still weak. |
| `src/segments.py` | **The unsupervised layer, done.** MLlib K-Means (vs bisecting k-means) on prior-window behaviour, two views, cross-tabbed against Fork A's risk scores and the plan's four hypothesised archetypes. Writes `docs/segments-results.json` and `gold/player_segments`. Run: `.venv/bin/python src/segments.py` (2.0 min). |
| `docs/segments-results.md` | **The segmentation write-up** — and where the ecosystem thesis stops being an assumption and becomes a measurement. |
| `src/rake_at_risk.py` | **The money-weighted headline, done.** Recalibrates Fork A's GBT (drops `weightCol`, verifies with a reliability curve), prices every hand's rake with a validated conditional-mean estimator, attributes it to players by contributed share, and multiplies. Writes `docs/rake-at-risk-results.json` and `gold/rake_at_risk`. Run: `.venv/bin/python src/rake_at_risk.py` (2.1 min reusing `data/_work/player_rake`; `--rebuild-rake` redoes the 116 M-row attribution, 5.1 min total). **Run twice end to end: every reported figure reproduced identically.** |
| `docs/rake-at-risk-results.md` | **The headline write-up** — the sentence, the reliability curve, the rake estimator and its validation, who to call, and four sensitivities. |
| `docs/delivery-gate.html` | 59 checks across 8 gates (48 blockers, 11 lifts) between here and submission. |
| **`docs/next-session.md`** | **START HERE if picking the project up fresh.** The next task specced in full — the money-weighted "rake at risk" headline — plus what to read first, the traps already paid for, and the order of everything after it. |
| `data/` | Bronze/Silver/Gold lake. **Contents gitignored**, structure committed via `.gitkeep`. `data/README.md` explains both download routes. |
| `notebooks/` | Empty. The executable Databricks/PySpark deliverable (Phases 2 & 3) goes here. |
| `deliverables/report/` · `deliverables/presentation/` | Empty. Consulting report PDF; exec deck (max 10 slides). |

The HTML files are self-contained (no external CSS/JS/fonts/images) and cross-link to each
other **by bare filename** — they only work if they all stay in the same folder. Do not split them.

**Where the work has got to:** the lake is built (Bronze → Silver → Gold), Step 5's bake-off chose
**Fork A**, both models exist and are measured (Fork A in MLlib plus K-Means as the unsupervised
layer), and the **money-weighted headline is now built too** — so **the analysis is complete**.
**No graded artifact has been started.** `notebooks/` and both `deliverables/` folders are still
empty. The next steps are all writing and packaging: the dashboard, the deck, the report PDF, and
the Databricks notebook (Steps 7–8). See `docs/next-session.md`.

## The brief (from `../Github folder/nmims_analytics/sessions/Big_Data_Analytics_Project_Guidelines.pdf`)

Consulting-team framing: build an end-to-end Big Data solution turning raw data into business
decisions. Architecture spine: `Business Problem → Data Sources → Ingestion → Distributed Storage →
Data Engineering → Feature Engineering → ML → Business Insights → Executive Recommendations`.

- **Phase 1** (sessions 1–9) — ingestion + storage. HDFS/cloud/APIs. Kafka & HBase optional.
- **Phase 2** (sessions 10–13) — **Databricks + PySpark**. Clean, join, missing values, outliers,
  feature engineering, encoding + scaling. *Justify every decision.*
- **Phase 3** (sessions 14–18) — **PySpark MLlib**, **at least two** approaches, compare metrics.

**Deliverables:** (1) exec deck, **max 10 slides incl. title & thank-you**; (2) fully executable
notebook; (3) consulting report PDF — Exec Summary · Business Context · Data Understanding (5 Vs) ·
Enterprise Architecture · Data Engineering · ML · Business Insights · Strategic Recommendations ·
Appendix (**references, AI-usage disclosure, team contributions** — all three are named requirements).

**Rubric (20):** Business problem & storytelling 3 · **Data Engineering & Architecture 4** ·
Feature Engineering 3 · **ML Implementation 4** · Model Evaluation 2 · Business Recommendations 2 ·
Professionalism & Documentation 2.

**Dates (2026):** topic submission 1 Aug · approval window 1–10 Aug (office hours Mon/Wed/Fri
10:00–10:30, booked **through the CR**, no 1:1s) · submission **8 Sept** · presentations 11–12 Sept.
Max 6 per team. **GitHub repo with commit history across the trimester is graded.**

## The chosen topic

**"The Ecosystem Engine"** — an online poker operator earns only *rake* (a small cut of each pot), so
it survives only while recreational players keep sitting down. A small group of high-volume
professionals drains them, they churn silently, and revenue collapses. A **retention + game-integrity**
problem, not a game-AI problem. Three models: K-Means segmentation, churn classification
(label derivable from the data itself), value regression.

## Ground rules (do not re-derive these)

- **Never invent numbers.** Every figure in these files is either measured from the real data or
  explicitly flagged as an estimate. Keep that discipline — the green "Verified" panels mean
  *someone actually ran it*, and the footers list what was measured vs estimated.
- **Rules of thumb must be labelled as such.** The VPIP/PFR profiles (23/19 for a reg, 45/8 for a
  casual) in Companion 01 §06 are community benchmarks for intuition, **not** findings from this
  dataset. Same for the four archetype names in the plan's segment explorer.
- **Use only `data/handhq/`.** The Zenodo archive also holds 619M Annual Computer Poker Competition
  hands — bots playing bots. Huge, and useless for a business project.
- **Clone the design system, do not re-derive it.** `docs/poker-project-plan.html` holds the canonical
  `<style>` block for this folder (itself cloned from `../notes/session-1.html`). New pages copy it
  verbatim; only content changes.

## The dataset (verified 1–2 Aug 2026)

- **Source:** `github.com/uoftcprg/phh-dataset` · Zenodo DOI `10.5281/zenodo.17136841`
  (CC BY 4.0, v3, 16 Sep 2025). Repo is MIT.
- **Cite:** Kim, Juho. *"Recording and Describing Poker Hands."* IEEE Conference on Games (CoG), 2024.
  DOI `10.1109/CoG60054.2024.10645611`.
- **The real-money portion:** **21,605,687** No-Limit Hold'em hands, nominally **1–23 July 2009**,
  stakes 25NL–1000NL, six commercial platforms.
- **CORRECTED 2026-08-07 — the window is 1–26 July, and the six venues do NOT share it.** Every
  source folder is labelled `2009-07-01_2009-07-23`, but the hands' own timestamps run to **day 26**:
  470,790 hands (2.2%) fall on days 24–26. Measured last day **per venue**:
  **ABS 20 · PS 20 · FTP 23 · ONG 24 · IPN 26 · PTY 26**. The tail hands are structurally normal
  (same median pot 2.5bb, same stakes) but come from only 97 tables — a collection overrun on three
  venues, not corrupt dates. **Consequence for Fork A: any global "played in the last week?" rule
  labels every ABS and PS player as lapsed by construction — 9.55M hands, 44% of the dataset —
  because their data simply stops. Lapse windows must be measured backwards from each venue's own
  last day.** `features.py` does this via `site_last_day` / `days_since_last`.
- **CORRECTED 2026-08-06 — PS and PTY were swapped in the per-site figures.** Earlier notes said
  "PTY 8,298,718 … PS 3,092,698". **Measured over all 21.5M parsed hands it is the reverse.** The
  check is decisive: each file states its own `venue` internally and folder-derived site agrees
  **1:1 on every hand** (`venue=PS` ⇒ "PokerStars"). Correct split (measured):
  **PS 8,293,797 · IPN 5,996,741 · PTY 3,092,685 · ONG 1,631,839 · FTP 1,283,795 · ABS 1,257,978**.
  **PokerStars is the LARGEST venue (38%)**, which is also the one `parse_phh.py` was developed on.
- **Two download routes.** Develop against the GitHub route:
  `git clone --filter=blob:none --sparse …` then `git sparse-checkout set data/handhq`.
  Zenodo is the full archive, **exactly 20,289,230,983 bytes** (HTTP 200 verified). Extract
  selectively — `unzip <zip> '*/data/handhq/*'` — never blind.
- **CORRECTED 2026-08-06 — the GitHub route is NOT a 1.71 GB subset.** Earlier notes said 1.71 GB;
  **measured on disk after the sparse-checkout completed it is 15 GB / 21,782 `.phhs` files**
  (~21.7M hands ⇒ effectively the *whole* real-money dataset). The 1.71 GB figure was almost
  certainly read *before* `git sparse-checkout set` finished downloading blobs — `--filter=blob:none`
  makes the initial clone tiny. **Consequence: "Step 6 · scale to the full 20 GB" is largely already
  done**; verify by comparing the Silver `hands` count against 21,605,687 before downloading Zenodo.
- **Folder naming carries free metadata:** `{SITE}-{start}_{end}_{stake}NLH_OBFU/{big_blind}/*.phhs`.

### Measured from 3 real files (2,986 hands), `PS-…25NLH_OBFU/0.25/`

| Fact | Value | Why it matters |
|---|---|---|
| Player-ID stability across files | **91.7% overlap** | The make-or-break check. Same human keeps the same code — player-level analytics works. |
| File structure | consecutive **~2¼-minute** time slices, ~330 tables live | Real second-level timestamps, not placeholders. |
| Player actions per hand | **10.56** *(PS only — see below)* | the ×21.6M ⇒ ~229M projection **was 25% too high** |
| Players per hand | **6.17** *(PS only)* | true all-venue figure is **5.41** |
| Showdown rate | **22.4%** | ⇒ **77.6% of hole cards are never revealed** — structural, not sloppiness |
| Hands missing `winnings` | 21.3% | real data-quality work for Phase 2 |
| Hands missing `seat_count` | 4.3% | recoverable by counting `players` |

### Measured by running `parse_phh.py` on one file (989 hands)

```
hands=989  hand_players=6098  actions=10440
action mix: fold 4874 · check 1849 · call 1531 · bet 791 · raise 769 · show 626
rake: preflop-only 109/109 = $0.00 · flop hands median $0.05 = 3.70%
pooled VPIP 27.9% · PFR 10.2% · big-blind VPIP 27.4%
288 of 991 hands fully reconciled for money
```

### FULL-SCALE Silver build — measured 2026-08-06 (`build_silver.py`, all 21,782 files, 27.7 min, 8 cores)

```
SKIPPED FILES: 0
DROPPED HANDS: {'no_big_blind': 49119, 'no_hand_id': 133}
hands         21,556,835 rows  (21,556,435 distinct hand_uid)
hand_players 116,621,636 rows  (5.41 per hand)
actions      183,671,936 rows  (8.52 per hand)
```

- **PERFECT RECONCILIATION — use this in the report.** 21,556,435 distinct + 49,252 dropped =
  **21,605,687 = the published count, exactly.** Every hand is either parsed or dropped with a
  recorded reason. Total loss **0.23%**.
- **Storage:** Bronze 15.0 GB text → Silver **2.2 GB** Parquet (`hands` 297 MB · `actions` 897 MB ·
  `hand_players` 1.0 GB), 44 parts each = **6.8× compression**.
- **Correctness test at full scale:** pooled VPIP **27.4%**, PFR **14.1%**, **big-blind VPIP 31.3%**
  over all 21.5M BB seats — sane on all six venues, nowhere near ~100%.
- **The ~229M action estimate was 25% too high.** It scaled PokerStars' 10.56 actions/hand across
  everything; PS runs fuller tables than average. **True figure: 183.7M rows @ 8.52/hand** — still
  **175× Excel's limit**, so the Big Data argument is unaffected. Update the estimate wherever it appears.

### FULL-SCALE Gold build — measured 2026-08-07 (`features.py`, all 44 Silver parts, 4.6 min, 8 cores)

```
colliding hand_uids dropped : 147 uids / 547 hand rows  (=> 2,369 seat-rows)
venues with no table identity: ['IPN']
observation window per venue : ABS 20 · IPN 26 · FTP 23 · ONG 24 · PS 20 · PTY 26
hand_features   21,556,288 rows · 20 cols · 389 MB
player_features    106,977 rows · 41 cols ·  18 MB   (from 280,363 seen; 61.8% dropped at <100 hands)
```

- **The whole point of the medallion layout, in one line:** 116,619,267 seat-rows in → **18 MB** out.
  ML never touches the big data. Say this in the architecture section.
- **Every sanity check matches the parser exactly:** pooled VPIP **27.4% / 27.4%**, pooled PFR
  **14.1% / 14.1%**, big-blind VPIP **31.3% / 31.3%** (25.7% when heads-up is excluded).
- **THE money regression test — keep it forever.** Poker is zero-sum apart from the rake, so
  `Σ net_bb` must equal `−Σ rake_bb`: measured **−2,064,330 vs −2,064,329 bb, ratio 1.0000**. This is
  what proves the uncalled-bet bug is gone at full scale, and it is the check that caught it.
- **`bb_per_100` has a wild tail, and it is a sample-size artifact, not talent.** Median is a sane
  **−5.3** (rake drag), but **6.4% of players sit beyond ±200**. Cause: the 100-hand threshold is on
  `hands_played`, not on usable-money hands — median `money_hands` is only **92**, and just
  **88,655 of 106,977** players have any usable money. Worst case: 103 hands played, **7** with money,
  −1,494 bb/100. **Fork B must filter on `money_hands`, never `hands_played`.** Gold stores the raw
  value plus both counts on purpose — clipping here would hide the problem instead of letting Phase 2
  handle it, and "outliers" is a named part of that mark.

### FORK A — measured 2026-08-14 (`features_lapse.py` 3.8 min + `fork_a.py` 5.0 min, seed 42)

Full results and caveats: **`docs/fork-a-results.md`** · raw metrics `docs/fork-a-results.json`.

```
gold/player_lapse   90,469 players x 61 cols   (>= 100 PRIOR-window hands)
prevalence          68.20% lapsed (61,702)     train 67,873 / test 22,596
prior-window checks pooled VPIP 27.2% · PFR 13.7% · BB-VPIP 30.7%  (lifetime: 27.4 / 14.1 / 31.3)
seat-rows           93,430,449 prior + 23,188,818 label = 116,619,267
```

- **THE POPULATION FILTER IS A SELECTION LEAK, and the arithmetic proves it.** Gold's `>= 100
  hands` is a *lifetime* rule, so a player with 30 prior hands only clears it by playing 70+ hands
  **in the week being predicted**. Measured: lifetime rule keeps 98,718 players, prior-window rule
  keeps 90,469, and **both contain exactly 61,702 lapsers** — so all 8,249 extra players are
  non-lapsed, every one. Fork A must filter on `p_hands`, never on `hands_played`. Prevalence
  moves 62.5% → 68.2% as a result, so **PR-AUC is not comparable to the bake-off's**; quote
  ROC-AUC across the two.
- **The leakage the bake-off flagged is closed, and the honest features now WIN.** Same GBT, same
  population, same split: old floor (`hands_prior`+`tenure`) ROC **0.690** · lifetime rates (leaky)
  **0.819** · prior-window only **0.856**. In the bake-off the leaky set beat the honest one
  (0.886 vs 0.798 PR-AUC) — that ordering reversing is the point. The honest claim rests on
  construction (`day <= cutoff`), not on the scores.
- **Headline (CORE, all six venues, n=22,596 held out):** GBT **ROC 0.857 / PR 0.917**, RF 0.857 /
  0.918 (a tie), logistic 0.837 / 0.901. Baselines: flag-everyone 0.500 / 0.683; **rank by
  days-since-last-seen 0.800 / 0.864** — that free spreadsheet sort is the real bar, and it is
  the number to put beside every model in the report.
- **At the default 0.5 threshold every model LOSES on F1 to the dumb rule** "quiet >= 1 day"
  (0.856 vs GBT's 0.829). Tune the cut-point and GBT wins at 0.866 @ 0.25. The operating point is
  a decision, not a default — say so; it is free Model-Evaluation marks.
- **The business number:** at a 10% contact budget (2,260 calls), GBT finds **2,192 true lapsers
  vs recency-sort's 2,001** — 191 fewer wasted offers. Recall is only 0.142 at that budget and
  that is honest: 68% of players go quiet in any week, so **the real question is which lapsers are
  worth calling** — i.e. Fork B as a second stage, not a discarded loser.
- **EXTENDED (stacks + tables + money) is not worth it.** +0.007 ROC-AUC for dropping iPoker,
  i.e. 14.6% of players. **CORE is the spine.**
- Weakest where prevalence is lowest: ONG ROC 0.769 · ABS 0.780 vs PS 0.861.
- **Not done:** a time-shifted backtest (train at `last_day-14`, test at `last_day-7`) — the split
  is by player, so nothing is tested on a *different period*. Do not claim temporal generalisation.

### K-MEANS SEGMENTATION — measured 2026-08-14 (`segments.py`, 2.0 min, seed 42)

Full write-up: **`docs/segments-results.md`** · raw metrics `docs/segments-results.json`.
**Clustered on the PRIOR-WINDOW columns, not `player_features`** — lifetime rates include the
label week, so segments built on them partly encode the lapse answer before the cross-tab is drawn.

```
ECOSYSTEM view (5 venues, 77,268 players, k=4)   pooled lapse 70.5%
seg  name(interp)   share   hands  tables  vpip   pfr   bb/100   LAPSE   % of all lapsers
 1   Grinder         3.7%   8,799   11.5   0.18  0.12    -3.3    61.3%    3.2%
 0   Regular        35.4%     884    3.0   0.22  0.11    -7.2    61.6%   30.9%
 2   Gambler        19.8%     331    1.3   0.59  0.20   -31.8    74.6%   20.9%
 3   Recreational   41.2%     352    1.4   0.39  0.11   -14.3    77.1%   45.0%
```

- **THE HEADLINE FINDING — the ecosystem thesis, measured.** The two segments that lose money
  fastest are the two that leave fastest; the Grinder, who loses least, is the **stickiest**
  player on the platform. Gambler + Recreational = **60.9% of players but 65.9% of all lapsers**.
  The operator's problem is not "we lose players", it is **"we lose the players we earn from and
  keep the ones we don't."**
- **The grinder cluster only exists when volume + multi-tabling are in the feature set.** On
  betting style alone, no grinder forms — two of four clusters both map onto "Recreational".
  **Style cannot tell a professional from a customer; multi-tabling can.** This is the argument
  for the showpiece feature, and for why iPoker's missing table identity was nulled not imputed.
- **Archetype check vs `poker-project-plan.html`'s four hypotheses: 2 confirmed, 2 wrong.**
  Grinder ✓ (11.5 tables) and Recreational ✓. **The "Nit" (VPIP 12%) does not exist** — no
  extreme-tight cluster forms. **The "Gambler" is loose-PASSIVE, not loose-aggressive** (PFR 20%,
  not the hypothesised 35%). Report both misses; they are findings.
- **These are GRADIENTS, not natural groups — do not oversell them.** Silhouette peaks at **k=2**
  (0.439); k=4 costs 0.12–0.15 and was chosen for business readability + one-to-one archetype
  comparison. **Bisecting k-means beats k-means on silhouette in both views** (0.356 vs 0.317 ·
  0.319 vs 0.228) and the two agree on only **~66% of players**. That instability is the honest
  result.
- **Fork A's risk scores rank correctly but are NOT calibrated.** Mean predicted risk per segment
  (0.52 · 0.53 · 0.64 · 0.67) orders the segments exactly as observed lapse does (0.616 · 0.613 ·
  0.746 · 0.771) but sits ~9 points low — balanced class weighting pulls probabilities toward 0.5.
  **Recalibrate before multiplying any rupee figure by a risk score.**
- Lapse rate by segment × venue spans **30.9% → 87.2%**. A global view mixes incomparable
  populations — this is why the dashboard's venue filter is a gate blocker, not a nicety.
- `share_losing` in the raw output is a **small-sample artifact** (median money hands ≈ 92) —
  cluster *means* of bb/100 are reliable, the per-player sign is not. Not quoted in the write-up.

### RAKE AT RISK — measured 2026-08-14 (`rake_at_risk.py`, 5.1 min, seed 42)

Full write-up: **`docs/rake-at-risk-results.md`** · raw metrics `docs/rake-at-risk-results.json`
· output table `gold/rake_at_risk` (77,268 players × 23 cols).

```
THE SENTENCE   $604,163 of next week's rake sits at risk across 77,268 players on five venues
               — 30% of it recreational — and contacting the top 10% by expected loss
               (7,727 players) puts $421,064 of it in reach.
weekly rake    $2,114,626   (34.4% MEASURED, the rest estimated)
at-risk        58,836 players at calibrated P >= 0.5, holding $498,156 of weekly rake
whole window   $8,217,411 of rake on 15.56 M hands / 5 venues = $0.53 per hand
```

- **THE CALIBRATION FIX WORKED, AND IT WAS WORTH 30% OF THE HEADLINE.** Refitting Fork A's CORE
  GBT with `weightCol` simply **removed** (at 68/32 there was never an imbalance to correct)
  moves the worst reliability decile from **0.200 → 0.026** and ECE from 0.106 → 0.011, while
  ROC-AUC does not move (0.8566 → 0.8575). A 0.03 tolerance was declared in the source *before*
  the run, so **isotonic regression was not needed and not fitted**. Using the old uncalibrated
  scores would have understated the headline by **29.7%** ($424,810 vs $604,163).
  **Fork A's tuned threshold 0.25 belongs to the weighted scores and does NOT transfer** — the
  recalibrated best-F1 point is **0.40 → F1 0.867**.
- **THE OBVIOUS RAKE METHOD IS WRONG, AND MEASURABLY SO.** "Scale each player's observed rake
  per 100 hands to all their hands" inflates by **25%**, because the hands where rake reconciles
  are **not a random sample**: the parser accepts a residual only if it is ≤15% of the pot, which
  is easier to pass on a big pot. Measured on PartyPoker: reconciling hands have a **median pot
  of 23.4 bb and 99.5% saw a flop**, against **2.5 bb and 39.0%** on the rest. Scaling off them
  charges every hand at the rate of the biggest pots and ignores "no flop, no drop".
- **THE PLAUSIBILITY SCREEN — a new, decisive measurement.** Median rake is **4.3%–5.0% of the
  pot on all six venues** (the real commercial rate), but **on PokerStars, whose exports
  reconcile completely, NOT ONE of 1,107,219 flop hands rakes above 5.0%**. FTP (0.13% above 6%)
  and ONG (0.70%) agree; **ABS 41.5% and PTY 21.9% do not** (p90 share 11.3% and 13.3%). That
  tail is missing money, not rake, and it is worth **$267,789 on ABS and $677,131 on PTY**. So
  `rake_at_risk.py` counts a hand as *measured* only if `rake_bb <= 0.06 * pot_bb`; the rest are
  estimated. Post-screen coverage: **ONG 96.8% · ABS 45.6% · PS 26.9% · FTP 26.8% · PTY 11.1% ·
  IPN 0.0% (excluded)**.
- **The estimator, and it validates.** Conditional-mean rake **in big blinds** over
  (site, big_blind, saw_flop, pot bucket), falling back to (site, flop, bucket) then
  (flop, bucket); dollars only at the end. Fitted on half the measured hands and scored against
  the other half: **ratios 0.993–1.006 on every venue.** Per *hand* it is rough (PTY MAE $0.76 on
  a $1.64 mean) — trustworthy for sums, never for one hand. 87.1% of estimated dollars come from
  the most specific cell.
- **NEVER FIT RAKE CELLS IN DOLLARS.** The first version did, and priced 59,910 PokerStars 25NL
  monster pots at **$29.79 each** from a fallback cell built out of Ongame $10-blind hands. Fit
  in bb, multiply by `big_blind` at the end.
- **Two regression tests, both green at full scale, keep them:** contributed-share and
  equal-split attribution must total the same (**$8,217,411.49 both, ratio 1.000000** — this is
  what catches a fan-out on the non-unique `hand_uid`), and prior-window hand counts must equal
  `p_hands` in `gold/player_lapse` player by player (**0 mismatches / 77,268**).
- **THE BUSINESS FINDING: a retention list sorted by churn risk is almost exactly the wrong
  list.** Top 10% by risk alone vs top 10% by expected loss share **0.2% of their names**. The
  risk-only list finds *more* leavers (1,876 vs 1,013 on the held-out fold, 96.9% precision) and
  reaches **0.9% of the money**; expected loss reaches **70%**. Money-only reaches 63.7%.
- **Per head the Grinder is worth 16× the Recreational player** ($220.88 vs $13.59 of weekly
  rake), yet Recreational + Gambler hold **56% of the rake at risk** against the Grinder's 18%.
  **Not a contradiction — contributed rake measures who PAID, not whose money FUNDED it.** A
  grinder pays with money lost to them by recreational players (bb/100: Gambler −31.8,
  Recreational −14.3, Grinder −3.3). A "funded-by" attribution weighted by net losses has not
  been run and is the natural next measurement.
- **The headline is substantially a PokerStars number** (63% of players, 44.8% of the rake at
  risk) — and PS has the *lowest* measured share (15.5%), because its reconciling hands are its
  smaller pots. Per-player weekly rake spans **$18.99 (PS) to $103.82 (ONG)** on stakes alone.
- Dollars are **2009 USD at 25NL–1000NL**. No INR conversion (the 2009 rate would be invented)
  and no inflation adjustment — say so rather than quietly converting.

## Data & code gotchas (these bite — they are already flagged in the HTML)

### Cross-venue defects — found 2026-08-06 by running all 21,782 files (one-file-per-venue MISSED them all)

- **TOML types are VALUE-dependent — this is the big one.** `25` loads as `int`, `25.0` as `float`,
  so **the same field is both types across hands**. Audited across all 6 venues: `starting_stacks`,
  `winnings`, `blinds_or_straddles`, `antes`, `min_bet` **all mix int/float inside one list**.
  Polars inference guessed from early rows and **killed the run twice at the same batch with two
  different errors**. **Fix (do not regress): `parse_phh.py` coerces every value via `_f/_i/_s`
  helpers, and `build_silver.py` declares explicit `SCHEMAS` for all 3 tables.** Never let Polars
  infer these. A pre-flight validating all 27 venue/stake partitions against the schema takes
  seconds — run it before any long job.

- **ONG (Ongame) hand IDs are STRINGS** (`'R5-2483622-63'`); all five other venues use ints.
  Mixed types **crash the Parquet write** (`ComputeError: could not append value ... of type: str`)
  ~8,700 files in. **Fix in place: `parse_phh.py` forces `hand_id` to `str`.** Never revert.
- **Some hands have no `hand` key at all.** The old parser did `h["hand"]` unconditionally, so one
  bad hand raised `KeyError` and `build_silver.py`'s file-level `except` **discarded the whole file**
  — 31 FTP files ≈ 31k good hands lost to protect 31 bad ones. **Fix: drop the individual hand and
  count it.** Lesson: catch at the smallest unit that can fail.
- **Hand IDs are REUSED within a venue — measured at full scale.** No *cross*-venue collisions exist
  (ABS 3.02–3.09bn and IPN 3.41–3.49bn are disjoint), but **iPoker reissues the same id to genuinely
  different hands** (different time, player count, pot). Over all 21,556,835 rows: **21,556,435
  distinct `hand_uid`.** Precisely (re-measured 2026-08-07): **147 uids covering 547 rows**, i.e.
  400 *excess* rows — don't conflate the two figures. So `hand_uid` (= `"{site}:{hand_id}"`)
  is the right join key **but is NOT unique** — 0.002%. **De-duplicate before any join;** do not
  assume it away. `features.py` drops all 547 rows from both `hands` and `hand_players` rather than
  keeping one arbitrarily: once two real hands share a uid, a seat-row cannot be attributed to the
  right one, and leaving them in fans the 116M-row join out silently.
- **`parse_phh.py` originally parsed folder metadata for the ORIGINAL dataset layout**, so under the
  Bronze `venue=X/stake=Y/` layout it produced `stake="stake=0.25"` and lost the date-carrying
  `venue_dir`. **Fixed: `_folder_meta()` handles both layouts** and returns `stake` as a float plus a
  clean `site` code. `venue_dir` is `None` under the Bronze layout — the dates live in Bronze's path.
- **`cmd | tee log` masks Python's exit code** (you get `tee`'s). The first full run "succeeded" with
  exit 0 while actually having crashed. Use `set -o pipefail`, or redirect instead of piping.

### Format & domain gotchas

- **`.phhs` files are valid TOML.** `tomllib.load(open(p,'rb'))` (stdlib since Python 3.11) parses a
  whole file into typed objects; `time` returns a real `datetime.time`. **No regex parser needed.**
  Must be opened in **binary** mode.
- **`cc` = check OR call. `cbr` = bet OR raise.** Disambiguate `cc` by whether
  `max(street_bet) > that player's street_bet`. **`cbr`'s number is the cumulative street total, not
  an increment.**
- **THE correctness test: big-blind VPIP must be ~27%, not ~100%.** The big blind can check for free
  pre-flop, which looks identical to a call. Count it wrong and *every* player's VPIP inflates and the
  clustering becomes meaningless. `parse_phh.py` prints this check on every run — never ignore it.
- **Rake is derivable, but only with the uncalled-bet correction.** `pot = sum(contributions) −
  (highest − second-highest contribution)`; `rake = pot − sum(winnings)`. Omit the correction and you
  get a nonsensical ~40% median rake. With it: **preflop-only hands rake exactly $0.00 (109/109 —
  "no flop, no drop" confirmed)**, flop hands median **3.7%** in $0.05 steps, capped ~$0.65 at 25NL.
- **The uncalled bet must be returned to the PLAYER, not just removed from the pot — fixed
  2026-08-07, never regress it.** `parse_phh.py` originally subtracted the uncalled amount from the
  pot (so rake reconciled perfectly) but left it inside that player's `invested`, so they never got
  their own uncalled bet back. `net_bb` was therefore wrong on **99.4% of hands**, by a mean of
  **7.35 bb — 17× the rake itself (0.434 bb)**. Every high-volume player came out at ≈ **−100 bb/100**,
  i.e. losing a full big blind per hand, which is impossible in a rake-only game.
  **The test that catches it: poker is zero-sum apart from the rake, so `Σ net_bb == −rake_bb` on
  every reconciled hand.** Before the fix that identity was off by exactly the uncalled bet
  (`Σ net_bb + uncalled_bb + rake_bb = 0` held on 100% of 852,293 hands — algebraic proof, not a
  guess). `parse_phh.py`'s self-test now prints this identity on every run, next to the BB-VPIP check.
  **This forced a full Silver rebuild** — the old, wrong Silver is why `hand_players` was rebuilt.
- **Money coverage is ~⅓ overall, but it is a VENUE PROPERTY, not random — measured 2026-08-07.**
  This answers the old "why are the winnings arrays all zero?" question. Per hand, winnings are
  recorded for **every seat or none** (partial coverage: exactly 0 hands). Of 21,556,435 hands:
  **25.0% reconcile against the pot ("ok")**, 13.8% carry a full set of winnings that sums to
  **~1.5× the pot** (median) and are therefore unusable, 52.2% are present-but-all-zero, 9.0% absent.
  **Usable money by venue: ONG 99.1% · ABS 82.0% · PS 27.3% · FTP 27.2% · PTY 100% present but only
  14% reconciling · IPN 0.0% — iPoker reconciles on NONE of its 6.0M hands.**
  **Consequences: use `rake_bb IS NOT NULL` (not "winnings present") as the money filter, or every
  player looks like a heavy loser; Fork B is a four-venue question and must exclude iPoker entirely;
  never impute the zeros — that invents a break-even player who never existed.**
  **Behavioural features work on 100% of hands; money features do not.** Say so in the report.
  **REFINED 2026-08-14 — `rake_bb IS NOT NULL` is necessary but not sufficient.** A further
  screen is needed: a derived rake above **6% of the pot** is missing money wearing a rake's
  clothes (PokerStars, whose exports reconcile completely, never exceeds 5.0% on 1.1 M flop
  hands). It drops 41.5% of ABS's and 21.9% of PTY's reconciling flop hands. See the RAKE AT
  RISK section above, and use `rake_at_risk.py`'s `measured()` helper rather than re-deriving it.
- **iPoker records NO table identity — it writes the literal string `'HandHQ'` (the data vendor's
  name) into all 5,996,194 of its `table` fields. Measured 2026-08-07.** Distinct `table_id` per
  venue: **PS 11,355 · ONG 1,252 · ABS 1,046 · FTP 809 · PTY 697 · IPN 1**. PS also has 256,213
  (3.1%) nulls. **Why this matters: the multi-tabling detector is the showpiece feature** — one human
  plays one table, a professional grinder plays a dozen — and on iPoker every player scores exactly
  `max_tables = 1`. That is **a missing value wearing a plausible number**, and feeding it to K-Means
  files all **15,549 iPoker players (14.5% of the Gold table) as recreational single-tablers**, which
  is the precise opposite of the truth for the grinders among them. **`features.py` nulls
  `max_tables` / `avg_tables` / `distinct_tables` for any venue with ≤1 distinct table id and sets
  `tables_recorded = false`.** Never impute 1 there.
- **Position is free.** Players are indexed in blind order: `p1`=SB, `p2`=BB, `p3` acts first pre-flop,
  **highest index = button**. Holds for 3+ players; heads-up differs — exclude it.
- **A few hands per file are legitimately dropped** (no big blind ⇒ money cannot be normalised) —
  **2 in the PS sample but 8 in an FTP one**, so don't quote "two per file" as a constant.
  Count and report drops; silent data loss is what the "justify each decision" mark tests.
- **Normalise all money by the big blind.** $50 at 25NL and at 1000NL are not comparable; 200bb vs 5bb
  is. Standard metric is `bb/100`.

## The machine (checked 2 Aug 2026)

`Apple M4 · 10 cores (4P/6E) · 16 GB RAM · 272 GB free · OpenJDK 17.0.19 · Homebrew · Python 3.13 & 3.14`

- **RAM is not the constraint.** Raw text ~18 GB (est.) → Parquet 2–4 GB (est.) → **Gold table ~200 MB
  (est.)**. ML never touches the big data.
- **No Python 3.11/3.12 installed.** PySpark may reject 3.13 → `brew install python@3.12` and rebuild
  the venv. `tomllib` works on any 3.11+, so only Spark is fussy.
- **Bare `python3` is Apple's 3.9.6 and has NO `tomllib`** (verified 2026-08-05: `/usr/bin/python3`
  wins the plain-shell PATH; the user's interactive shell is conda `(base)`, which may differ).
  `parse_phh.py` dies with `ModuleNotFoundError: No module named 'tomllib'`. Always invoke it as
  **`python3.13`** (or `/opt/homebrew/bin/python3`, which is 3.14.4). Both have `tomllib`.
- **Local Spark tuning (as actually used in `src/features.py`):** `master("local[8]")` ·
  `spark.driver.memory=9g` · **`spark.sql.shuffle.partitions=96`** (200 is cluster-tuned; **48 was
  too coarse** for 116.6M-row shuffles on a 9 GB heap) · `spark.sql.files.maxPartitionBytes=64m` ·
  `spark.local.dir=data/_work/spark-scratch`.
- **`spark.driver.memory` DOES work from the builder in local mode (PySpark 4.2) — verified.** A lot
  of advice says it cannot. Measured JVM max heap: **no config → 1.0 GB**, `.config(...,"9g")` → 9.0 GB,
  `PYSPARK_SUBMIT_ARGS="--driver-memory 9g"` → 9.0 GB. Use the builder form.
- **NEVER put `spark.local.dir` in `/tmp` — this killed two full runs, 2026-08-07.** Spark spills
  many GB of shuffle data there, and anything that tidies `/tmp` mid-run (macOS's cleaner, a sandbox,
  a CI agent) deletes it underneath the job. It dies ~10 min in with
  `java.io.FileNotFoundException: /private/tmp/spark-scratch/blockmgr-*/shuffle_*.data`, which reads
  like a Spark bug and is not one. **Point scratch at the project disk** (`data/_work/spark-scratch`,
  gitignored, 230 GB free). Symptom to recognise: the job dies at a *shuffle read*, never at a write.
- **Never `.toPandas()` or `.collect()` on anything but the Gold table.** On a laptop the driver *is*
  your 16 GB. This is the #1 way to kill a local Spark job.
- **PIN `PYSPARK_PYTHON`, or Spark launches its workers under Apple's Python 3.9 — found
  2026-08-14.** In a plain (non-conda) shell `/usr/bin/python3` wins the PATH, so Spark starts
  workers with 3.9, which **cannot even import pyspark 4.2**: `TypeError: unsupported operand
  type(s) for |: 'type' and 'type'` in `sql/types.py`, buried inside a Java stack trace that says
  nothing about Python versions. **Fix, already in `features_lapse.py` and `fork_a.py`:**
  `os.environ.setdefault("PYSPARK_PYTHON", sys.executable)` (and `PYSPARK_DRIVER_PYTHON`)
  **before `import pyspark`**. `features.py` and `build_silver.py` do NOT have this guard yet —
  they happen to work from the conda shell. Only bites where a Python worker is needed.
- **MLlib's `Imputer` accepts only `DoubleType`/`FloatType`.** An int/long column fails the schema
  check rather than being promoted. Cast every numeric feature to double once, up front.
- **`GBTClassifier` is NOT bit-reproducible even with a fixed seed** — it sums tree histograms in
  task-completion order, so two runs of identical code differ around the 6th decimal (measured:
  ROC 0.856638 vs 0.856641). `LogisticRegression` and `RandomForestClassifier` reproduce exactly.
  Quote 3 decimals and do not chase the wobble.
- **`spark.ui.showConsoleProgress=false`** whenever output is redirected to a log — the progress
  bar writes `\r`-heavy noise that makes the log unreadable afterwards.
- **Databricks Community Edition retired 1 Jan 2026** → use **Databricks Free Edition**.

## HTML build rules

New pages go in `docs/`. Clone the `<style>` block and page structure from
`docs/poker-project-plan.html`. Structure: `.progress` ·
`.topbar` · sticky `.rail` scroll-spy · hero (eyebrow, serif headline, deck, 4-colour legend,
grounded-note) · numbered `.sec` sections · callout boxes (`.box.take` cyan · `.box.board` gold ·
`.box.brk` red · `.box.discuss` gold-dashed) · green `.ground` "Verified" panels · `.tw`/`table` ·
`.code` blocks with hand-applied `.cm/.kw/.fn/.st` spans · `.closer` + `.next` card · footer listing
exactly what was measured vs estimated.

### Baked-in fixes — DO NOT regress

- **Set SVG `fill` via inline `style`, never `setAttribute('fill', …)`.** A CSS rule like
  `.demo .pt{fill:…}` **beats a presentation attribute**, so attribute writes are silently ignored and
  highlighted elements stay grey. (Found and fixed in the plan's segment explorer.)
- **SVG state elements need an explicit base `fill`/`stroke` in CSS**, or a bare `transition` rule
  leaves them black.
- **`.tree` blocks need `white-space:pre`** or ASCII trees collapse onto one wrapped line.
- **Scroll reveal must be scroll-driven** (reveal `.rv` when `getBoundingClientRect().top <
  innerHeight*0.92`), not a bare IntersectionObserver — the IO version blank-flashes on rail jumps.
- **Scroll-spy** = last section whose top passed a ~150 px line.

### Verify workflow (the Chrome extension cannot open `file://`)

```bash
cd Project/docs && python3 -m http.server 8793   # background — serve from docs/, not the root
# open http://localhost:8793/<file>.html ; add ?v=N to bust cache after edits
```
Test every interactive demo with **real clicks**. Kill the server when done. Known screenshot
artifacts after a programmatic scroll: `.rv` caught mid-transition, stale rail highlight. Force
`document.querySelectorAll('.rv').forEach(e=>e.classList.add('in'))` and
`document.documentElement.style.scrollBehavior='auto'` before screenshotting.

## Still open — do not present these as settled

1. ~~Is a gambling-industry dataset acceptable to the professor?~~ **CLEARED — the professor
   approved the topic (confirmed by the user 14 Aug 2026; the approval itself was earlier).**
   Delivery-gate blocker 0.1 is closed. Keep the responsible-gaming framing in the report anyway —
   it is the right framing for the work, not just a defence: this is a retention and
   player-protection problem, not a how-to-win-at-poker problem.
2. **Which Databricks tier does the course use?** Free Edition vs a provided workspace. Still
   unanswered — but **the team has deliberately deferred Databricks to a later part of the project
   (decided 14 Aug 2026)**, so this is no longer near-term blocking. Note the notebook deliverable
   itself does not have to wait for it: the logic already runs as local PySpark, so moving it is a
   re-hosting job, not a rewrite.
3. ~~Only PokerStars files have been sampled.~~ **DONE 6 Aug** — all six venues sampled (Step 2) and
   then all 21,782 files parsed (Step 3). One-file-per-venue proved **insufficient**; see the
   cross-venue defects above.
4. ~~Why are 486/991 `winnings` arrays all zero?~~ **ANSWERED 7 Aug — it is a venue property, not a
   random defect.** Full breakdown in the money-coverage gotcha above. iPoker records nothing usable
   on any of its 6.0M hands; PartyPoker records winnings on 100% of hands but only 14% reconcile.
   What remains genuinely unexplained is *why the operators' exports differ* — but the effect is now
   measured, bounded, and handled in code.
5. ~~Nothing has been run at full scale yet.~~ **Silver rebuilt from all 21,782 Bronze files with the
   corrected parser (7 Aug), Gold/Spark (Step 4) built from it, Step 5's bake-off run (12 Aug),
   **Fork A trained in MLlib, K-Means segmentation, and the money-weighted "rake at risk"
   headline (14 Aug)** — see the three measured sections above. **The analysis is complete.**
   Still unrun/unstarted: the dashboard, the deck, the report PDF, and the Databricks notebook
   (Steps 7–8) — i.e. writing and packaging only.**
6. **Is player-ID stability real ACROSS venues?** 91.7% overlap was measured *within* PokerStars only.
   Never merge IDs across venues — and confirm the within-venue figure holds for the other five.

## Honesty rules for anything added here

The professor will probe. Flag openly, in red `.box.brk` callouts:
- the data is from **July 2009** (17 years old) — argue the *method* transfers, don't hide the date;
- rake is **estimated**, not a recorded column;
- 23 days only ⇒ call it **"short-horizon lapse"**, not churn;
- **no ground truth for bots** — which is exactly why churn is the supervised label;
- players on two sites look like two different people — **never merge IDs across venues**.
