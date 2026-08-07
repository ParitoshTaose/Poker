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
| `data/` | Bronze/Silver/Gold lake. **Contents gitignored**, structure committed via `.gitkeep`. `data/README.md` explains both download routes. |
| `notebooks/` | Empty. The executable Databricks/PySpark deliverable (Phases 2 & 3) goes here. |
| `deliverables/report/` · `deliverables/presentation/` | Empty. Consulting report PDF; exec deck (max 10 slides). |

All four HTML files are self-contained (no external CSS/JS/fonts/images) and cross-link to each
other **by bare filename** — they only work if all four stay in the same folder. Do not split them.

`src/` will grow one more script the guides already name and describe: `features.py` (PySpark).
Keep that name.

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
- **The real-money portion:** **21,605,687** No-Limit Hold'em hands, **1–23 July 2009**, stakes
  25NL–1000NL, six commercial platforms.
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
  distinct `hand_uid`, i.e. 147 uids covering 400 rows.** So `hand_uid` (= `"{site}:{hand_id}"`)
  is the right join key **but is NOT unique** — 0.002%. **De-duplicate before any join;** do not
  assume it away.
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
- **Money coverage is ~⅓.** Only 288/991 hands reconcile (211 lack `winnings`, 486 have it all-zero).
  **Behavioural features work on 100% of hands; money features do not.** Say so in the report.
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
- **Local Spark tuning:** `spark.driver.memory=9g` · `master("local[8]")` ·
  `spark.sql.shuffle.partitions=48` (default 200 is cluster-tuned and wrong locally) ·
  `spark.local.dir=/tmp/spark-scratch` · `spark.sql.files.maxPartitionBytes=64m` if OOM.
- **Never `.toPandas()` or `.collect()` on anything but the Gold table.** On a laptop the driver *is*
  your 16 GB. This is the #1 way to kill a local Spark job.
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

1. **Is a gambling-industry dataset acceptable to the professor?** Unanswered as of 2 Aug. A
   30-second office-hours question that de-risks three weeks of work. The responsible-gaming framing
   is the defence. **Ask before the team invests.**
2. **Which Databricks tier does the course use?** Free Edition vs a provided workspace.
3. ~~Only PokerStars files have been sampled.~~ **DONE 6 Aug** — all six venues sampled (Step 2) and
   then all 21,782 files parsed (Step 3). One-file-per-venue proved **insufficient**; see the
   cross-venue defects above.
4. **Why are 486/991 `winnings` arrays all zero?** Not explained. Affects money-feature coverage.
5. ~~Nothing has been run at full scale yet.~~ **Silver now built from all 21,782 Bronze files**
   (6 Aug). Gold/Spark (Step 4) and everything after it are still unrun.
6. **Is player-ID stability real ACROSS venues?** 91.7% overlap was measured *within* PokerStars only.
   Never merge IDs across venues — and confirm the within-venue figure holds for the other five.

## Honesty rules for anything added here

The professor will probe. Flag openly, in red `.box.brk` callouts:
- the data is from **July 2009** (17 years old) — argue the *method* transfers, don't hide the date;
- rake is **estimated**, not a recorded column;
- 23 days only ⇒ call it **"short-horizon lapse"**, not churn;
- **no ground truth for bots** — which is exactly why churn is the supervised label;
- players on two sites look like two different people — **never merge IDs across venues**.
