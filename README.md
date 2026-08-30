# The Ecosystem Engine

**Big Data Analytics group project · NMIMS MBA Business Analytics · Trimester IV (2026)**

An end-to-end Big Data solution for an online-poker operator: segment the player base,
predict which recreational players are about to stop playing, and estimate what each
player is worth — so retention spend goes where it actually protects revenue.

---

## The business problem in one paragraph

An online poker operator does not win or lose money on the cards. It earns **rake** — a small
percentage of each pot, capped. That means its revenue depends entirely on *how many hands get
played*, which depends on recreational players continuing to sit down. A small group of
high-volume professionals systematically wins money from those recreationals; the recreationals
lose, get discouraged, and leave **without ever complaining**. Liquidity drains, tables stop
filling, and revenue collapses from the bottom up. This is a **retention and game-integrity**
problem, not a game-AI problem.

## The data

| | |
|---|---|
| Source | [`uoftcprg/phh-dataset`](https://github.com/uoftcprg/phh-dataset) · Zenodo DOI [`10.5281/zenodo.17136841`](https://doi.org/10.5281/zenodo.17136841) (CC BY 4.0) |
| Scale | **21,605,687** real-money No-Limit Hold'em hands → **116,621,636 seat-rows** and **183,671,936 action-rows** (111× and 175× Excel's row limit) |
| Period | **1–26 July 2009**, and the six venues do **not** share a calendar — each stops on its own last day (ABS 20 · PS 20 · FTP 23 · ONG 24 · IPN 26 · PTY 26) · stakes 25NL–1000NL |
| Format | `.phhs` files — valid TOML, one hand per `[[hand]]` table |

**Cite:** Kim, Juho. *"Recording and Describing Poker Hands."* IEEE Conference on Games (CoG), 2024.
DOI [`10.1109/CoG60054.2024.10645611`](https://doi.org/10.1109/CoG60054.2024.10645611)

> **Honest limitations** (stated up front, also in the report): the data is from **2009** — we argue
> the *method* transfers, we do not hide the date. Rake is **derived**, not a recorded column, and only
> **34.4%** of it reconciles exactly. The window is 26 days, so the churn label is a **short-horizon
> lapse**, not true churn. There is **no ground truth** for who is a bot. Player IDs are per-venue and
> are **never merged across venues**.

## Repository layout

```
Project/
├── docs/                    Guides, results notes and the decision surface — self-contained HTML
│   ├── dashboard.html               **The decision surface** — 77,268 players, opens from file://
│   ├── poker-project-plan.html      The plan: problem → architecture → 3 phases → rubric map
│   ├── poker-explained.html         Companion 01 — poker from zero, game concept → data column
│   ├── data-pipeline-guide.html     Companion 02 — the 20 GB pipeline, Spark from scratch
│   ├── *-results.md · *-results.json  One measured write-up per model, each with its weaknesses
│   └── pipeline-facts.json          The lake, re-measured from disk by src/pipeline_facts.py
├── src/                     The pipeline — Bronze → Silver (Polars) → Gold → MLlib (PySpark)
│   ├── parse_phh.py                 PHH → 3 flat tables; prints both correctness checks
│   ├── build_silver.py              All 21,782 files → 3 Parquet tables, 8 cores, 27.7 min
│   ├── features.py · features_lapse.py  Spark → Gold; the second is prior-window only
│   ├── bakeoff.py                   5 candidate questions scored against their own baselines
│   ├── fork_a.py · segments.py      The graded MLlib models — GBT/RF/logistic, K-means
│   ├── rake_at_risk.py              Calibration + rake pricing + attribution → the headline
│   ├── pipeline_facts.py            Re-measures the lake → docs/pipeline-facts.json
│   └── dashboard_data.py · deck_charts.py · report_figures.py · report_fit.py
│                                    Inject every figure into the three deliverables, and check them
├── notebooks/               The executable Databricks / PySpark deliverable (Phases 2 & 3)
├── data/                    The data lake — Bronze / Silver / Gold. Contents gitignored.
├── deliverables/
│   ├── report/                      **Consulting report — 35 pages, HTML + PDF** (see its README)
│   └── presentation/                **Executive deck — 10 slides, HTML + PDF** (see its README)
├── CLAUDE.md                Working context for AI-assisted development
└── README.md                This file
```

## Architecture

Medallion (Bronze → Silver → Gold) on Databricks with PySpark. Never destroy the original;
never re-download 20 GB because of a bug downstream.

| Layer | What lives there | Why |
|---|---|---|
| **Bronze** | `.phhs` files copied untouched, partitioned `venue=…/stake=…` | Permanent replayable record. Folder names carry free metadata. |
| **Silver** | `hands`, `hand_players`, `actions` as Parquet | Typed, columnar, queryable. Parsing done once. |
| **Gold** | `player_lapse` — one row per player, 61 prior-window features | **21 MB** measured. 116.6 M seat-rows in, 21 MB out — ML never touches the big data. |

## Models (Phase 3 — at least two required; we shipped three approaches)

1. **K-Means segmentation** (vs bisecting K-means) — four player archetypes from prior-window
   behaviour. Two of the four hypothesised archetypes turned out to be **wrong**, and we report it.
2. **Short-horizon lapse classification** — GBT · random forest · logistic regression, scored
   against three baselines. GBT **ROC-AUC 0.857** against a free spreadsheet sort's **0.800**.
3. **The money layer** — recalibrate the classifier, price every hand's rake, attribute it to the
   players who paid it. **$604,163 of next week's rake at risk**; a churn-sorted call list shares
   **0.1%** of its names with the right one.

Full results, with their own weaknesses sections: `docs/fork-a-results.md`,
`docs/segments-results.md`, `docs/rake-at-risk-results.md`, `docs/bakeoff-decision.md`.

## Running it

```bash
# Parse one hand-history file into 3 tables (prints the big-blind VPIP correctness check)
python3.13 src/parse_phh.py <file.phhs>

# Rebuild the lake and the models, in order (needs the venv: PySpark + Polars)
.venv/bin/python src/build_silver.py        # 27.7 min · Bronze -> Silver
.venv/bin/python src/features_lapse.py      # 3.8 min  · Silver -> Gold, prior-window only
.venv/bin/python src/fork_a.py              # 5.0 min  · the graded MLlib models
.venv/bin/python src/segments.py            # 2.0 min  · K-means vs bisecting K-means
.venv/bin/python src/rake_at_risk.py        # 2.1 min  · calibration + money -> the headline

# Rebuild the deliverables' numbers from those runs (seconds, no Spark)
.venv/bin/python src/pipeline_facts.py      # re-measure the lake
.venv/bin/python src/report_figures.py      # the report   (--check to verify only)
.venv/bin/python src/report_fit.py          # no page of the report may overflow its sheet
.venv/bin/python src/deck_charts.py         # the deck     (--check to verify only)
.venv/bin/python src/dashboard_data.py      # the dashboard

# Read the guides (the plan cross-links to both companions)
cd docs && python3.13 -m http.server 8793
# → http://localhost:8793/poker-project-plan.html
```

`parse_phh.py` needs **Python 3.11+** and no third-party packages — it uses stdlib `tomllib`.
Everything else needs the project venv.

> **Use `python3.13`, not `python3`.** On macOS the bare `python3` resolves to Apple's system
> Python **3.9.6**, which has no `tomllib`, and the parser dies with `ModuleNotFoundError`.

## Timeline

| Date (2026) | Milestone |
|---|---|
| 1 Aug | Topic submitted |
| 1–10 Aug | Approval window — office hours Mon/Wed/Fri 10:00–10:30, booked through the CR |
| **8 Sept** | **Submission** — deck, executable notebook, consulting report |
| 11–12 Sept | Presentations |

## Team

<!-- TODO: fill in before submission. Max 6 members. The same split is a graded appendix
     requirement — Appendix C of the report carries this table and must match. -->

| Name | Roll no. | Workstream owned | Artifacts |
|---|---|---|---|
| _TBD_ | _TBD_ | Data acquisition & Bronze/Silver build | `src/parse_phh.py` · `src/build_silver.py` |
| _TBD_ | _TBD_ | Feature engineering & Gold | `src/features.py` · `src/features_lapse.py` |
| _TBD_ | _TBD_ | Supervised modelling & evaluation | `src/bakeoff.py` · `src/fork_a.py` |
| _TBD_ | _TBD_ | Segmentation & the money layer | `src/segments.py` · `src/rake_at_risk.py` |
| _TBD_ | _TBD_ | Decision surface & visual design | `docs/dashboard.html` · `src/dashboard_data.py` |
| _TBD_ | _TBD_ | Narrative, report & deck | the report · the deck · `src/report_figures.py` |

**Also outstanding:** which Databricks tier the course uses (Free Edition vs a provided
workspace). The notebook does not have to wait for it — the logic already runs as local PySpark,
so moving it is a re-hosting job.

## Assessment

Marks are also awarded for **commit history across the trimester**. Commit small and often —
one commit per real step (parser fix, Silver build, feature set, report draft), not one dump
at the end.
