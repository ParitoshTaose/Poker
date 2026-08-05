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
| Scale | **21,605,687** real-money No-Limit Hold'em hands ≈ **229 M action rows** (~218× Excel's row limit) |
| Period | **1–23 July 2009** · stakes 25NL–1000NL · six commercial platforms |
| Format | `.phhs` files — valid TOML, one hand per `[[hand]]` table |

**Cite:** Kim, Juho. *"Recording and Describing Poker Hands."* IEEE Conference on Games (CoG), 2024.
DOI [`10.1109/CoG60054.2024.10645611`](https://doi.org/10.1109/CoG60054.2024.10645611)

> **Honest limitations** (stated up front, also in the report): the data is from **2009** — we argue
> the *method* transfers, we do not hide the date. Rake is **derived**, not a recorded column. The
> window is 23 days, so the churn label is a **short-horizon lapse**, not true churn. There is **no
> ground truth** for who is a bot. Player IDs are per-venue and are **never merged across venues**.

## Repository layout

```
Project/
├── docs/                    Written guides — open in a browser, self-contained HTML
│   ├── poker-project-plan.html      The plan: problem → architecture → 3 phases → rubric map
│   ├── poker-explained.html         Companion 01 — poker from zero, game concept → data column
│   └── data-pipeline-guide.html     Companion 02 — the 20 GB pipeline, Spark from scratch
├── src/                     Reusable Python (Bronze → Silver, local, no Spark needed)
│   └── parse_phh.py                 Tested PHH → 3 flat tables parser
├── notebooks/               The executable Databricks / PySpark deliverable (Phases 2 & 3)
├── data/                    The data lake — Bronze / Silver / Gold. Contents gitignored.
├── deliverables/
│   ├── report/                      Consulting report (PDF) + appendix material
│   └── presentation/                Executive deck — max 10 slides incl. title & thank-you
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
| **Gold** | `player_features` — one row per player | ~200 MB. Small enough that ML never touches the big data. |

## Models (Phase 3 — at least two required; we do three)

1. **K-Means segmentation** — unsupervised. Who are the natural player archetypes?
2. **Churn classification** — supervised. The label is derivable from the data itself.
3. **Value regression** — supervised. What is a player worth over the window?

## Running it

```bash
# Parse one hand-history file into 3 tables (prints the big-blind VPIP correctness check)
python3.13 src/parse_phh.py <file.phhs>

# Read the guides (the plan cross-links to both companions)
cd docs && python3.13 -m http.server 8793
# → http://localhost:8793/poker-project-plan.html
```

`parse_phh.py` needs **Python 3.11+** and no third-party packages — it uses stdlib `tomllib`.

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

<!-- TODO: fill in before submission. Max 6 members. Contribution split is a graded
     appendix requirement of the report — record it as you go, not the night before. -->

| Name | Roll no. | Role |
|---|---|---|
| _TBD_ | _TBD_ | _TBD_ |

## Assessment

Marks are also awarded for **commit history across the trimester**. Commit small and often —
one commit per real step (parser fix, Silver build, feature set, report draft), not one dump
at the end.
