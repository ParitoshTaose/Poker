# Fork A — short-horizon lapse classification: built, trained, measured

**Status: the bake-off's condition is closed.** `docs/bakeoff-decision.md` made Fork A the
recommended spine *conditional on* extending the feature build with prior-window-only rate
columns, because without them "the headline metric is not defensible." Those columns now exist,
the models are rebuilt in PySpark MLlib, and the leakage ablation has been re-run on the new
features. Everything below is measured.

**Runs** (Apple M4, 8 cores, local Spark, seed 42):

| Script | What it does | Wall clock |
|---|---|---|
| `.venv/bin/python src/features_lapse.py` | Silver → `data/gold/player_lapse`, 90,469 × 61 cols | 3.8 min |
| `.venv/bin/python src/fork_a.py` | 3 algorithms × 2 feature sets + ablation → `docs/fork-a-results.json` | 5.0 min |

Raw metrics: [`fork-a-results.json`](fork-a-results.json). Spec: [`fork-specs.html`](fork-specs.html) §03.
Prior decision: [`bakeoff-decision.md`](bakeoff-decision.md).

**On reproducibility:** the train/test split is a hash of the player key, so it is identical on
every machine, and logistic regression and random forest reproduce bit-for-bit. **GBT reproduces
only to about five decimal places** (0.856638 vs 0.856641 across two runs of identical code) —
Spark sums its tree histograms in task-completion order, which varies. Nothing here is quoted
past three decimals, so it does not move a single reported figure; it is recorded so that a small
wobble is not mistaken for a bug later.

---

## The headline

Predicting, for every player active at a venue's cutoff, whether they play **zero hands in the
next seven days**. All six venues, 90,469 players, 22,596 held out.

| | ROC-AUC | PR-AUC | best F1 | precision @ 10% contact budget |
|---|---|---|---|---|
| Baseline · flag everyone | 0.500 | 0.683 | 0.811 | — |
| Baseline · **sort by days-since-last-seen** | 0.800 | 0.864 | 0.856 | 0.885 |
| MLlib · logistic regression | 0.837 | 0.901 | 0.857 | 0.944 |
| MLlib · **gradient-boosted trees** | **0.857** | **0.917** | **0.866** | **0.970** |
| MLlib · random forest | 0.857 | 0.918 | 0.865 | 0.966 |

GBT and random forest are indistinguishable from each other (0.8566 vs 0.8569 ROC-AUC — noise).
Logistic regression gives up about 0.02 of ROC-AUC and returns readable coefficients for it.
**That trade is the comparison the rubric asks for**, and it is small enough that the linear model
is a legitimate choice if a slide has to explain *why* a player was flagged.

---

## 1. The population changed — and that is the first finding

`gold/player_features` keeps players with **≥ 100 hands over their whole history**. Reusing that
rule for Fork A leaks: a player with 30 prior-window hands only clears 100 by playing 70+ hands
*in the week being predicted* — that is, by being definitely-not-lapsed. The filter would then be
partly built out of the answer.

Selecting on **≥ 100 prior-window hands** instead — a rule that could actually be applied on the
cutoff day — is the fix. The arithmetic is decisive:

| Population | Players | Lapsed | Prevalence |
|---|---|---|---|
| Gold's lifetime ≥ 100 rule (what the bake-off used) | 98,718 | 61,702 | 62.5% |
| Prior-window ≥ 100 rule (this build) | **90,469** | **61,702** | **68.2%** |
| Difference | 8,249 | **0** | — |

**The lifetime filter admits 8,249 extra players and every single one of them is labelled
"active."** Not approximately — exactly. The lapsed count is identical in both populations, so
the entire difference is non-lapsers who were let in by a filter that had already seen the label.
That is a textbook selection leak, measured rather than argued.

Consequence: **the prevalence a metric is read against moves from 62.5% to 68.2%**, so PR-AUC
figures in this document are *not* comparable to the bake-off's. ROC-AUC is prevalence-independent
and is the safer number to quote across the two.

Per-venue prevalence varies enormously, and the model is scored on each separately for that reason:

| Venue | Players | Lapsed | Mean recency (days) | Prior window |
|---|---|---|---|---|
| PokerStars | 48,561 | 77.9% | 5.47 | days 1–13 |
| iPoker | 13,201 | 54.6% | 5.03 | days 1–19 |
| PartyPoker | 12,976 | 73.8% | 6.04 | days 1–19 |
| Absolute | 6,737 | 45.8% | 3.22 | days 1–13 |
| Full Tilt | 5,272 | 40.4% | 3.84 | days 1–16 |
| Ongame | 3,722 | 49.8% | 4.62 | days 1–17 |

PokerStars and Absolute stop recording on day 20, so they get a 13-day prior window against
iPoker's and PartyPoker's 19. `site` is a one-hot feature and `p_prior_span_days` is carried, so
the model can calibrate for the difference instead of silently absorbing it.

## 2. The label, and what it is not

`lapsed = 1` if the player logged zero hands in the seven days ending on **their own venue's last
recorded day** (ABS/PS 20 · FTP 23 · ONG 24 · IPN/PTY 26). Measured per venue, never against a
global calendar — a global window would mark every ABS and PS player as lapsed purely because
their data stops, which is 44% of the dataset mislabelled by construction.

**This is short-horizon lapse, not churn.** Twenty-six days cannot show that anyone left for
good. A week of silence is an early warning. Say it every time.

## 3. The features are the actual work

Every modelling column is recomputed from rows with `day <= cutoff` — thirty of them, prefixed
`p_`. Beyond simply re-filtering the old rates, the build adds what a retention team would
actually reach for and what the old table could not express:

- `p_recency_days` — days between the last prior-window hand and the cutoff
- `p_hands_w1` / `p_hands_w2` / `p_late_share` — is their volume decaying into the cutoff, or ramping?
- `p_active_day_share` — how many of their tenure days they actually showed up on
- `p_active_minutes`, `p_hands_per_minute` — time at the table, not just hands dealt

Two feature sets, because two of the six venues do not record everything:

- **CORE** — 30 prior-window features + 2 missing-value indicators + 5 venue dummies (PokerStars,
  the largest venue, is the reference level). All six venues, 90,469 players.
- **EXTENDED** — CORE + stack depth, multi-tabling and realised money. iPoker records none of the
  three, so this is a **five-venue model covering 77,268 players (85.4%)**. Reported separately
  rather than imputed: a median stack on a venue that recorded none is a number nobody measured.

EXTENDED buys about **+0.007 ROC-AUC** (0.8631 vs 0.8566 for GBT) for the loss of 14.6% of the
players. On the evidence, **CORE is the better spine** — the extra features are not worth
excluding a whole venue for.

Two independent checks that the table is correct: `p_hands + hands_final7 == hands_played`
against the separately-built `gold/player_features` on **all 90,469 rows, zero mismatches**; and
prior-window pooled VPIP 27.2% / PFR 13.7% / big-blind VPIP **30.7%** against the parser's
lifetime 27.4 / 14.1 / 31.3 — close, and nowhere near the ~100% that would mean the
check-versus-call bug had returned.

## 4. Three baselines, because a score with nothing beside it is not evidence

**Flag everyone.** Recall 1.0 by construction, precision = prevalence, F1 0.811. This is what
"accuracy" flatters and why accuracy is not used here.

**Sort by days-since-last-seen, descending.** No model, no training, no data engineering — a
retention manager can do this in Excel in four minutes, and it scores **ROC-AUC 0.800**. This is
the number MLlib has to beat to justify its place in the architecture.

**The same idea as a deployable rule.** "Quiet for ≥ 1 day ⇒ flag" scores **F1 0.856** — and at
the default 0.5 threshold, *every model loses to it* (GBT 0.829). The models only win once their
operating point is chosen deliberately: at threshold 0.25, GBT reaches **F1 0.866**.

**That is worth stating plainly in the report.** A default threshold is an arbitrary inheritance
from a balanced world; on a 68/32 problem it is a decision, and leaving it at 0.5 would have made
a working model look worse than a spreadsheet sort.

## 5. The leakage re-check — the reason this build exists

Same estimator, same population, same split, three feature sets:

| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| The old floor — `hands_prior` + `tenure_days` only | 0.690 | 0.805 |
| **Lifetime rates only** — the leaky set, includes the label window | 0.819 | 0.911 |
| **Prior-window only** — this build, structurally cannot see the label week | **0.856** | **0.917** |

In the bake-off the ordering was the other way round: lifetime rates (0.886 PR-AUC) *beat* the
honestly-prior features (0.798), which is the signature of leakage rather than of behaviour. It
has now reversed. **The honest feature set out-predicts the leaky one**, so the model no longer
depends on any column that saw its own answer — and it got *better*, not worse, in the process.

Two cautions, so this is not oversold. The honest claim rests on **construction, not on this
table**: `p_*` columns are computed from `day <= cutoff` and therefore cannot contain label-week
information, whatever the scores had said. And the two rows are not a controlled comparison of
equal capacity — 30 features against 11. What the table rules out is the specific failure the
bake-off found, namely a model that needs the leaky columns to perform.

## 6. From a metric to a decision

A retention team has capacity, not an appetite for "contact all 15,423." At a **10% contact
budget** on the 22,596 held-out players — 2,260 calls:

| | True lapsers among those contacted | Precision | Lift vs. random |
|---|---|---|---|
| Random selection | 1,543 | 0.683 | 1.00 |
| Sort by recency (free) | 2,001 | 0.885 | 1.30 |
| **GBT** | **2,192** | **0.970** | **1.42** |

Against the free baseline the model finds **191 more genuinely at-risk players per 2,260
contacts** — equivalently, 191 fewer bonus offers posted to people who were never leaving. That
is the sentence that turns a metrics table into a budget line, and it is the form the Model
Evaluation marks are looking for.

Recall at that budget is only 0.142, and that is honest: when 68% of the population goes quiet in
any given week, no 10% list can cover them. **The interesting question is therefore not "who is
lapsing" but "which lapsers are worth a call"** — which is precisely what Fork B's value model
was built to answer, and the strongest argument for running the loser of the bake-off as a second
stage rather than shelving it.

`data/gold/lapse_scores` holds all 90,469 players with a risk score, which is the deliverable a
retention team actually receives.

## 7. Where it is weak

- **One cutoff, one week.** The split is by player, so no player appears in both folds — but the
  model is never tested on a *different* time period. A true backtest (train at
  `site_last_day − 14`, test at `site_last_day − 7`) is the obvious next robustness check and has
  not been run. Do not claim temporal generalisation.
- **Recency does most of the work.** It is the top feature in every model (GBT importance 0.249,
  standardised logistic coefficient +0.642) and the next feature is worth a third as much. The
  behavioural columns — the whole ecosystem thesis — add real but secondary signal. An examiner
  will ask; the ablation above is the answer, and it is a decent one, but do not oversell VPIP.
- **The model is weakest exactly where prevalence is lowest** — Ongame ROC 0.769, Absolute 0.780
  against PokerStars' 0.861. Small test sets (969 and 1,679 players) explain part of it, but not all.
- **The data is from July 2009.** Argue that the method transfers; never hide the date.

## 8. What is still not done

1. **A time-shifted backtest** — the robustness check named above.
2. **K-Means segmentation as the unsupervised layer.** The roadmap ships it regardless of which
   fork wins; it is not in this build. Crossing segment against lapse risk ("segment 3 is 78%
   lapsed") is the bridge from this model to the business narrative.
3. **The Databricks notebook.** These are local Spark scripts; Steps 7–8 of the roadmap want the
   graded, executable notebook. The logic is now written and measured, so that step is re-hosting,
   not re-deriving.
4. **`features.py` still writes `+inf` into `avg_stack_bb`** for all 15,549 iPoker players. The
   guard exists in `features_lapse.py` (`finite()`, plus a `stacks_recorded` flag); porting it back
   into `features.py` and rebuilding Gold has not been done.
