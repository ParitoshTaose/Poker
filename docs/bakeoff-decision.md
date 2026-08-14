# Step 5 — Fork bake-off: the decision note

**Status: DRAFT — awaiting team sign-off on the winner.** Everything below the scorecard is
measured. The scorecard's two judgement columns are proposed, not settled.

> **Update, 14 Aug 2026 — the condition below has been met.** This note recommended Fork A
> *conditional on* extending `features.py` with prior-window-only rate columns before Step 8.
> `src/features_lapse.py` now does exactly that and `src/fork_a.py` rebuilds the models in
> PySpark MLlib. The leakage ablation reversed: the honest feature set now out-predicts the leaky
> one (ROC-AUC 0.856 vs 0.819). One correction to this note: `hands_prior > 0` was not a strict
> enough population filter — Gold's *lifetime* ≥100-hand rule admits 8,249 players who clear 100
> only by playing in the label week, and every one of them is labelled "active". Prevalence is
> therefore **68.2%, not 62.5%**. Full results in **[`fork-a-results.md`](fork-a-results.md)**.

**Run:** `.venv/bin/python src/bakeoff.py` · 41 s · seed 42 · full Gold tables
(106,977 players / 21,556,288 hands; Fork C on a 600,000-hand sample).
**Raw metrics:** [`bakeoff-results.json`](bakeoff-results.json).
**Specs these prototypes implement:** [`fork-specs.html`](fork-specs.html).

Five candidates, each scored against **its own stated dumb baseline** — a score with no
baseline beside it is not evidence. Two algorithms per supervised candidate (a linear model
and a gradient-boosted tree), which also de-risks Step 8's "at least two approaches" rule.

---

## The scorecard

Signal and coverage are measured. Business story and rubric fit are judgement calls, marked as such.

| Candidate | Signal | Coverage | Business story | Rubric fit | **Total** |
|---|---|---|---|---|---|
| **Fork A · lapse classification** | 3 | 5 | 5 | 4 | **17** |
| **Fork B · player value (money)** | 4 | 3 | 4 | 4 | **15** |
| K-Means segmentation | 2 | 5 | 4 | 2 | 13 |
| Fork B · player value (volume) | 2 | 5 | 3 | 3 | 13 |
| Fork C · hand economics | 1 | 5 | 3 | 3 | 12 |

**The measured column, in full:**

| Candidate | Baseline | Best model | Result |
|---|---|---|---|
| Fork A · lapse | majority class, PR-AUC 0.625 | HistGBM | **PR-AUC 0.927**, ROC-AUC 0.877, F1 0.807 |
| Fork B · money | per-venue mean, RMSE 82.64 | HistGBM | **RMSE 72.82**, skill **+0.224** |
| Fork B · volume | per-venue mean, RMSE 2.464 (log) | HistGBM | RMSE 1.866, skill +0.427 |
| Fork C · pot_bb | per-stake mean, RMSE 36.92 | Ridge | RMSE 36.84, skill **+0.004** |
| K-Means | — | k=2 | silhouette 0.279 (0.200 at k=4) |

*Skill = 1 − SSE(model)/SSE(baseline). Plain R² measures against the global mean, which
flatters any model whose baseline is already per-group.*

---

## The finding that decided it: a leakage ablation

Every rate column in Gold — `vpip`, `pfr`, `fold_rate`, `wtsd`, `aggression_factor` — is
averaged over a player's **entire** recorded history, final week included. So a model
predicting the final week from those rates is reading its own answer. `fork-specs.html`
flagged this on paper; this run measured it.

The same GBM, three feature sets:

| Feature set | Fork A (PR-AUC) | Fork B volume (R²) |
|---|---|---|
| `hands_prior` + `tenure_days` only — honestly prior-window | 0.798 | +0.284 |
| lifetime rates only — all include the label window | 0.886 | +0.320 |
| both — the headline number | **0.927** | **+0.470** |

**The lifetime rates alone out-predict the legitimately-prior features.** That is the
signature of leakage, not of behaviour. Both forward-looking forks are therefore quoting an
optimistic ceiling. Closing it means extending `features.py` to compute rate columns
restricted to `day <= cutoff` — one extra Spark pass, roughly the shape of work already done.

**Fork B's money variant is the only supervised candidate this does not touch.** It describes
realised value over a player's whole history and makes no forward-looking claim, so lifetime
aggregation is the correct treatment rather than a defect. Its +0.224 needs no asterisk.

---

## Why each candidate placed where it did

**Fork A · lapse — 17, the measured winner.** Best combination of coverage (98,718 players,
all six venues) and operator action (a retention list is something a team can literally work
through on a Monday). Its honest floor — 0.798 PR-AUC on prior-window features alone — still
beats the 0.625 baseline decisively, so the signal survives even if every leakage-suspect
feature is deleted. Marked down to 3 on signal purely because the headline needs the
`features.py` extension before it can be defended in a viva.

**Fork B · money — 15, close second, and the cleanest methodology.** Real signal, and
positive on **every** venue independently (PS +0.169 · PTY +0.254 · ABS +0.128 · FTP +0.075 ·
ONG +0.059), which is much harder to explain away than a single pooled number. It loses on
coverage: iPoker reconciles no money at all, and `money_hands >= 50` drops 41,245 of 106,977
players — 38.6% of Gold, leaving a five-venue question. It also has a framing gap: `bb_per_100`
is **the player's own profit**, not the rake the operator earns from them, so "player value"
needs saying carefully or a probing examiner will find the seam.

**K-Means — 13, but it ships regardless.** Silhouette 0.279 at k=2 and 0.200 at k=4 is weak
separation; the clusters are gradients in a continuum, not natural groups. The k=4 centroids
are still readable and nameable (a 36% tight-and-passive block at VPIP 0.22, a 16% loose block
at VPIP 0.61, a 17% multi-stake block averaging 2.8 stakes). Scored low on rubric fit because
unsupervised work alone doesn't exercise the classification or regression marks — it is the
base layer under whichever fork wins, not a competitor to it.

**Fork B · volume — 13.** Highest raw skill of any candidate (+0.427), and the least
trustworthy: the ablation above hits it hardest, and the target is zero for **62.5%** of
players, so it is really a classification problem wearing a regression's clothes. Loses to its
own money sibling, which asks a similar question without the asterisk.

**Fork C · hand economics — 12, eliminated on evidence.** Skill **+0.004**. Ridge and GBM both
land within 0.2% of a per-stake mean, and the per-stake breakdown shows no stake where either
model helps. Pot size is driven by the cards dealt and the decisions players make, none of
which is knowable before the deal — so day, hour, seat count and stake carry essentially no
information about it. This is a genuine negative result and worth one honest line in the
report: **the cleanest fork methodologically turned out to have nothing to predict.** Finding
that in an afternoon is exactly what the bake-off was for.

---

## Recommendation

**Fork A as the spine, with K-Means as the unsupervised base layer** — it wins the scorecard,
it keeps all six venues, and its business story needs no reframing. **Conditional on** extending
`features.py` with prior-window-only rate columns before Step 8; without that the headline
metric is not defensible.

**Choose Fork B (money) instead if** the team would rather not spend that extension, or wants
the money story specifically. It is a legitimate winner on the "cleanest methodology" axis and
needs no new Gold columns — the cost is saying "five venues, 61% of players" out loud, every time.

Either way, **both models already exist in `src/bakeoff.py`**, so the loser becomes the
report's *alternatives considered* paragraph with real numbers behind it rather than a
hand-wave.

---

## Two defects this run surfaced

**1. iPoker records no starting stacks — and it arrives as `inf`, not as null.** The raw
`.phhs` files contain the literal `starting_stacks = [inf, inf]`. `inf` is a **valid TOML float**,
so `tomllib` parses it silently, nothing upstream ever raised, and it reached Gold intact:
`avg_stack_bb` is `+inf` on all 15,549 iPoker players and all 5,996,194 iPoker hands
(`min_stack_bb` too). It detonates `StandardScaler`, Ridge and any distance metric on contact —
which is how it was found. This is the third iPoker-shaped hole, after no table identity and no
reconciling winnings: **iPoker is behaviourally complete and economically blind.**
`src/bakeoff.py` patches the symptom; the fix belongs in `features.py`, next to the
`tables_recorded` flag it exactly parallels — null the column and carry a `stacks_recorded`
boolean. *Never impute it:* a median stack on a venue that recorded none is the same class of
lie as `max_tables = 1`.

**2. `hands_per_day` is an algebraic leak, not a soft one.** It is `hands_played / days_active`,
and `hands_played = hands_prior + hands_final7`. With `hands_prior` also in the feature set a
model recovers the label outright: `hands_final7 = hands_per_day × days_active − hands_prior`.
The first pass of this script scored Fork A at 0.960 PR-AUC on exactly that. It is now in
`FORWARD_LEAKS` alongside `money_coverage`, which leaks the same denominator. It remains legal
for the money variant, which is not forward-looking.

---

## Decisions recorded here on purpose

- **`money_hands >= 50`** for Fork B money. The sweep: `>=1` keeps 88,655 players with 6.4%
  beyond ±200 bb/100; `>=30` keeps 81,853 at 4.4%; **`>=50` keeps 65,732 at 2.8%**; `>=100`
  keeps 41,654 at 1.6%. 50 is the knee — it more than halves the small-sample artifact while
  keeping three quarters of the money-bearing players. A judgement call, not a verified optimum.
- **Fork A class balance: 62.5% lapsed** (61,702 of 98,718 players with `hands_prior > 0`).
  `fork-specs.html` listed this as unmeasured; it is now measured, not assumed.
- **Fork C split by time** (last 4 days per venue), not at random — a shuffle would put the
  same table-session on both sides of the split.
- **Filtered to `hands_prior > 0`** for both forward-looking forks. A player who first appears
  inside the final week is new, not lapsed; including them labels arrivals as departures.
- **iPoker excluded from Fork B money outright** rather than imputed.
