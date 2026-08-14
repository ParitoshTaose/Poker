# K-Means segmentation — the unsupervised layer, and what it does to the story

**Run:** `.venv/bin/python src/segments.py` · 2.0 min · seed 42 · 90,469 players.
Raw metrics: [`segments-results.json`](segments-results.json). Feeds on
[`fork-a-results.md`](fork-a-results.md)'s risk scores. Output table: `data/gold/player_segments`.

Fork A answers *who* is drifting. It cannot answer *what kind of player* is drifting — and that
is the entire thesis of the project, because an operator earns rake only while recreational
players keep sitting down. This is the layer that connects them.

---

## The finding, in one table

Clustered on prior-window behaviour + intensity + multi-tabling. Five venues (iPoker records no
table identity), 77,268 players, k=4. Pooled lapse rate **70.5%**.

| Segment | Share | Hands | Tables | VPIP | PFR | bb/100 | **Lapse rate** | Share of all lapsers |
|---|---|---|---|---|---|---|---|---|
| **The Grinder** | 3.7% (2,831) | 8,799 | **11.5** | 0.18 | 0.12 | **−3.3** | **61.3%** | 3.2% |
| **The Regular** | 35.4% (27,376) | 884 | 3.0 | 0.22 | 0.11 | −7.2 | 61.6% | 30.9% |
| **The Gambler** | 19.8% (15,262) | 331 | 1.3 | 0.59 | 0.20 | **−31.8** | **74.6%** | 20.9% |
| **The Recreational** | 41.2% (31,799) | 352 | 1.4 | 0.39 | 0.11 | −14.3 | **77.1%** | **45.0%** |

*Names are interpretation, not output — the algorithm produces numbered clusters. Every figure
in the table is measured.*

**The two segments that lose money fastest are the two that leave fastest.** The Gambler bleeds
−31.8 bb/100 and lapses at 74.6%; the Recreational loses −14.3 and lapses at 77.1%. The Grinder,
who loses least (−3.3, essentially just the rake), is the *stickiest* player on the platform.

Together those two loose, money-losing segments are **60.9% of players and 65.9% of everyone who
goes quiet**. That is the ecosystem thesis stated as a measurement rather than an assumption: the
churn is concentrated precisely in the population that funds the rake, and the professionals who
extract from them are the ones who stay.

The operator's problem is therefore not "we lose players." It is **"we lose the players we earn
from, and keep the ones we don't."**

## Why it was clustered on the prior window

`gold/player_features` would be the obvious input and would quietly poison this table. Its rate
columns average over a player's whole history *including the week being predicted* — a lapsed
player contributes nothing to their own final week, an active one does — so segments built that
way partly encode the answer before the cross-tab is drawn.

Clustering the `p_*` columns from `gold/player_lapse` instead means **every segment is knowable on
the cutoff day**, which is also the only version an operator could act on. It also appears to
cluster better — silhouette **0.317 at k=4** against the bake-off's **0.200** on the same eleven
columns aggregated over the lifetime — though the populations differ (90,469 vs 106,977), so treat
that as suggestive rather than a controlled comparison.

## Two views, because two venues do not record everything

**STYLE** (all six venues, 90,469 players) — how they play, deliberately *excluding* volume, so
that "this segment lapses more" is a statement about behaviour rather than an arithmetic
restatement of "this segment plays less."

It recovers the same broad split without being told about volume. The two clusters that *turn
out* to be high-volume (1,872 and 1,405 prior hands) lapse at **46.9% and 65.2%**; the two
low-volume ones (320 and 350 hands) lapse at **74.7% and 79.4%**. Note the split is not monotone
in looseness — the loosest cluster (VPIP 0.61) lapses slightly *less* than the middling one
(VPIP 0.42). Style is a proxy for engagement here, not a direct cause of it.

**ECOSYSTEM** (five venues, 77,268 players — 85.4%) — style plus intensity and multi-tabling.
iPoker writes the literal string `HandHQ` into all 6.0M of its table fields, so it is dropped
rather than imputed at `max_tables = 1`, which would file all 13,201 of its players as
recreational single-tablers.

## The archetype check — two confirmed, two wrong, and the wrongness is the interesting part

`poker-project-plan.html` hypothesised four archetypes from community rules of thumb (explicitly
*not* findings). The clusters were matched against them by VPIP, PFR and table count:

| Hypothesis | Found | Verdict |
|---|---|---|
| Grinder — VPIP 22 / PFR 18 / 12 tables | VPIP 18 / PFR 12 / **11.5 tables** | **Confirmed.** Closest match of the four. |
| Recreational — VPIP 45 / PFR 8 / 1–2 tables | VPIP 39 / PFR 11 / 1.4 tables | **Confirmed.** |
| Nit — VPIP **12** / PFR 9 / 3–5 tables | nearest cluster is VPIP **22** | **Not found.** No extreme-tight cluster exists. |
| Gambler — VPIP 60 / PFR **35** | VPIP 59 / PFR **20** | **Half right.** The loose cluster is loose-**passive**, not loose-aggressive. |

The loose cluster calling far more than it raises is a real behavioural finding, not a rounding
error: it voluntarily enters 59% of hands but raises pre-flop in only 20% — so it *calls* on
roughly two thirds of the hands it plays. That is the profile of someone paying to see cards
rather than someone attacking pots, and it is also the segment losing money fastest (−31.8
bb/100), which is exactly what that profile predicts.

**And the grinder only appears when volume and table count are in the feature set.** In the STYLE
view — betting behaviour alone — no grinder cluster forms at all; two of the four clusters both map
onto "Recreational." Style does not distinguish a professional from a customer. **Multi-tabling
does.** That is a direct argument for the showpiece feature, and for why iPoker's missing table
identity was worth nulling rather than filling in.

## Two configurations, compared

The roadmap asks for at least two clustering configurations. K-Means was compared against
**bisecting k-means** at the same k:

| View | K-Means silhouette | Bisecting silhouette | Players placed together |
|---|---|---|---|
| Style, six venues | 0.317 | **0.356** | 65.7% |
| Ecosystem, five venues | 0.228 | **0.319** | 67.4% |

**Bisecting k-means wins on silhouette in both views, and the two algorithms agree on only about
two thirds of players.** That instability is itself the honest result, and it agrees with what the
silhouette scores already say — see below.

## Where this is weak, and it must be said

- **These are gradients, not natural groups.** Silhouette peaks at **k=2** (0.439) in the style
  view and k=3 in the ecosystem view; k=4 costs 0.12–0.15 of silhouette. k=4 was chosen for
  business readability and to make the archetype comparison one-to-one — a judgement call,
  recorded on purpose, not a silhouette optimum. Two algorithms disagreeing on a third of players
  is what weak separation looks like from the inside. **Do not present these as four discovered
  species of poker player.** They are four useful cuts through a continuum.
- **The `share_losing` column in the raw output is a small-sample artifact** and is deliberately
  not quoted above. Median money-bearing hands per player is only ~92, so an individual's bb/100
  is extremely noisy; cluster *means* over thousands of players are reliable, the per-player sign
  is not.
- **Fork A's risk scores rank correctly but are not calibrated probabilities.** Mean predicted
  risk per segment (0.52 · 0.53 · 0.64 · 0.67) orders the four segments exactly as the observed
  lapse rates do (0.616 · 0.613 · 0.746 · 0.771), but sits about 9 points low throughout, because
  balanced class weighting pulls probabilities toward 0.5. **Anything that multiplies a rupee
  figure by these scores must recalibrate first** — this matters directly for the money-weighted
  headline that comes next.
- Money columns are five-venue only, and bb/100 is the *player's* profit, not the operator's rake.

## Where the risk actually sits — lapse rate by segment × venue (style view)

| Venue | Seg 0 | Seg 1 | Seg 2 | Seg 3 |
|---|---|---|---|---|
| Absolute Poker | 35.7 | 39.4 | 53.9 | 52.9 |
| Full Tilt | 44.0 | 30.9 | 65.5 | 54.6 |
| Ongame | 43.3 | 44.6 | 58.7 | 62.0 |
| PartyPoker | 64.0 | 72.2 | 79.8 | 77.7 |
| PokerStars | 46.1 | 75.0 | 80.7 | **87.2** |
| iPoker | 45.2 | 52.8 | 64.6 | 63.7 |

The spread runs from 30.9% to 87.2%. **A global view mixes incomparable populations** — which is
why the dashboard's venue filter is a blocker in the delivery gate, not a nicety.

## What this unlocks

The segment table is what turns Fork A's ranked list into a recommendation. "Contact 2,260
accounts" is analytics; **"the recreational segment is 41% of players, 45% of all churn, and the
only segment the rake actually comes from"** is a management decision with a budget attached.

The missing piece is the money weight: attaching estimated rake contribution per player would let
the headline read *"₹X of monthly rake sits with N at-risk recreational players"*, which is
Gate A8 of the delivery gate and the number the dashboard should open on.
