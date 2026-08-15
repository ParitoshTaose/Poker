# Rake at risk — the money-weighted headline

**Status: built, measured, and the two things it depends on were both broken when this
started.** Fork A ranks players by the chance they go quiet; segmentation says what kind of
player each one is. Neither knows what a player is *worth*. This is the layer that attaches
money, and it is the last piece of analysis in the project — everything after it is writing
and packaging.

**Run** (Apple M4, 8 cores, local Spark, seed 42):

| Script | What it does | Wall clock |
|---|---|---|
| `.venv/bin/python src/rake_at_risk.py --rebuild-rake` | recalibrate · price every hand · attribute · headline | **4.9 min** |
| `.venv/bin/python src/rake_at_risk.py` | the same, reusing `data/_work/player_rake` | 2.1 min |

Reads `gold/player_lapse`, `gold/lapse_scores`, `gold/player_segments`, `gold/hand_features`
and the 116.6 M-row seat join in `data/_work/hp_enriched`. Writes **`data/gold/rake_at_risk`**
(77,268 players × 23 columns — the table the dashboard and the deck read),
`data/_work/player_rake`, and [`rake-at-risk-results.json`](rake-at-risk-results.json).

**On reproducibility:** the script was run **three times end to end**, and every figure in this
document came back identical — the only field that changed between runs was the wall clock. The
split is a hash of the player key, the estimator is a group mean, and the GBT wobble recorded for
Fork A (which reproduces only to about five decimals) never surfaced at the precision quoted here.
Every number below was also machine-checked against
[`rake-at-risk-results.json`](rake-at-risk-results.json) rather than transcribed by eye.

---

## The headline

> **$604,163 of next week's rake sits at risk across 77,268 players on five venues — 30% of it
> in the recreational segment — and contacting the top 10% of the list by expected loss (7,727
> players) puts $421,064 of it in reach.**

The small print travels with the sentence, always:

- Weekly rake across these players is **$2,114,626**, of which **34.4% is measured** and the
  rest estimated from what comparable hands actually paid.
- **58,836 players (76.1%)** are more likely than not to play zero hands next week, and
  **$498,156** of weekly rake sits with them.
- **iPoker is excluded entirely** — its rake reconciles on none of its 6.0 M hands.
- July 2009, **US dollars**, 25NL–1000NL. No currency conversion (a 2009 INR rate would be
  invented, not measured) and no inflation adjustment.

Everything above is one multiplication — P(lapse) × weekly rake — and **both factors had a
defect that had to be closed before multiplying them meant anything.**

---

## 1. The probability was ranked correctly and calibrated wrongly

Fork A's GBT was fitted with balanced class weights. That drags predicted probabilities toward
0.5, which ROC-AUC and PR-AUC cannot see — they only look at order — and which a dollar figure
notices immediately, because every dollar gets multiplied by a number that is systematically
too small.

The fix tried first was simply to **drop the weights**: at 68/32 there was never a real
imbalance to correct. Weighting exists for problems where the positive class is 1% of the data;
here the majority class *is* the event of interest. Same features, same hash split, same seed,
same hyper-parameters — only `weightCol` removed.

| Test fold, n = 22,596 | mean predicted | observed | **worst decile gap** | ECE | Brier | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Fork A's shipped GBT (weighted) | 0.576 | 0.683 | **0.200** | 0.106 | 0.1516 | 0.8566 | 0.9173 |
| Same GBT, no class weights | **0.680** | 0.683 | **0.026** | 0.011 | 0.1365 | **0.8575** | **0.9187** |

The reliability curve is the evidence, and it is Model Evaluation material in its own right —
ten equal-sized deciles of predicted probability, each compared against what actually happened:

| decile | predicted (before) | observed | gap | | predicted (after) | observed | gap |
|---|---|---|---|---|---|---|---|
| 1 | 0.056 | 0.106 | −0.049 | | 0.100 | 0.107 | −0.007 |
| 2 | 0.152 | 0.304 | −0.152 | | 0.274 | 0.300 | −0.026 |
| 3 | 0.304 | 0.504 | **−0.200** | | 0.474 | 0.499 | −0.025 |
| 4 | 0.448 | 0.621 | −0.173 | | 0.636 | 0.624 | +0.011 |
| 5 | 0.579 | 0.742 | −0.162 | | 0.750 | 0.743 | +0.007 |
| 6 | 0.698 | 0.808 | −0.110 | | 0.823 | 0.814 | +0.009 |
| 7 | 0.807 | 0.884 | −0.077 | | 0.892 | 0.881 | +0.010 |
| 8 | 0.879 | 0.942 | −0.063 | | 0.938 | 0.942 | −0.004 |
| 9 | 0.909 | 0.946 | −0.037 | | 0.953 | 0.949 | +0.004 |
| 10 | 0.930 | 0.970 | −0.040 | | 0.963 | 0.967 | −0.004 |

**The worst decile moves from 20.0 points off to 2.6 points off, and the ranking does not
move.** ROC-AUC goes from 0.8566 to 0.8575 — inside GBT's own reproducibility wobble. That is
exactly the result to want: calibration changed, discrimination did not.

A tolerance of **0.03 on the worst decile was declared in the source before the run**, with
isotonic regression (fitted on the *train* fold only, never on test) as the fallback. At 0.026
the refit cleared it, so **no calibration layer was fitted** — the probabilities are the
model's own.

Calibration was then re-measured per fold, because 75% of the players in the headline are ones
the model trained on: **train fold mean predicted 0.682 against observed 0.682** (worst decile
0.035), **test fold 0.680 against 0.683** (worst decile 0.026). The means agree on both folds,
which is what matters for a sum — the headline does not inherit an in-sample optimism — and the
train fold is not the *better*-calibrated of the two, which it would be if the model were
memorising.

One consequence for the report: **Fork A's tuned threshold of 0.25 does not transfer.** It
belonged to the weighted scores. The recalibrated model's best-F1 operating point is
**0.40 → F1 0.867** (precision 0.808, recall 0.934) — a shade better than Fork A's tuned 0.866,
at a threshold that now means what it says.

## 2. Rake is derived, and the hands where it can be derived are not a random sample

This is the trap the script was written to walk into and did not, so it is recorded in full.

`parse_phh.py` derives rake as *pot − paid-out winnings* and accepts the result only when the
residual is at most 15% of the pot. On a venue whose exports are incomplete, **that test is
easier to pass on a big pot** — a fixed shortfall is 40% of a small pot and 3% of a large one.
So the hands that survive are the big ones:

| Venue | measured hands | median pot | mean pot | saw flop | | unmeasured hands | median pot | mean pot | saw flop |
|---|---|---|---|---|---|---|---|---|---|
| PokerStars | 2,231,818 | 2.50 bb | 6.84 | 49.6% | | 6,061,979 | 3.00 bb | 14.95 | 54.8% |
| PartyPoker | 341,781 | **23.40 bb** | 41.28 | **99.5%** | | 2,750,904 | **2.50 bb** | 9.40 | **39.0%** |
| Absolute | 573,513 | 3.00 bb | 7.82 | 50.7% | | 684,465 | 6.50 bb | 20.88 | 63.7% |
| Full Tilt | 344,059 | 2.50 bb | 8.32 | 46.3% | | 939,736 | 2.50 bb | 14.39 | 46.2% |
| Ongame | 1,579,938 | 2.50 bb | 10.74 | 44.4% | | 51,901 | 56.50 bb | 117.04 | 87.4% |

**The obvious method — take each player's observed rake per 100 hands and scale it to all their
hands — is therefore wrong**, and wrong in a direction that flatters the answer. It charges
every hand at the rate of the pots big enough to reconcile, and it ignores that most hands never
see a flop and pay nothing at all. ("No flop, no drop", measured here: pre-flop-only hands rake
exactly $0.00 on 100.000% of PokerStars, Ongame and PartyPoker hands, 99.999% of Full Tilt's and
99.6% of Absolute's.) That method is still computed, and it comes out **25% high** — see §9.

## 3. A plausibility screen, and the measurement that justifies it

The 15% bound in the parser was deliberately loose, chosen when there was nothing better to test
against. There is now something better.

Measured across all six venues, the **median rake is 4.3%–5.0% of the pot on every one of
them** — which is exactly the commercial rake of the period, and a strong sign the derivation is
sound. And on **PokerStars, whose exports reconcile completely, not one of 1,107,219 flop hands
has a rake above 5.0% of the pot.** Full Tilt (0.13% above 6%) and Ongame (0.70%) agree.

Two venues do not:

| Venue | median share | p90 share | flop hands above 6% | at the parser's 15% ceiling |
|---|---|---|---|---|
| PokerStars | 4.26% | **5.00%** | **0.00%** | 0.00% |
| Full Tilt | 4.55% | 5.00% | 0.13% | 0.01% |
| Ongame | 4.89% | 5.00% | 0.70% | 0.07% |
| Absolute | 5.00% | 11.33% | **41.5%** | 3.13% |
| PartyPoker | 4.94% | 13.33% | **21.9%** | 7.98% |

*(The script prints this table on every run, so the screen's justification is reproduced rather
than remembered — if a Silver rebuild ever changes the picture, the run says so instead of the
constant quietly going stale.)*

No operator charges 15%. That upper tail is those venues' **incomplete winnings leaking into the
residual**, and it is worth real money — $267,789 on Absolute and $677,131 on PartyPoker — sitting
precisely on the players who play the biggest pots.

So a hand counts as **measured** only if its derived rake is a plausible rake (≤ 6% of the pot,
a 20% margin above the highest value the clean venues ever produce). Everything else is treated
as unmeasured and priced by the estimator, like any hand with no figure at all. Coverage after
the screen:

| Venue | hands | reconciling | plausible | coverage | $ dropped by the screen |
|---|---|---|---|---|---|
| PokerStars | 8,293,797 | 2,231,818 | 2,231,818 | 26.9% | $0.00 |
| iPoker | 5,996,194 | 0 | 0 | **0.0%** | — **excluded** |
| PartyPoker | 3,092,685 | 437,142 | 341,781 | 11.1% | $677,130.87 |
| Ongame | 1,631,839 | 1,584,889 | 1,579,938 | 96.8% | $303,037.26 |
| Full Tilt | 1,283,795 | 344,262 | 344,059 | 26.8% | $6,712.55 |
| Absolute | 1,257,978 | 780,763 | 573,513 | 45.6% | $267,789.24 |

## 4. The estimator: charge every hand what hands like it actually paid

Conditional-mean imputation on four columns that are recorded for **100% of hands** — site, big
blind, whether a flop was dealt, and pot size bucketed in big blinds:

```
rake_est_bb = mean observed rake among (site, big_blind, saw_flop, pot_bucket)
              -> fall back to (site, saw_flop, pot_bucket)      [same venue, other stakes]
              -> fall back to (saw_flop, pot_bucket)            [last resort]
rake_est_usd = rake_est_bb x big_blind        # dollars come back at the end, never before
```

Three decisions inside that, each of which was a bug first:

- **Fitted in big blinds, not dollars.** A dollar cell cannot be borrowed across stakes. The
  first version of this script priced 59,910 PokerStars 25NL monster pots at **$29.79 each**,
  from a fallback cell built mostly out of Ongame hands at a $10 big blind. In big blinds the
  same fallback is approximate instead of absurd.
- **Pot bucketed finely at the bottom and bounded at the top** (every whole bb to 20, then
  20/30/50/100/200/500/1000). On the venues where the dollar cap bites, the rake share falls
  from 5% to under 1% between a 20 bb pot and a 200 bb one; a bucket spanning those would
  average two different rake structures together.
- **Where rake is measured, the measurement is used.** Nothing is modelled over the top of it.

**Validated out of sample.** The cells were fitted on half the measured hands (split on a hash
of `hand_uid`) and scored against the other half. This ratio is the estimator's error bar:

| Venue | held-out hands | actual | predicted | **ratio** | MAE per hand | mean per hand |
|---|---|---|---|---|---|---|
| Absolute | 286,797 | $111,400.64 | $111,732.53 | **1.0030** | $0.1048 | $0.3884 |
| Full Tilt | 171,955 | $107,357.00 | $108,009.33 | **1.0061** | $0.0385 | $0.6243 |
| Ongame | 790,027 | $727,201.66 | $731,109.09 | **1.0054** | $0.1710 | $0.9205 |
| PokerStars | 1,116,165 | $239,959.93 | $241,018.89 | **1.0044** | $0.0172 | $0.2150 |
| PartyPoker | 171,007 | $281,265.87 | $279,282.65 | **0.9929** | $0.7570 | $1.6448 |

**Every venue lands within 0.7% on totals.** Per *hand* it is much rougher — PartyPoker's mean
absolute error is $0.76 against a mean of $1.64 — which is the honest way to state it: this
estimator is trustworthy for sums over thousands of hands and should never be quoted for one.

Of the hands that had to be estimated, **87.1% of the estimated dollars came from the most
specific cell** (same venue, stake, flop status and pot bucket); 1.7% from the second level and
11.1% from the last resort. By hand count the last resort looks larger (27.8%) — but 90% of
those hands are pre-flop folds worth exactly $0.00, which is why the split is reported in
dollars as well as hands.

Priced across each venue's whole observation window (days 1–20 to 1–26, venue by venue), this
comes to **$8,217,411 of rake on 15.56 M hands** across the five venues — $0.53 per hand, against a measured $0.2150 per hand on PokerStars and
$0.9231 on Ongame, whose stakes are ten times larger. The levels are consistent with the stakes.

## 5. Attribution, and the two regression tests that keep it honest

Rake comes out of the pot, not out of a player, so "whose rake is it?" is a modelling choice.
This uses **contributed rake**, the industry-standard answer:

```
player_rake = hand_rake x (what this player put in / what everyone put in)
```

Someone who folds pre-flop contributed nothing and is charged nothing; the two players who built
a 40 bb pot are charged in proportion. **86,216,438 seat-rows** are attributed this way — the
116.6 M-row seat join, less iPoker.

Two tests run on every execution, and both passed at full scale:

1. **Contributed and equal-split attribution must return the same grand total**, because shares
   sum to 1 inside a hand. Measured: **$8,217,411.49 both ways, ratio 1.000000.** This is what
   would catch a fan-out from the non-unique `hand_uid` (iPoker reissues ids; 147 uids cover 547
   hands), which is a live risk, not a theoretical one.
2. **Prior-window hand counts must equal `p_hands` in `gold/player_lapse`, player by player.**
   Measured: **0 mismatches on all 77,268 players.** That catches both a fan-out and any drift
   in the per-venue cutoff between this script and `features_lapse.py` — the money and the label
   have to describe the same week or the multiplication is meaningless.

## 6. Where the money is — by segment

Weekly rake = the last 7 days of each venue's own prior window (the same window as
`p_hands_w1`), so both halves of the multiplication describe the same period.

| Segment | players | weekly rake | **per player** | at-risk players | lapse rate | **$ at risk** | share | measured |
|---|---|---|---|---|---|---|---|---|
| **Recreational** | 31,799 | $432,177 | $13.59 | 26,870 | 77.1% | **$182,482** | **30.2%** | 37.7% |
| **Gambler** | 15,262 | $380,162 | $24.91 | 12,677 | 74.6% | $157,897 | 26.1% | 29.2% |
| **Regular** | 27,376 | $676,975 | $24.73 | 17,434 | 61.6% | $155,173 | 25.7% | 33.1% |
| **Grinder** | 2,831 | $625,312 | **$220.88** | 1,855 | 61.3% | $108,610 | 18.0% | 36.7% |

*(Segment names are re-derived at runtime from the centroids — highest table count is the
Grinder, then highest volume, then highest VPIP — rather than hard-coded to a cluster number
that could change under a re-run. The mapping printed by the run matches
[`segments-results.md`](segments-results.md) exactly.)*

**The money view sharpens the ecosystem thesis and complicates it at the same time, and both
halves belong in the report.**

Sharpens: the recreational segment is 41% of players and carries **30% of all the rake at
risk** — the largest single block — because it is the segment that leaves fastest (77.1%). The
Gambler adds another 26%. **Together those two loose, money-losing segments hold 56% of the
rake at risk while the professionals hold 18%.**

Complicates: per head, a Grinder generates **$220.88 of weekly rake against a Recreational
player's $13.59 — sixteen times as much.** On a pure contributed-rake basis the professionals
are the operator's best customers, which is the opposite of the segmentation write-up's story
if that story is read carelessly.

**It is not a contradiction, and the distinction matters enough to state explicitly:
contributed rake measures who *paid* the rake, not whose money *funded* it.** A grinder pays a
lot of rake with money that was, in the main, lost to them by recreational players — that is
what the bb/100 column in [`segments-results.md`](segments-results.md) says
(Gambler −31.8, Recreational −14.3, Grinder −3.3). Kill the recreational supply and the
grinder's contribution does not survive it, because there is nothing left to recycle. A
"funded-by" attribution — allocating rake in proportion to *net losses* rather than
contributions — is the natural next measurement and has not been run.

## 7. Where the money is — by venue

| Venue | players | weekly rake | per player | lapse rate | **$ at risk** | share | measured |
|---|---|---|---|---|---|---|---|
| PokerStars | 48,561 | $922,104 | $18.99 | 77.9% | **$270,726** | 44.8% | 15.5% |
| PartyPoker | 12,976 | $488,652 | $37.66 | 73.8% | $184,389 | 30.5% | 33.3% |
| Ongame | 3,722 | $386,417 | $103.82 | 49.8% | $77,486 | 12.8% | 84.7% |
| Absolute | 6,737 | $206,273 | $30.62 | 45.8% | $46,966 | 7.8% | 33.3% |
| Full Tilt | 5,272 | $111,180 | $21.09 | 40.4% | $24,596 | 4.1% | 23.4% |

Two things a pooled number would hide. **PokerStars is 63% of the players and 45% of the rake at
risk**, so the headline is substantially a PokerStars statement — and it is the venue with the
*lowest* measured share (15.5%), because its reconciling hands are its smaller pots. And the
per-player figures span $18.99 to $103.82 because the venues sit at different stakes: Ongame's
hands are almost all at a $4–$10 big blind, PokerStars' are 53% at $0.25. **A venue filter on the dashboard is a
blocker, not a nicety** — the same conclusion the segment × venue lapse table reached.

## 8. Who to call — and whether money changes the answer

Ranking 77,268 players three ways and taking the top 10% (7,727 contacts). "Realised $" is the
weekly rake of the contacted players who actually *did* go quiet:

**Held-out fold only** (19,360 players, 1,936 contacts — labels never trained on):

| Ranked by | $ at risk reached | % of total | realised $ lost | true lapsers | precision | overlap with row 1 |
|---|---|---|---|---|---|---|
| **expected loss (risk × $)** | **$107,598** | **70.0%** | **$104,523** | 1,013 | 52.3% | 100% |
| risk alone | $1,375 | **0.9%** | $1,359 | **1,876** | **96.9%** | **0.2%** |
| weekly rake alone | $97,814 | 63.7% | $95,144 | 622 | 32.1% | 66.8% |

The full-population version is the same picture: 69.7% · 0.8% · 63.1%, with 0.1% overlap.

**Yes — weighting by money changes the list almost completely, and this is the single most
useful result in the file.** Ranking by risk alone and ranking by expected loss produce lists
that share **0.2% of their names**. The risk-only list is *better* at finding leavers — 1,876
of them against 1,013, at 96.9% precision — and it reaches **0.9% of the money**. It is a list
of certain departures worth nothing.

That is the sentence the deck should carry: *a retention list sorted by churn risk is almost
exactly the wrong list.* Sorting by money alone is much closer to right (63.7% against 70.0%)
but still misses about a tenth of the rake the better list reaches, and wastes 68% of its calls
on players who were not going anywhere. The product of the two beats both.

## 9. Sensitivity — how much of the headline is a decision?

Four decisions, each re-run:

| Decision | Alternative | Headline under the alternative |
|---|---|---|
| **Which probability** | Fork A's shipped weighted scores | **$424,810 — 29.7% lower.** This is the entire reason §1 exists: the uncalibrated model would have understated the business case by nearly a third. |
| **Rake estimator** | the naive per-player rate × hand count | **$2,483,876 of weekly rake vs $1,981,801 — 25% high**, on the 70,866 players it can even be computed for. Directionally exactly as §2 predicts. |
| **Attribution rule** | equal split between everyone dealt in | prior-window rake $5,569,696 vs $5,682,447, **ratio 0.980**. The rule moves ~2% of the money between players; it does not move the headline. |
| **Modelled venues** | the two best-covered venues only (Ongame + Absolute, 67% measured) | 21.0% of their weekly rake is at risk against 28.6% pooled. |

That last row needs its explanation rather than an alarm: the gap tracks **lapse prevalence,
not the estimator**. Ongame and Absolute have the lowest lapse rates in the dataset (49.8% and
45.8% against PokerStars' 77.9%), so a smaller share of their rake is at risk for reasons that
have nothing to do with how their rake was priced.

## 10. Where it is weak

- **Two thirds of the weekly rake is modelled, not measured** (34.4% measured). The estimator
  validates within 0.7% per venue on held-out *measured* hands — but those hands are, by
  construction, the ones that reconcile. The model conditions on everything recorded for every
  hand (site, stake, flop, pot size); it cannot condition on anything the data does not record,
  and if unmeasured hands differ in some other way, this misses it. That risk is largest on
  PartyPoker (11.1% coverage) and smallest on Ongame (96.8%).
- **Rake is derived, never recorded.** Everything here inherits that. The plausibility screen in
  §3 makes the derivation defensible; it does not make it a recorded column.
- **The headline is substantially a PokerStars number** — 45% of the rake at risk, at the venue
  with the lowest measured share.
- **One cutoff, one week.** Inherited from Fork A: the split is by player, so no player is in
  both folds, but nothing is tested on a *different period*. Do not claim temporal
  generalisation. A backtest at `site_last_day − 14` remains the obvious next robustness check.
- **The dollars are 2009 US dollars at 25NL–1000NL.** The method transfers; the magnitude is a
  seventeen-year-old micro-stakes market. Converting to rupees would require inventing a 2009
  exchange rate, so it is not done.
- **Contributed rake is one attribution among several**, and it answers "who paid", not "whose
  money funded it" — see §6.
- **The population is Fork A's**: players with ≥ 100 prior-window hands. Smaller players are not
  in the $2.11 M, and nothing is imputed for them. An unmeasured spend is not a zero spend.

## 11. What this unlocks

`data/gold/rake_at_risk` carries, per player: the segment, the risk score (calibrated), weekly
rake measured and estimated, the measured share, and expected rake at risk. Every figure in this
document traces to one of those columns — there are no hand-typed numbers.

That is the input for the three things left:

1. **The dashboard** — Gate A1 says it must open on a decision, not a chart, and Gate A8 asks
   for exactly this money-weighted view. The headline sentence is the opening line; the segment
   and venue tables are the two mandatory filters.
2. **The deck** — slide 1 is the headline sentence; the "risk-only list shares 0.2% of its names
   with the expected-loss list" finding is the slide that earns the Business Recommendations
   marks.
3. **The report** — §1's reliability curve is Model Evaluation, §2–§5 are Data Engineering and
   Feature Engineering, and §6's who-paid-versus-whose-money distinction is the paragraph that
   shows the analysis was understood rather than executed.
