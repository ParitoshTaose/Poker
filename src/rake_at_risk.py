"""The money-weighted headline -- "rake at risk".

    reads   data/gold/player_lapse      prior-window features + the lapse label
            data/gold/lapse_scores      Fork A's SHIPPED (class-weighted) risk scores
            data/gold/player_segments   K-Means segment per player
            data/gold/hand_features     rake_bb, pot_bb, big_blind, saw_flop per hand
            data/_work/hp_enriched      116.6 M seat-rows: who invested what, in which hand
    writes  data/_work/player_rake      per-player rake attribution (intermediate, reusable)
            data/gold/rake_at_risk      THE table the dashboard and the deck read
            docs/rake-at-risk-results.json

Run from the repo root:

    .venv/bin/python src/rake_at_risk.py                  # reuse data/_work/player_rake
    .venv/bin/python src/rake_at_risk.py --rebuild-rake   # redo the 86 M-row attribution
    .venv/bin/python src/rake_at_risk.py --quick --parts 8   # smoke test, writes *_sample

WHY THIS SCRIPT EXISTS
----------------------
Fork A ranks players by the chance they go quiet next week. Segmentation says what kind of
player each one is. Neither knows what a player is WORTH, and an operator cannot act on a
ranking that treats a player who generates two cents of rake a week and one who generates
nine dollars as the same row. This script attaches money, so the headline stops being

    "15,423 players are at risk"                     (analytics)

and becomes

    "$X of next week's rake is at risk, most of it recreational, and the top 10% of the
     list by expected loss holds $Z of it"            (a budget decision)

THE HEADLINE IS A MULTIPLICATION, AND BOTH FACTORS WERE BROKEN
--------------------------------------------------------------
1.  P(lapse) was RANKED CORRECTLY BUT NOT CALIBRATED. Fork A's GBT was fitted with
    balanced class weights, which drags predicted probabilities toward 0.5: mean risk per
    segment 0.52 / 0.53 / 0.64 / 0.67 against observed lapse rates of
    0.616 / 0.613 / 0.746 / 0.771 -- right order, about 9 points low. ROC-AUC and PR-AUC
    never notice, because they only look at order. A dollar figure notices immediately.
    Part 1 refits without weights and verifies with a reliability curve.

2.  RAKE IS NOT A RECORDED COLUMN, and the hands where it can be derived are NOT A RANDOM
    SAMPLE. This is the trap this script was originally going to walk into, so it is
    written down. Rake is derived as (pot - paid-out winnings) and `parse_phh.py` accepts
    the result only when it lands in a plausible range (<= 15% of the pot). On a venue
    whose exports are incomplete, that test is *easier to pass on a big pot* -- a fixed
    shortfall is 40% of a small pot and 3% of a large one. Measured consequence on
    PartyPoker: the hands where rake reconciles have a median pot of 22.2 bb and 99.6% of
    them saw a flop, against 2.5 bb and 36.8% on the hands where it does not.

    So the obvious method -- take each player's observed rake per 100 hands and scale it
    to all their hands -- inflates the answer roughly tenfold, because it charges every
    hand at the rate of the biggest pots and ignores that most hands never see a flop and
    pay nothing at all ("no flop, no drop", measured here at 99.6-100% of pre-flop-only
    hands paying exactly $0.00). That naive figure is still computed below, and reported
    beside the real one, because knowing how wrong it is is the reason to trust the other.

THE ESTIMATOR, IN ONE SENTENCE
------------------------------
Charge every unmeasured hand what hands *just like it* actually paid: the average observed
rake among hands on the same site, at the same big blind, with the same flop status, in the
same pot-size bucket. Conditional-mean imputation on four fully-recorded columns.

    * it conditions only on things recorded for 100% of hands -- site, big blind, whether
      a flop was dealt, pot size -- so it can be applied everywhere, not just where the
      money reconciles;
    * pot size is the dominant driver of rake and it carries the dollar cap for free,
      without anyone having to guess what the cap was;
    * where rake IS observed, the observed value is used and nothing is modelled;
    * and it is VALIDATED OUT OF SAMPLE: the cells are fitted on half the reconciling
      hands and scored against the other half, per venue. That ratio is the error bar.

What it cannot do: condition on anything the data does not record. If unmeasured hands
differ from measured ones in some way beyond stake, pot size and flop status, this misses
it. Said out loud in the write-up, every time.

WHY CONTRIBUTED-RAKE ATTRIBUTION
--------------------------------
Rake comes out of the pot, not out of a player, so "whose rake is it?" is a choice, not a
fact. The industry-standard answer is contributed rake:

    player_rake = hand_rake x (what this player put in / what everyone put in)

Someone who folds pre-flop contributed nothing and is charged nothing; the two players who
built a 40 bb pot are charged in proportion. The naive alternative -- split the rake
equally between everyone dealt in -- is computed too, purely as a sensitivity check: the
two MUST agree on the grand total while disagreeing per player.

THE REGRESSION TESTS TO KEEP FOREVER
------------------------------------
    1. Shares sum to 1 inside a hand, so contributed and equal-split attribution must
       return the same grand total to the cent. If they diverge, the seat x hand join has
       fanned out -- and hand_uid is NOT unique (iPoker reissues ids), so that is a live
       risk, not a theoretical one.
    2. Prior-window hands counted through this join must equal `p_hands` in
       gold/player_lapse, player by player. That catches both a fan-out and any drift in
       the per-venue cutoff between the two scripts.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

# Pin both ends of the Spark <-> Python bridge before pyspark is imported, or Spark starts
# its workers under Apple's Python 3.9, which cannot import pyspark 4.2 -- and it fails
# inside a Java stack trace that says nothing about Python versions. See CLAUDE.md.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import GBTClassifier  # noqa: E402
from pyspark.ml.feature import Imputer, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.ml.regression import IsotonicRegression  # noqa: E402
from pyspark.sql import SparkSession, Window  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

# The one cross-import in this repo, and it is deliberate: the feature list, the seed and
# the split fraction MUST be the ones Fork A used, or "same population, same split, only
# the weighting changed" stops being true and the calibration comparison is worthless.
# Copy-pasting them would let the two drift apart silently.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fork_a import CORE, SEED, TEST_FRACTION, best_f1  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAPSE = ROOT / "data" / "gold" / "player_lapse"
SCORES = ROOT / "data" / "gold" / "lapse_scores"
SEGMENTS = ROOT / "data" / "gold" / "player_segments"
HANDS = ROOT / "data" / "gold" / "hand_features"
SEATS = ROOT / "data" / "_work" / "hp_enriched"
SCRATCH = ROOT / "data" / "_work" / "spark-scratch"

RAKE_WORK = ROOT / "data" / "_work" / "player_rake"
OUT_TABLE = ROOT / "data" / "gold" / "rake_at_risk"
OUT_JSON = ROOT / "docs" / "rake-at-risk-results.json"

# ---- decisions, declared up front so none of them can be chosen after seeing a number ----

# A venue whose money never reconciles has nothing to fit an estimator on. Measured, not
# hard-coded, so this follows the data if Silver is ever rebuilt. iPoker comes out at
# 0.0% and is the only venue this excludes.
MIN_VENUE_COVERAGE = 0.01

# THE PLAUSIBILITY CEILING, and the evidence for it.
#
# `parse_phh.py` accepts a derived rake if the residual (pot - paid-out winnings) is at
# most 15% of the pot -- a deliberately loose bound, chosen when there was nothing better
# to test against. There is now something better. Measured across all six venues, the
# MEDIAN rake is 4.3%-5.0% of the pot on every one of them, which is exactly the
# commercial rake of the period; and on PokerStars, whose exports reconcile completely,
# **not one of 1,107,219 flop hands has a rake above 5.0% of the pot.** Full Tilt (0.13%)
# and Ongame (0.70%) agree.
#
# Two venues do not: 41.5% of Absolute's flop hands and 21.9% of PartyPoker's come out
# above 6%, up to the 15% ceiling. That is not a different rake structure -- no operator
# charges 15% -- it is those venues' incomplete winnings leaking into the residual. It is
# also worth $267 k on ABS and $677 k on PTY, so leaving it in would inflate exactly the
# players who play the biggest pots.
#
# So a hand counts as MEASURED only if its derived rake is a plausible rake. The rest are
# treated as unmeasured and priced by the estimator, like any other hand with no figure.
# Never silently: the run prints what the screen drops, per venue.
MAX_RAKE_SHARE = 0.06

# Minimum observed hands in a cell before its mean is trusted; below it the estimator
# falls back to (site, flop, pot bucket) and then to (flop, pot bucket). The share of
# hands priced at each level is reported.
MIN_CELL = 30

# For the honesty slice: the venues where the most hands are measured rather than modelled.
# 0.40 takes in Absolute (45.6% after the screen) alongside Ongame (96.8%); the next venue
# down is PokerStars at 26.9%.
HIGH_COVERAGE = 0.40

# For the naive-method comparison only -- the number of observed hands the discarded
# "scale each player's own rake rate" approach would have required.
MIN_MONEY_HANDS = 20

# Calibration is judged by the largest gap between predicted and observed lapse rate
# across ten equal-sized deciles of the test fold. Inside this tolerance the unweighted
# refit is used raw; outside it, isotonic regression is fitted ON THE TRAIN FOLD. Stated
# here, before the run, so the choice is a rule and not a preference.
CAL_TOLERANCE = 0.03

# "At risk" for the headline count. With a calibrated model this is a real statement --
# more likely than not to play zero hands next week -- not an arbitrary default.
AT_RISK_THRESHOLD = 0.50

# What a retention team can actually work through in a week.
CONTACT_BUDGET = 0.10

LAPSE_WINDOW = 7          # must match features_lapse.py, or the windows drift apart
W1_DAYS = 7               # "weekly" = the last 7 days of the prior window (= p_hands_w1)

results: dict = {}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def usd(x):
    return f"${x:,.2f}"


def session():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return (SparkSession.builder
            .appName("ecosystem-engine-rake-at-risk")
            .master("local[8]")
            .config("spark.driver.memory", "9g")
            .config("spark.sql.shuffle.partitions", "96")     # the seat-row join needs it
            .config("spark.sql.files.maxPartitionBytes", "64m")
            .config("spark.local.dir", str(SCRATCH.resolve()))  # never /tmp -- see CLAUDE.md
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def fold(df):
    """Fork A's split, reproduced exactly: a hash of the player key, so the same player
    lands in the same fold on every run and on every machine. One row per
    (site, player_id), so no player can appear in both folds -- and IDs are never merged
    across venues, because the same human on two sites is two different codes."""
    return df.withColumn(
        "fold",
        F.when(F.pmod(F.hash(F.concat_ws(":", "site", "player_id", F.lit(SEED))),
                      F.lit(100)) < TEST_FRACTION, "test").otherwise("train"))


# ======================================================================================
# PART 1 -- calibration
# ======================================================================================
def reliability(y, p, bins=10):
    """The reliability curve: sort by predicted probability, cut into equal-sized bins,
    and ask whether the model's stated confidence matches what actually happened.

    Equal-COUNT bins (split the sorted order) rather than equal-WIDTH bins, because the
    predictions bunch up: a fixed 0.0-0.1 bucket can end up holding eleven players, and a
    deviation measured on eleven players is noise being reported as a finding.

    Three numbers come out of it:
        max_abs_deviation  the worst decile. This is the one to quote -- it is the largest
                           error a dollar figure can inherit from the probability.
        ece                the average gap, weighted by how many players sit in each bin.
        brier              mean squared error of the probabilities themselves; unlike
                           ROC-AUC it punishes being confidently wrong, which is exactly
                           the failure mode that matters when multiplying by money.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    order = np.argsort(p, kind="stable")
    rows = []
    for i, g in enumerate(np.array_split(order, bins)):
        rows.append({
            "decile": i + 1, "n": int(len(g)),
            "p_low": float(p[g].min()), "p_high": float(p[g].max()),
            "mean_predicted": float(p[g].mean()), "observed": float(y[g].mean()),
            "deviation": float(p[g].mean() - y[g].mean()),
        })
    return {
        "bins": rows,
        "max_abs_deviation": float(max(abs(r["deviation"]) for r in rows)),
        "ece": float(sum(r["n"] * abs(r["deviation"]) for r in rows) / len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "mean_predicted": float(p.mean()), "observed_rate": float(y.mean()),
        "n": int(len(y)),
    }


def show_reliability(rel, label):
    print(f"\n  reliability curve · {label}   (n={rel['n']:,})")
    print(f"  {'decile':>6s} {'players':>8s} {'p range':>15s} {'predicted':>10s} "
          f"{'observed':>9s} {'gap':>8s}")
    for b in rel["bins"]:
        print(f"  {b['decile']:6d} {b['n']:8,} "
              f"{b['p_low']:6.3f}-{b['p_high']:<8.3f} {b['mean_predicted']:10.3f} "
              f"{b['observed']:9.3f} {b['deviation']:+8.3f}")
    print(f"  mean predicted {rel['mean_predicted']:.3f} vs observed "
          f"{rel['observed_rate']:.3f}   ·   worst decile gap "
          f"{rel['max_abs_deviation']:.3f}   ·   ECE {rel['ece']:.3f}   ·   "
          f"Brier {rel['brier']:.4f}")


def roc_pr(y, p):
    """ROC-AUC and PR-AUC on the collected test fold, because the comparison here is
    against Fork A's shipped scores, which arrive as a plain Parquet column rather than
    through an MLlib pipeline.

    ROC-AUC via the Mann-Whitney identity (mean rank of the positives), so ties are
    handled the way MLlib handles them. Both are RANKING metrics and neither can see
    calibration -- which is exactly why this script exists.
    """
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    n1, n0 = y.sum(), (1 - y).sum()
    r = pd.Series(p).rank().to_numpy()
    roc = (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    o = np.argsort(-p, kind="stable")
    ys = y[o]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    rec = tp / n1
    pr = float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))
    return {"roc_auc": float(roc), "pr_auc": pr}


def calibrate(spark, quick):
    """Refit Fork A's CORE GBT without class weights, verify the calibration, and fall
    back to isotonic regression only if the refit is still outside tolerance.

    Why dropping the weights is the first thing to try: at 68/32 there was never a real
    imbalance to correct. Weighting exists for problems where the positive class is 1% of
    the data and a model can score 99% accuracy by predicting "no" forever. Here the
    majority class IS the event of interest, so the weights bought nothing and cost the
    probabilities their meaning.
    """
    hr("PART 1 · RECALIBRATION — a probability has to mean what it says")
    df = fold(spark.read.parquet(str(LAPSE)).withColumnRenamed("lapsed", "label"))

    # MLlib's Imputer accepts only double/float -- an int column fails the schema check
    # rather than being promoted. Cast once, up front. (Recorded trap; see CLAUDE.md.)
    for c in CORE:
        df = df.withColumn(c, F.col(c).cast("double"))

    # Missing indicators BEFORE imputation, so "this was filled in" survives into the
    # model as information instead of being laundered into a plausible-looking value.
    nulls = df.select([F.avg(F.col(c).isNull().cast("double")).alias(c)
                       for c in CORE]).first().asDict()
    gappy = [c for c, r in nulls.items() if r > 0.005]
    for c in gappy:
        df = df.withColumn(f"{c}_missing", F.col(c).isNull().cast("double"))
    indicators = [f"{c}_missing" for c in gappy]

    sites = sorted(r["site"] for r in df.select("site").distinct().collect())
    dummies = []
    for s in sites:
        if s == "PS":                        # the largest venue -> the reference level
            continue
        df = df.withColumn(f"site_{s}", (F.col("site") == s).cast("double"))
        dummies.append(f"site_{s}")

    # 96 tiny Parquet parts for 90 k rows; GBT runs one Spark job per boosting iteration,
    # so fewer, fuller partitions is worth one shuffle here.
    df = df.repartition(8).cache()
    train, test = df.filter("fold = 'train'"), df.filter("fold = 'test'")
    n_tr, n_te = train.count(), test.count()
    print(f"  {n_tr:,} train / {n_te:,} test — the same hash split as fork_a.py, "
          f"{len(CORE) + len(indicators) + len(dummies)} features, NO weightCol")

    feats = [f"{c}_i" for c in CORE] + indicators + dummies
    pipe = Pipeline(stages=[
        Imputer(strategy="median", inputCols=CORE,
                outputCols=[f"{c}_i" for c in CORE]),
        VectorAssembler(inputCols=feats, outputCol="v", handleInvalid="keep"),
        GBTClassifier(featuresCol="v", labelCol="label",     # <- no weightCol: the fix
                      maxIter=30 if quick else 120, maxDepth=5,
                      stepSize=0.1, seed=SEED),
    ])
    t = time.time()
    model = pipe.fit(train)
    print(f"  refit in {time.time() - t:.0f}s")

    scored = (model.transform(df)
              .withColumn("risk_uncal", vector_to_array("probability")[1])
              .select("site", "player_id", "fold", "label", "risk_uncal"))

    # The "before" curve comes from the shipped scores, not from a re-fit: that is the
    # model whose miscalibration was reported, so it is the honest comparison.
    before = (spark.read.parquet(str(SCORES))
              .select("site", "player_id", F.col("lapse_risk").alias("risk_weighted")))
    both = scored.join(before, ["site", "player_id"], "left").cache()
    te = both.filter("fold = 'test'").toPandas()          # 22 k rows x 5 cols: Gold grain
    y = te["label"].to_numpy()

    rel_w = reliability(y, te["risk_weighted"].to_numpy())
    rel_u = reliability(y, te["risk_uncal"].to_numpy())
    show_reliability(rel_w, "BEFORE · Fork A's shipped GBT, balanced class weights")
    show_reliability(rel_u, "AFTER · same GBT, same split, no class weights")

    rank_w = roc_pr(y, te["risk_weighted"].to_numpy())
    rank_u = roc_pr(y, te["risk_uncal"].to_numpy())
    print(f"\n  ranking is what must NOT move:  weighted ROC {rank_w['roc_auc']:.4f} / "
          f"PR {rank_w['pr_auc']:.4f}   ->   unweighted ROC {rank_u['roc_auc']:.4f} / "
          f"PR {rank_u['pr_auc']:.4f}")
    print("  (fork_a.py measured 0.8566 / 0.9173 for the weighted GBT — the recomputation")
    print("   above should land on it; GBT is not bit-reproducible past ~5 decimals.)")

    out = {"weighted": {"reliability": rel_w, **rank_w},
           "unweighted": {"reliability": rel_u, **rank_u},
           "n_train": n_tr, "n_test": n_te, "gappy_columns": gappy,
           "tolerance": CAL_TOLERANCE}

    # ---- isotonic, only if the refit is still outside the tolerance declared up front
    if rel_u["max_abs_deviation"] <= CAL_TOLERANCE:
        print(f"\n  worst decile gap {rel_u['max_abs_deviation']:.3f} <= {CAL_TOLERANCE} "
              "tolerance -> the unweighted probabilities are used as they")
        print("  are. Isotonic regression would add a fitted layer for no measured gain.")
        final, chosen = both.withColumn("risk_cal", F.col("risk_uncal")), "unweighted"
    else:
        print(f"\n  worst decile gap {rel_u['max_abs_deviation']:.3f} > {CAL_TOLERANCE}"
              " -> fitting isotonic regression on the TRAIN fold only.")
        iso_in = VectorAssembler(inputCols=["risk_uncal"], outputCol="fv").transform(
            both.filter("fold = 'train'").select("label", "risk_uncal"))
        iso = IsotonicRegression(featuresCol="fv", labelCol="label").fit(iso_in)
        final = (iso.transform(
            VectorAssembler(inputCols=["risk_uncal"], outputCol="fv").transform(both))
            .withColumnRenamed("prediction", "risk_cal").drop("fv"))
        iso_te = final.filter("fold = 'test'").select("label", "risk_cal").toPandas()
        rel_i = reliability(iso_te["label"], iso_te["risk_cal"])
        show_reliability(rel_i, "AFTER · unweighted GBT + isotonic (fitted on train)")
        out["isotonic"] = {"reliability": rel_i,
                           **roc_pr(iso_te["label"], iso_te["risk_cal"])}
        chosen = "isotonic"

    # Calibration measured per fold: the train fold's scores are in-sample, so if the
    # model is sharper there than on test, a headline summed over everybody inherits it.
    fin = final.select("site", "player_id", "fold", "label", "risk_weighted",
                       "risk_uncal", "risk_cal").cache()
    fin.count()
    per_fold = {}
    for f_ in ("train", "test"):
        s = fin.filter(F.col("fold") == f_).select("label", "risk_cal").toPandas()
        r = reliability(s["label"], s["risk_cal"])
        per_fold[f_] = {k: r[k] for k in
                        ("n", "mean_predicted", "observed_rate", "max_abs_deviation",
                         "ece", "brier")}
        print(f"  {f_:5s} fold: mean predicted {r['mean_predicted']:.3f} vs observed "
              f"{r['observed_rate']:.3f}, worst decile {r['max_abs_deviation']:.3f}"
              + ("   <- in-sample, expected to be sharper" if f_ == "train" else ""))
    out["per_fold_calibration"] = per_fold
    out["chosen"] = chosen

    # The operating point, recomputed: fork_a.py's 0.25 belongs to the weighted scores
    # and does not transfer to a calibrated model.
    bf = best_f1(y, te["risk_uncal"].to_numpy())
    print(f"  best-F1 operating point for the recalibrated model: threshold "
          f"{bf['threshold']:.2f} -> F1 {bf['f1']:.3f} "
          f"(precision {bf['precision']:.3f}, recall {bf['recall']:.3f})")
    out["best_f1"] = bf

    results["calibration"] = out
    both.unpersist()
    df.unpersist()
    return fin


# ======================================================================================
# PART 2 -- the rake estimator, and attribution to players
# ======================================================================================
def pot_bucket(col):
    """Pot size in big blinds, bucketed: every whole big blind up to 20, then widening.

    Fine at the bottom because that is where most hands live (median pot 2.5 bb) and where
    the rake percentage is at its full rate; coarse at the top because those hands are rare
    -- but not unbounded at the top, because on the venues where the dollar cap bites, the
    rake share falls from 5% to under 1% between a 20 bb pot and a 200 bb one, and a bucket
    that mixed those would average two different rake structures together.
    """
    return (F.when(col < 20, F.floor(col))
            .when(col < 30, F.lit(20)).when(col < 50, F.lit(30))
            .when(col < 100, F.lit(50)).when(col < 200, F.lit(100))
            .when(col < 500, F.lit(200)).when(col < 1000, F.lit(500))
            .otherwise(F.lit(1000))).cast("int")


# Fitted in BIG BLINDS, not dollars, and this matters. A cell fitted in dollars cannot be
# borrowed across stakes: the first version of this script priced 59,910 PokerStars 25NL
# monster pots at $29.79 each, from a fallback cell built mostly out of Ongame hands at a
# $10 big blind. In big blinds the same fallback is merely approximate instead of absurd,
# and the dollar figure comes back by multiplying by the hand's own big blind.
CELL_KEYS = [["site", "big_blind", "saw_flop", "pot_bin"],   # venue, stake, pot: exact
             ["site", "saw_flop", "pot_bin"],                # same venue, other stakes
             ["saw_flop", "pot_bin"]]                        # last resort, all venues


def fit_cells(obs):
    """The estimator: what hands like this one actually paid, in big blinds.

    Three nested tables so a thin cell can fall back instead of returning nothing. All
    three are tiny -- a few thousand rows -- and broadcast.
    """
    return [obs.groupBy(*k).agg(F.avg("rake_bb_obs").alias(f"cell{i}"),
                                F.count("*").alias(f"n{i}")).cache()
            for i, k in enumerate(CELL_KEYS, start=1)]


def apply_cells(hands, cells):
    """Attach the estimate and record WHICH level priced each hand, so the fallbacks are
    reported rather than hidden."""
    h = hands
    for k, c in zip(CELL_KEYS, cells):
        h = h.join(F.broadcast(c), k, "left")
    return (h.withColumn("rake_est_bb",
                         F.when(F.col("n1") >= MIN_CELL, F.col("cell1"))
                         .when(F.col("n2") >= MIN_CELL, F.col("cell2"))
                         .otherwise(F.coalesce(F.col("cell3"), F.lit(0.0))))
            .withColumn("rake_est_usd", F.col("rake_est_bb") * F.col("big_blind"))
            .withColumn("cell_level",
                        F.when(F.col("n1") >= MIN_CELL, F.lit(1))
                        .when(F.col("n2") >= MIN_CELL, F.lit(2)).otherwise(F.lit(3)))
            .drop("cell1", "cell2", "cell3", "n1", "n2", "n3"))


def measured(rake_bb, pot_bb):
    """Is this hand's rake a MEASUREMENT, or is it a residual wearing a rake's clothes?

    Both conditions have to hold: a figure exists, and it is a plausible rake. See
    MAX_RAKE_SHARE for the evidence behind the second one. Pre-flop-only hands pass
    trivially -- their rake is exactly $0.00 on 99.6-100% of hands, venue by venue.
    """
    return rake_bb.isNotNull() & (rake_bb <= MAX_RAKE_SHARE * F.greatest(pot_bb,
                                                                        F.lit(0.0)))


def venue_coverage(spark):
    """Where can rake be measured at all? Measured every run, even when the attribution is
    being reused, because the exclusion decision and the coverage caveat are quoted
    alongside every dollar figure this script produces."""
    hr("PART 2a · WHERE RAKE CAN BE MEASURED")
    cov = (spark.read.parquet(str(HANDS))
           .withColumn("plausible", measured(F.col("rake_bb"), F.col("pot_bb")))
           .groupBy("site", "venue")
           .agg(F.count("*").alias("hands"),
                F.sum(F.col("rake_bb").isNotNull().cast("int")).alias("ok"),
                F.sum(F.col("plausible").cast("int")).alias("plaus"),
                F.sum(F.when(F.col("rake_bb").isNotNull(),
                             F.col("rake_bb") * F.col("big_blind"))).alias("usd_ok"),
                F.sum(F.when(F.col("plausible"),
                             F.col("rake_bb") * F.col("big_blind"))).alias("usd_plaus"))
           .withColumn("coverage", F.col("plaus") / F.col("hands"))
           .orderBy(F.desc("hands")).collect())
    print("  The money filter is rake_bb IS NOT NULL — never 'winnings present', because a")
    print("  further 13.8% of hands carry a full set of winnings summing to ~1.5x the pot.")
    print(f"  Then the plausibility screen: a derived rake above {MAX_RAKE_SHARE:.0%} of the "
          f"pot is missing\n  money, not rake (PokerStars: 0 of 1.1 M flop hands exceed 5%).")
    print(f"\n    {'site':5s} {'venue':16s} {'hands':>11s} {'reconcile':>11s} "
          f"{'plausible':>11s} {'cov':>6s} {'$ dropped by screen':>20s}")
    excluded = []
    for r in cov:
        flag = ""
        if r["coverage"] < MIN_VENUE_COVERAGE:
            excluded.append(r["site"])
            flag = "  <- EXCLUDED: nothing to fit on"
        drop = (r["usd_ok"] or 0) - (r["usd_plaus"] or 0)
        print(f"    {r['site']:5s} {r['venue']:16s} {r['hands']:11,} {r['ok']:11,} "
              f"{r['plaus']:11,} {r['coverage']:5.1%} {usd(drop):>20s}{flag}")
    results["money_coverage_by_venue"] = [
        {"site": r["site"], "venue": r["venue"], "hands": r["hands"],
         "reconciling": r["ok"], "plausible": r["plaus"],
         "coverage": float(r["coverage"]),
         "usd_reconciling": float(r["usd_ok"] or 0),
         "usd_plausible": float(r["usd_plaus"] or 0)} for r in cov]
    results["excluded_venues"] = excluded
    results["max_rake_share"] = MAX_RAKE_SHARE

    # ---- the evidence FOR the screen, measured here rather than asserted, because it is
    # the most contestable decision in this script. If a later rebuild of Silver changes
    # the picture, this table says so instead of the constant quietly going stale.
    sh = (spark.read.parquet(str(HANDS))
          .filter(F.col("rake_bb").isNotNull() & F.col("saw_flop") & (F.col("pot_bb") > 0))
          .withColumn("share", F.col("rake_bb") / F.col("pot_bb"))
          .groupBy("site")
          .agg(F.count("*").alias("hands"),
               F.expr("percentile_approx(share, 0.5)").alias("median"),
               F.expr("percentile_approx(share, 0.9)").alias("p90"),
               F.avg((F.col("share") > MAX_RAKE_SHARE).cast("double")).alias("over"),
               F.avg((F.col("share") > 0.14).cast("double")).alias("ceiling"))
          .orderBy("site").collect())
    print("\n  the evidence for that screen — derived rake as a share of the pot, flop hands:")
    print(f"    {'site':5s} {'flop hands':>11s} {'median':>8s} {'p90':>8s} "
          f"{'> ' + format(MAX_RAKE_SHARE, '.0%'):>8s} {'at 15% bound':>13s}")
    for r in sh:
        print(f"    {r['site']:5s} {r['hands']:11,} {r['median']:7.2%} {r['p90']:7.2%} "
              f"{r['over']:7.2%} {r['ceiling']:12.2%}")
    print("    A real rake is ~5% of the pot and every venue's MEDIAN agrees. The venues that")
    print("    disagree do so only in the upper tail, which is where incomplete winnings land.")
    results["rake_share_by_venue"] = [
        {"site": r["site"], "flop_hands": r["hands"], "median_share": float(r["median"]),
         "p90_share": float(r["p90"]), "above_ceiling": float(r["over"]),
         "at_parser_bound": float(r["ceiling"])} for r in sh]
    return excluded


def price_hands(spark, excluded):
    """Fit the estimator, validate it, and attach a rake figure to every hand.

    Deliberately NOT part of the attribution step below: that one is skipped when
    `data/_work/player_rake` is being reused, and the estimator's validation is quoted in
    the write-up, so it has to be produced on every run or the JSON goes stale in a way
    nobody would notice. It reads only `gold/hand_features` (21.5 M rows) and costs
    seconds, so running it always is cheap insurance.
    """
    hr("PART 2b · PRICING EVERY HAND — what did a hand like this actually pay?")

    hands_all = spark.read.parquet(str(HANDS)).select(
        "hand_uid", "site", "venue", "day", "big_blind", "pot_bb", "saw_flop", "rake_bb",
        "n_players")

    hands = (hands_all.filter(~F.col("site").isin(excluded) if excluded else F.lit(True))
             .withColumn("pot_bin", pot_bucket(F.coalesce("pot_bb", F.lit(0.0))))
             .withColumn("is_obs", measured(F.col("rake_bb"), F.col("pot_bb")))
             # kept separately: the estimator is fitted in big blinds, the report is in
             # dollars, and conflating the two is how the first version priced a 25NL pot
             # at $29.79.
             .withColumn("rake_bb_obs", F.when(F.col("is_obs"), F.col("rake_bb")))
             .withColumn("rake_usd_obs",
                         F.when(F.col("is_obs"),
                                F.col("rake_bb") * F.col("big_blind"))))

    # ---- 2b. THE DIAGNOSTIC that rules out the naive method. If reconciling hands were a
    # random sample of all hands, these two rows would match inside each venue. They do
    # not: the acceptance test (residual <= 15% of pot) is easier to pass on a big pot, so
    # on a venue with incomplete exports it selects big post-flop pots.
    print("\n  are the measurable hands representative? (if not, scaling a per-player rate")
    print("  off them charges every hand at the rate of the biggest pots):")
    print(f"    {'site':5s} {'measured?':>10s} {'hands':>11s} {'median pot bb':>14s} "
          f"{'mean pot bb':>12s} {'saw flop':>9s}")
    rep = (hands.groupBy("site", "is_obs")
           .agg(F.count("*").alias("hands"),
                F.expr("percentile_approx(pot_bb, 0.5)").alias("med_pot"),
                F.avg("pot_bb").alias("mean_pot"),
                F.avg(F.col("saw_flop").cast("double")).alias("flop"))
           .orderBy("site", F.desc("is_obs")).collect())
    for r in rep:
        print(f"    {r['site']:5s} {str(r['is_obs']):>10s} {r['hands']:11,} "
              f"{r['med_pot']:14.2f} {r['mean_pot']:12.2f} {r['flop']:8.1%}")
    results["representativeness"] = [
        {"site": r["site"], "measured": bool(r["is_obs"]), "hands": r["hands"],
         "median_pot_bb": float(r["med_pot"]), "mean_pot_bb": float(r["mean_pot"]),
         "saw_flop_share": float(r["flop"])} for r in rep]

    # ---- 2c. fit the estimator, and validate it on reconciling hands it never saw.
    obs = hands.filter("is_obs")
    half = obs.withColumn("h", F.pmod(F.hash("hand_uid", F.lit(SEED)), F.lit(2)))
    val = (apply_cells(half.filter("h = 1"), fit_cells(half.filter("h = 0")))
           .groupBy("site")
           .agg(F.count("*").alias("hands"),
                F.sum("rake_usd_obs").alias("actual"),
                F.sum("rake_est_usd").alias("predicted"),
                F.avg(F.abs(F.col("rake_est_usd") - F.col("rake_usd_obs")))
                 .alias("mae"),
                F.avg("rake_usd_obs").alias("mean_actual"))
           .orderBy("site").collect())
    print("\n  OUT-OF-SAMPLE VALIDATION · cells fitted on half the reconciling hands,")
    print("  scored against the other half. This ratio is the estimator's error bar:")
    print(f"    {'site':5s} {'hands':>10s} {'actual $':>13s} {'predicted $':>13s} "
          f"{'ratio':>7s} {'MAE $/hand':>11s} {'mean $/hand':>12s}")
    v_out = []
    for r in val:
        ratio = r["predicted"] / r["actual"] if r["actual"] else float("nan")
        print(f"    {r['site']:5s} {r['hands']:10,} {r['actual']:13,.2f} "
              f"{r['predicted']:13,.2f} {ratio:7.4f} {r['mae']:11.4f} "
              f"{r['mean_actual']:12.4f}")
        v_out.append({"site": r["site"], "hands": r["hands"],
                      "actual_usd": float(r["actual"]),
                      "predicted_usd": float(r["predicted"]), "ratio": float(ratio),
                      "mae_usd_per_hand": float(r["mae"]),
                      "mean_actual_usd_per_hand": float(r["mean_actual"])})
    results["estimator_validation"] = v_out

    cells = fit_cells(obs)
    priced = apply_cells(hands, cells).withColumn(
        "rake_used_usd", F.coalesce("rake_usd_obs", "rake_est_usd"))

    # Reported in DOLLARS as well as hands: 90% of the hands priced at the coarsest level
    # are pre-flop folds worth $0.00, so a hand count alone makes the fallbacks look far
    # more load-bearing than they are.
    lvl = (priced.filter(~F.col("is_obs")).groupBy("cell_level")
           .agg(F.count("*").alias("hands"),
                F.sum("rake_est_usd").alias("usd")).collect())
    tot_h = sum(r["hands"] for r in lvl) or 1
    tot_u = sum(r["usd"] for r in lvl) or 1
    print("\n  of the hands that had to be ESTIMATED, which cell priced them")
    print("  (level 1 = same venue, stake, flop status and pot bucket):")
    for r in sorted(lvl, key=lambda r: r["cell_level"]):
        print(f"    level {r['cell_level']}  {r['hands']:12,} hands "
              f"({r['hands'] / tot_h:5.1%})   {usd(r['usd']):>16s} "
              f"({r['usd'] / tot_u:5.1%} of the estimated dollars)")
    print("  and what that comes to, per venue, on ALL hands vs only the measured ones:")
    pv = (priced.groupBy("site")
          .agg(F.count("*").alias("hands"),
               F.avg("rake_used_usd").alias("used"),
               F.avg(F.when(F.col("is_obs"), F.col("rake_usd_obs"))).alias("obs"),
               F.sum("rake_used_usd").alias("tot_used"),
               F.sum(F.when(F.col("is_obs"), F.col("rake_usd_obs"))).alias("tot_obs"))
          .orderBy("site").collect())
    print(f"    {'site':5s} {'hands':>11s} {'$/hand all':>11s} {'$/hand measured':>16s} "
          f"{'total $ all':>14s} {'total $ measured':>17s}")
    for r in pv:
        print(f"    {r['site']:5s} {r['hands']:11,} {r['used']:11.4f} "
              f"{(r['obs'] or 0):16.4f} {r['tot_used']:14,.2f} "
              f"{(r['tot_obs'] or 0):17,.2f}")
    results["priced_by_venue"] = [
        {"site": r["site"], "hands": r["hands"], "usd_per_hand_all": float(r["used"]),
         "usd_per_hand_measured": float(r["obs"] or 0),
         "total_usd_all": float(r["tot_used"]),
         "total_usd_measured": float(r["tot_obs"] or 0)} for r in pv]

    # ---- per-venue window. Never a global calendar date: the venues stop recording on
    # different days, so a global window would mark every ABS and PS player as lapsed.
    window = (hands_all.groupBy("site")
              .agg(F.max("day").alias("site_last_day"))
              .withColumn("cutoff", F.col("site_last_day") - LAPSE_WINDOW))
    print("\n  per-venue window (identical to features_lapse.py, or the money and the")
    print("  label would describe different weeks):")
    for r in window.orderBy("site").collect():
        print(f"    {r['site']:5s} last day {r['site_last_day']:3d}  cutoff "
              f"{r['cutoff']:3d}   weekly window = days "
              f"{r['cutoff'] - W1_DAYS + 1}-{r['cutoff']}")
    return priced, window


def build_player_rake(spark, parts, priced, window):
    """Attribute every priced dollar to the players who built the pot, and roll up to one
    row per player. The only heavy step in this script: 116.6 M seat-rows."""
    hr("PART 2c · ATTRIBUTION — from a pot to a player")
    seat_files = ([str(p) for p in sorted(SEATS.glob("*.parquet"))[:parts]]
                  if parts else [str(SEATS)])
    seats = spark.read.parquet(*seat_files).select(
        "hand_uid", "player_id", "invested_bb")

    # Inner join drops the excluded venues in one step.
    j = seats.join(priced.select("hand_uid", "site", "day", "rake_usd_obs",
                                 "rake_used_usd", "is_obs"), "hand_uid")

    # One shuffle by hand_uid gives both denominators: what the whole table put in
    # (contributed-rake attribution) and how many seats were dealt in (the equal-split
    # alternative, kept only as a sensitivity check on the attribution choice).
    w = Window.partitionBy("hand_uid")
    j = (j.withColumn("inv", F.coalesce("invested_bb", F.lit(0.0)))
         .withColumn("hand_invested", F.sum("inv").over(w))
         .withColumn("hand_seats", F.count("*").over(w))
         .withColumn("share", F.when(F.col("hand_invested") > 0,
                                     F.col("inv") / F.col("hand_invested"))
                     .otherwise(F.lit(1.0) / F.col("hand_seats")))
         .join(F.broadcast(window), "site")
         .withColumn("rd", F.col("cutoff") - F.col("day")))

    prior = F.col("day") <= F.col("cutoff")
    w1 = prior & (F.col("rd") <= W1_DAYS - 1)
    obs_usd = F.coalesce("rake_usd_obs", F.lit(0.0))

    per_player = j.groupBy("site", "player_id").agg(
        # ---- prior window: what a model may see, and the base for every comparison
        F.sum(F.when(prior, F.col("rake_used_usd") * F.col("share")))
         .alias("p_rake_usd"),
        F.sum(F.when(prior, obs_usd * F.col("share"))).alias("p_rake_usd_obs"),
        F.sum(F.when(prior, F.col("rake_used_usd") / F.col("hand_seats")))
         .alias("p_rake_usd_equal"),
        F.sum(F.when(prior, 1).otherwise(0)).alias("p_rake_rows"),
        F.sum(F.when(prior & F.col("is_obs"), 1).otherwise(0)).alias("p_rake_hands_obs"),
        F.sum(F.when(prior, F.col("inv"))).alias("p_invested_bb"),
        # ---- the last 7 days of the prior window: the "weekly" figure
        F.sum(F.when(w1, F.col("rake_used_usd") * F.col("share"))).alias("w1_rake_usd"),
        F.sum(F.when(w1, obs_usd * F.col("share"))).alias("w1_rake_usd_obs"),
        F.sum(F.when(w1, 1).otherwise(0)).alias("w1_rows"),
        F.sum(F.when(w1 & F.col("is_obs"), 1).otherwise(0)).alias("w1_hands_obs"),
        # ---- whole observation window, for the reconciliation test
        F.sum(F.col("rake_used_usd") * F.col("share")).alias("all_rake_contrib"),
        F.sum(F.col("rake_used_usd") / F.col("hand_seats")).alias("all_rake_equal"),
        F.count("*").alias("all_rows"),
    )

    if RAKE_WORK.exists():
        shutil.rmtree(RAKE_WORK)
    t = time.time()
    per_player.write.mode("overwrite").parquet(str(RAKE_WORK))
    print(f"\n  attributed and rolled up in {(time.time() - t) / 60:.1f} min -> "
          f"{RAKE_WORK}")


def check_attribution(rake, parts):
    """Regression test 1. Shares sum to 1 inside a hand, so both attribution methods must
    return the same grand total -- to the cent. A ratio that is not 1.000000 means the
    seat x hand join fanned out, which is what a duplicated hand_uid does."""
    tot = rake.select(F.sum("all_rake_contrib").alias("c"),
                      F.sum("all_rake_equal").alias("e"),
                      F.sum("all_rows").alias("h"),
                      F.sum("p_rake_usd").alias("p"),
                      F.sum("p_rake_usd_obs").alias("po")).first()
    ratio = tot["c"] / tot["e"]
    print(f"  contributed-share total {usd(tot['c'])}   ·   equal-split total "
          f"{usd(tot['e'])}   ·   ratio {ratio:.6f}  <- must be 1.000000")
    print(f"  seat-rows attributed {tot['h']:,}   ·   prior-window rake {usd(tot['p'])}, "
          f"of which {usd(tot['po'])} ({tot['po'] / tot['p']:.1%}) is MEASURED and the "
          f"rest modelled")
    if abs(ratio - 1) > 1e-6:
        print("  !! ATTRIBUTION DOES NOT RECONCILE — find the fan-out before reading a")
        print("     single dollar figure below.")
    results["attribution_check"] = {
        "contributed_total_usd": float(tot["c"]), "equal_split_total_usd": float(tot["e"]),
        "ratio": float(ratio), "seat_rows": int(tot["h"]),
        "prior_window_usd": float(tot["p"]),
        "prior_window_measured_usd": float(tot["po"]),
        "measured_share": float(tot["po"] / tot["p"]), "subset_run": bool(parts),
    }


# ======================================================================================
# PART 3 -- the headline
# ======================================================================================
def name_segments(pdf):
    """Give the K-Means clusters their names by RULE, not by remembered index number.

    segments.py writes numbered clusters; the write-up named them by reading the centroid
    table. Hard-coding "segment 1 = the Grinder" here would break silently the moment
    anything upstream is re-run, and the number would still look plausible. So the names
    are re-derived from the same measured quantities the write-up used, and the mapping is
    printed so it can be checked against docs/segments-results.md.
    """
    prof = (pdf.dropna(subset=["ecosystem_segment"]).groupby("ecosystem_segment")
            .agg(players=("player_id", "size"), tables=("p_max_tables", "mean"),
                 hands=("p_hands", "mean"), vpip=("p_vpip", "mean"),
                 pfr=("p_pfr", "mean")).sort_index())
    names, left = {}, list(prof.index)
    for rule, col in (("Grinder", "tables"),      # multi-tabling = professional
                      ("Regular", "hands"),       # volume without the tables
                      ("Gambler", "vpip")):       # plays the most hands dealt
        pick = prof.loc[left, col].idxmax()
        names[pick] = rule
        left.remove(pick)
    names[left[0]] = "Recreational"
    print("  segment names re-derived from the centroids (check against "
          "docs/segments-results.md):")
    for s in prof.index:
        r = prof.loc[s]
        print(f"    ecosystem segment {int(s)} -> {names[s]:13s} "
              f"players {int(r['players']):6,}  tables {r['tables']:5.1f}  "
              f"hands {r['hands']:7,.0f}  vpip {r['vpip']:.2f}  pfr {r['pfr']:.2f}")
    results["segment_names"] = {str(int(k)): v for k, v in names.items()}
    return names


def slice_table(pdf, by, label):
    """Weekly rake, expected rake at risk, and who it belongs to -- summed over one cut.

    Both cuts are mandatory, not decorative: lapse rate by segment x venue spans
    30.9%-87.2%, so a single pooled figure averages populations that do not belong in the
    same average.
    """
    pdf = pdf.assign(_flag=(pdf.risk_cal >= AT_RISK_THRESHOLD).astype(int))
    g = pdf.groupby(by, dropna=False)
    t = pd.DataFrame({
        "players": g.size(),
        "at_risk": g["_flag"].sum(),
        "lapsed_actual": g["lapsed"].sum(),
        "weekly_rake": g["weekly_rake_usd"].sum(),
        "weekly_rake_measured": g["weekly_rake_usd_obs"].sum(),
        "expected_at_risk": g["expected_rake_at_risk"].sum(),
        "mean_risk": g["risk_cal"].mean(),
    })
    t["lapse_rate"] = t["lapsed_actual"] / t["players"]
    t["share_of_risk"] = t["expected_at_risk"] / t["expected_at_risk"].sum()
    t["at_risk_rate"] = t["expected_at_risk"] / t["weekly_rake"]
    t["measured_share"] = t["weekly_rake_measured"] / t["weekly_rake"]
    t["usd_per_player"] = t["weekly_rake"] / t["players"]
    t = t.sort_values("expected_at_risk", ascending=False)

    print(f"\n  {label}")
    print(f"  {'':14s} {'players':>8s} {'at-risk':>8s} {'lapsed':>7s} {'weekly $':>12s} "
          f"{'$/player':>9s} {'$ at risk':>12s} {'% of $':>7s} {'measured':>9s}")
    for k, r in t.iterrows():
        print(f"  {str(k):14s} {int(r.players):8,} {int(r.at_risk):8,} "
              f"{r.lapse_rate:6.1%} {r.weekly_rake:12,.2f} {r.usd_per_player:9.3f} "
              f"{r.expected_at_risk:12,.2f} {r.share_of_risk:6.1%} "
              f"{r.measured_share:8.1%}")
    return t


def ranking_comparison(pdf, label):
    """Does weighting by money change who you would actually call?

    Three lists of the same length -- ranked by expected loss, by risk alone, and by money
    alone -- compared on the dollars they reach and on how many names they share. If the
    lists turn out nearly identical that is a finding too, and a cheaper operating model.

    `realised $` uses the actual outcome: the weekly rake of the contacted players who did
    in fact go quiet. On the held-out fold that is an honest out-of-sample figure.
    """
    k = max(1, int(round(CONTACT_BUDGET * len(pdf))))
    total_exp = pdf["expected_rake_at_risk"].sum()
    lists = {
        "expected loss (risk x $)": pdf.nlargest(k, "expected_rake_at_risk"),
        "risk alone": pdf.nlargest(k, "risk_cal"),
        "weekly rake alone": pdf.nlargest(k, "weekly_rake_usd"),
    }
    base = set(lists["expected loss (risk x $)"].index)
    print(f"\n  {label} — contact budget {CONTACT_BUDGET:.0%} = {k:,} of {len(pdf):,} "
          f"players")
    print(f"  {'ranked by':26s} {'$ at risk reached':>18s} {'% of total':>11s} "
          f"{'realised $ lost':>16s} {'lapsers':>8s} {'precision':>10s} {'overlap':>8s}")
    out = {}
    for name, sel in lists.items():
        reached = sel["expected_rake_at_risk"].sum()
        realised = (sel["weekly_rake_usd"] * sel["lapsed"]).sum()
        overlap = len(base & set(sel.index)) / k
        out[name] = {"contacted": k, "expected_at_risk_reached": float(reached),
                     "share_of_total": float(reached / total_exp),
                     "realised_lost_rake": float(realised),
                     "true_lapsers": int(sel["lapsed"].sum()),
                     "precision": float(sel["lapsed"].mean()),
                     "overlap_with_expected_loss": float(overlap)}
        print(f"  {name:26s} {reached:18,.2f} {reached / total_exp:10.1%} "
              f"{realised:16,.2f} {int(sel['lapsed'].sum()):8,} "
              f"{sel['lapsed'].mean():9.1%} {overlap:7.1%}")
    return out


def main():
    quick = "--quick" in sys.argv
    parts = int(sys.argv[sys.argv.index("--parts") + 1]) if "--parts" in sys.argv else None
    global RAKE_WORK, OUT_TABLE, OUT_JSON
    if parts:
        RAKE_WORK = Path(f"{RAKE_WORK}_sample")
        OUT_TABLE = Path(f"{OUT_TABLE}_sample")
        OUT_JSON = OUT_JSON.with_name("rake-at-risk-results-sample.json")
        print(f"SUBSET RUN: first {parts} seat-row parts -> {RAKE_WORK}, {OUT_TABLE}")
        print("A subset is NOT representative — Silver parts are batched by venue/stake "
              "folder, so the money numbers below are structurally wrong. Smoke test only.")

    t0 = time.time()
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")

    # ---- part 1
    scored = calibrate(spark, quick)

    # ---- part 2
    excluded = venue_coverage(spark)
    priced, window = price_hands(spark, excluded)
    if "--rebuild-rake" in sys.argv or not RAKE_WORK.exists():
        build_player_rake(spark, parts, priced, window)
    else:
        print(f"\n  reusing {RAKE_WORK} (pass --rebuild-rake to redo the attribution)")
    rake = spark.read.parquet(str(RAKE_WORK)).cache()

    hr("PART 2d · do the attributions reconcile?")
    check_attribution(rake, parts)

    # ---- part 3
    hr("PART 3 · THE HEADLINE — weekly rake x calibrated risk")
    lapse = spark.read.parquet(str(LAPSE)).select(
        "site", "venue", "player_id", "p_hands", "p_hands_w1", "p_money_hands",
        "p_recency_days", "p_vpip", "p_pfr", "p_max_tables", "p_max_stake", "lapsed")
    seg = spark.read.parquet(str(SEGMENTS)).select(
        "site", "player_id", "style_segment", "ecosystem_segment")

    j = (lapse.join(scored.drop("label"), ["site", "player_id"])
         .join(rake, ["site", "player_id"], "inner")     # inner: money venues only
         .join(seg, ["site", "player_id"], "left"))

    # ---- regression test 2: the join must reproduce Fork A's own hand counts, player by
    # player. Catches a fan-out from the non-unique hand_uid AND any drift in the per-venue
    # cutoff between this script and features_lapse.py. Meaningless on a subset run, where
    # only a slice of the seat-rows is read on purpose.
    bad = j.filter(F.col("p_rake_rows") != F.col("p_hands"))
    n_bad = bad.count()
    if parts:
        print(f"  prior-window hand counts vs gold/player_lapse: {n_bad:,} differ — "
              f"expected on a subset run, which reads {parts} of 96 seat-row parts")
    else:
        print(f"  prior-window hand counts vs gold/player_lapse: {n_bad:,} mismatches "
              f"{'<- MUST be 0 (fan-out or a cutoff drift)' if n_bad else '(clean)'}")
        if n_bad:
            bad.select("site", "player_id", "p_hands", "p_rake_rows").show(
                5, truncate=False)
    results["hand_count_check"] = {"mismatched_players": int(n_bad),
                                   "subset_run": bool(parts)}

    j = (j
         .withColumn("weekly_rake_usd", F.coalesce("w1_rake_usd", F.lit(0.0)))
         .withColumn("weekly_rake_usd_obs", F.coalesce("w1_rake_usd_obs", F.lit(0.0)))
         # ANSI SQL is on in Spark 4, so a player whose whole prior window folded pre-flop
         # (rake $0.00, legitimately) would abort the job on a divide-by-zero.
         .withColumn("measured_share",
                     F.when(F.col("p_rake_usd") > 0,
                            F.col("p_rake_usd_obs") / F.col("p_rake_usd")))
         .withColumn("expected_rake_at_risk",
                     F.col("risk_cal") * F.col("weekly_rake_usd"))
         # The discarded method, kept for the comparison that justifies discarding it:
         # each player's own observed rake per 100 hands, scaled to all their hands.
         .withColumn("naive_rate_per_100",
                     F.when(F.col("p_rake_hands_obs") >= MIN_MONEY_HANDS,
                            F.col("p_rake_usd_obs") / F.col("p_rake_hands_obs") * 100))
         .withColumn("naive_weekly_usd",
                     F.col("naive_rate_per_100") * F.col("p_hands_w1") / 100)
         .cache())

    n = j.count()
    print(f"  players on the money venues: {n:,} of Fork A's 90,469 "
          f"({n / 90469:.1%}) — the rest are iPoker, which reconciles on none of its")
    print("  6.0 M hands. Nothing is imputed for them: an unmeasured spend is not zero.")

    pdf = j.toPandas().set_index(["site", "player_id"])       # ~77 k rows: Gold grain
    names = name_segments(pdf.reset_index())
    pdf["segment"] = pdf["ecosystem_segment"].map(names).fillna("unsegmented")

    weekly = pdf.weekly_rake_usd.sum()
    weekly_obs = pdf.weekly_rake_usd_obs.sum()
    exp_risk = pdf.expected_rake_at_risk.sum()
    flagged = pdf[pdf.risk_cal >= AT_RISK_THRESHOLD]
    n_flag, flagged_rake = len(flagged), flagged.weekly_rake_usd.sum()
    print(f"\n  weekly rake across these players        : {usd(weekly)}   "
          f"({usd(weekly_obs)} = {weekly_obs / weekly:.1%} measured, the rest modelled)")
    print(f"  expected rake at risk = Σ P x weekly $  : {usd(exp_risk)}   "
          f"({exp_risk / weekly:.1%} of the weekly total)")
    print(f"  players at risk (calibrated P >= {AT_RISK_THRESHOLD})  : {n_flag:,} of "
          f"{n:,} ({n_flag / n:.1%}), holding {usd(flagged_rake)} of weekly rake")

    seg_t = slice_table(pdf, "segment", "by SEGMENT (ecosystem view, the money venues)")
    ven_t = slice_table(pdf, "venue", "by VENUE (lapse alone spans 31%-87% across these)")
    rank = ranking_comparison(pdf, "WHO TO CALL · all money-venue players")
    rank_te = ranking_comparison(pdf[pdf.fold == "test"],
                                "WHO TO CALL · held-out fold only (labels never trained on)")

    # ---- sensitivity 1: which probability the dollars are multiplied by
    sens_p = {nm: float((pdf[col].fillna(0) * pdf.weekly_rake_usd).sum())
              for nm, col in (("shipped weighted scores", "risk_weighted"),
                              ("unweighted refit", "risk_uncal"),
                              ("chosen (calibrated)", "risk_cal"))}
    print("\n  sensitivity · which probability the dollars are multiplied by:")
    for nm, v in sens_p.items():
        print(f"    {nm:26s} {usd(v):>14s}   "
              f"({v / sens_p['chosen (calibrated)'] - 1:+.1%} vs chosen)")

    # ---- sensitivity 2: the estimator versus the naive per-player scaling
    nv = pdf.dropna(subset=["naive_weekly_usd"])
    print(f"\n  sensitivity · the DISCARDED method, on the {len(nv):,} players who had "
          f">= {MIN_MONEY_HANDS} measured hands:")
    print(f"    conditional-mean estimator  {usd(nv.weekly_rake_usd.sum()):>16s}")
    print(f"    naive per-player scaling    {usd(nv.naive_weekly_usd.sum()):>16s}   "
          f"(x{nv.naive_weekly_usd.sum() / nv.weekly_rake_usd.sum():.2f})")
    print("    The naive figure charges every hand at the rate of the pots big enough to")
    print("    reconcile, and ignores that most hands never see a flop and rake nothing.")
    sens_est = {"players": int(len(nv)),
                "estimator_usd": float(nv.weekly_rake_usd.sum()),
                "naive_usd": float(nv.naive_weekly_usd.sum()),
                "inflation_x": float(nv.naive_weekly_usd.sum() / nv.weekly_rake_usd.sum())}

    # ---- sensitivity 3: only the venues where most hands are actually measured
    hi = [v["site"] for v in results.get("money_coverage_by_venue", [])
          if v["coverage"] >= HIGH_COVERAGE]
    sub = pdf[pdf.index.get_level_values("site").isin(hi)]
    sens_hi = {"venues": hi, "players": int(len(sub))}
    if len(sub):
        sens_hi |= {"weekly_usd": float(sub.weekly_rake_usd.sum()),
                    "measured_share": float(sub.weekly_rake_usd_obs.sum()
                                            / sub.weekly_rake_usd.sum()),
                    "expected_at_risk_usd": float(sub.expected_rake_at_risk.sum()),
                    "at_risk_rate": float(sub.expected_rake_at_risk.sum()
                                          / sub.weekly_rake_usd.sum())}
        print(f"\n  sensitivity · the high-coverage venues only {hi} — "
              f"{sens_hi['measured_share']:.0%} of their")
        print(f"    weekly rake is measured, and {sens_hi['at_risk_rate']:.1%} of it is "
              f"at risk, against {exp_risk / weekly:.1%} pooled. If those two agree,")
        print("    the modelled venues are not driving the headline.")

    # ---- sensitivity 4: the attribution rule itself
    eq, co = pdf.p_rake_usd_equal.sum(), pdf.p_rake_usd.sum()
    print(f"\n  sensitivity · attribution rule over the prior window: contributed-share "
          f"{usd(co)}\n    vs equal-split {usd(eq)} — ratio {co / eq:.4f} (they must "
          f"agree over ALL players;")
    print("    a gap here is the rule moving rake between players, which is its job.)")

    results["headline"] = {
        "players": int(n),
        "weekly_rake_usd": float(weekly),
        "weekly_rake_measured_usd": float(weekly_obs),
        "measured_share": float(weekly_obs / weekly),
        "expected_rake_at_risk_usd": float(exp_risk),
        "at_risk_share_of_weekly": float(exp_risk / weekly),
        "at_risk_players": int(n_flag),
        "at_risk_players_weekly_rake_usd": float(flagged_rake),
        "top_budget_reach_usd": rank["expected loss (risk x $)"]["expected_at_risk_reached"],
        "top_budget_players": rank["expected loss (risk x $)"]["contacted"],
        "recreational_share_of_risk": (float(seg_t.loc["Recreational", "share_of_risk"])
                                      if "Recreational" in seg_t.index else None),
    }
    results["by_segment"] = json.loads(seg_t.reset_index().to_json(orient="records"))
    results["by_venue"] = json.loads(ven_t.reset_index().to_json(orient="records"))
    results["ranking_full"] = rank
    results["ranking_test_fold"] = rank_te
    results["sensitivity_probability"] = sens_p
    results["sensitivity_estimator"] = sens_est
    results["sensitivity_high_coverage"] = sens_hi
    results["sensitivity_attribution"] = {"contributed_usd": float(co),
                                          "equal_split_usd": float(eq),
                                          "ratio": float(co / eq)}

    # ---- THE SENTENCE. Everything above exists to make this line quotable.
    rec = results["headline"]["recreational_share_of_risk"]
    z = results["headline"]["top_budget_reach_usd"]
    hr("THE HEADLINE SENTENCE")
    print(f"  {usd(exp_risk)} of next week's rake sits at risk across {n:,} players on "
          f"{len(ven_t)} venues —")
    print(f"  {rec:.0%} of it in the recreational segment — and contacting the top "
          f"{CONTACT_BUDGET:.0%} by expected loss")
    print(f"  ({rank['expected loss (risk x $)']['contacted']:,} players) puts "
          f"{usd(z)} of it in reach.")
    print(f"\n  Small print that travels with it: weekly rake is {usd(weekly)}, of which "
          f"{weekly_obs / weekly:.0%}")
    print("  is measured and the rest estimated from what comparable hands actually paid;")
    print(f"  {n_flag:,} players are more likely than not to go quiet and hold "
          f"{usd(flagged_rake)} of it;")
    print("  iPoker is excluded entirely (rake reconciles on none of its 6.0 M hands);")
    print("  July 2009, US dollars, 25NL-1000NL — no currency conversion, no inflation.")

    # ---- the deliverable table
    out = (j.withColumn("segment_id", F.col("ecosystem_segment"))
           .select("site", "venue", "player_id", "fold", "lapsed", "segment_id",
                   "style_segment",
                   F.col("p_hands").cast("int").alias("p_hands"),
                   F.col("p_hands_w1").cast("int").alias("p_hands_w1"),
                   F.col("p_recency_days").cast("int").alias("p_recency_days"),
                   F.col("p_rake_hands_obs").cast("int").alias("p_rake_hands_measured"),
                   F.col("w1_hands_obs").cast("int").alias("w1_hands_measured"),
                   F.round("measured_share", 4).alias("measured_share"),
                   F.round("p_rake_usd", 4).alias("p_rake_usd"),
                   F.round("p_rake_usd_obs", 4).alias("p_rake_usd_measured"),
                   F.round("p_rake_usd_equal", 4).alias("p_rake_usd_equalsplit"),
                   F.round("weekly_rake_usd", 4).alias("weekly_rake_usd"),
                   F.round("weekly_rake_usd_obs", 4).alias("weekly_rake_usd_measured"),
                   F.round("naive_weekly_usd", 4).alias("weekly_rake_usd_naive"),
                   F.round("risk_weighted", 4).alias("risk_weighted"),
                   F.round("risk_uncal", 4).alias("risk_uncal"),
                   F.round("risk_cal", 4).alias("risk_cal"),
                   F.round("expected_rake_at_risk", 4).alias("expected_rake_at_risk")))
    out.write.mode("overwrite").parquet(str(OUT_TABLE))
    print(f"\n  wrote {OUT_TABLE} — {n:,} players, {len(out.columns)} columns")
    print("  the 12 players with the most rake at risk (the top of the call list):")
    (out.orderBy(F.desc("expected_rake_at_risk"))
        .select("site", "segment_id", "p_hands", "p_hands_w1", "p_rake_hands_measured",
                "measured_share", "weekly_rake_usd", "risk_cal",
                "expected_rake_at_risk", "lapsed")
        .show(12, truncate=False))

    results["seed"] = SEED
    results["quick_mode"] = quick
    results["subset_parts"] = parts
    results["runtime_seconds"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"  wrote {OUT_JSON}\n  total {(time.time() - t0) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()
