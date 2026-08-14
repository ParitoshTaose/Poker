"""Fork A, part 2 of 2 -- the models, in PySpark MLlib.

    reads   data/gold/player_lapse        (built by src/features_lapse.py)
    writes  docs/fork-a-results.json      every number this prints
            data/gold/lapse_scores        one risk score per active player

This is Step 8 of docs/execution-roadmap.html for the winning fork: two algorithms on
one question, scored against a baseline, with the threshold framed as a business
decision. It replaces the throwaway scikit-learn prototype in src/bakeoff.py -- that
script existed to CHOOSE the question, this one answers it.

THE QUESTION
------------
For every player still active on a venue at the cutoff, will they play zero hands in
the seven days that follow? The operator earns rake only while people keep sitting
down, so a ranked list of who is drifting is something a retention team can work
through on a Monday morning.

    Honesty, every time this is presented: 26 days of data cannot show churn. This is
    SHORT-HORIZON LAPSE -- a week of silence, which is an early warning, not a proof
    that anybody has left for good.

WHAT IS BEING TESTED, AND AGAINST WHAT
--------------------------------------
Three baselines, because a score with nothing beside it is not evidence:

    majority class    flag everyone. Recall 1.0 by construction, precision = prevalence.
                      This is what "accuracy" flatters, and why accuracy is not used here.
    recency rank      sort by days-since-last-seen, descending. No model, no training,
                      no data engineering -- a retention manager can do this in Excel in
                      four minutes. If MLlib cannot beat it, MLlib has not earned its
                      place in the architecture, and the report should say so.
    recency rule      the same idea as a hard rule ("quiet for >= N days => flag"),
                      scored as a confusion matrix, because that is the form an
                      operations team would actually deploy.

Two feature sets, because two of the six venues do not record everything:

    CORE      behaviour, volume, recency and trend. Present on all six venues.
    EXTENDED  CORE + stack depth, multi-tabling and realised money. iPoker records
              none of the three, so this is a FIVE-venue model covering fewer players.
              Reported separately rather than imputed -- a median stack on a venue
              that recorded none is a number nobody measured.

Run from the repo root:

    .venv/bin/python src/fork_a.py
    .venv/bin/python src/fork_a.py --quick     # fewer trees, for a smoke test
"""
import json
import os
import sys
import time
from pathlib import Path

# See features_lapse.py: pin both ends of the bridge before pyspark is imported, or
# Spark launches its workers under Apple's Python 3.9, which cannot import pyspark 4.2.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import numpy as np  # noqa: E402
from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.classification import (  # noqa: E402
    GBTClassifier,
    LogisticRegression,
    RandomForestClassifier,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator  # noqa: E402
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler  # noqa: E402
from pyspark.ml.functions import vector_to_array  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAPSE = ROOT / "data" / "gold" / "player_lapse"
PLAYERS = ROOT / "data" / "gold" / "player_features"
SCORES = ROOT / "data" / "gold" / "lapse_scores"
OUT = ROOT / "docs" / "fork-a-results.json"
SCRATCH = ROOT / "data" / "_work" / "spark-scratch"

SEED = 42
TEST_FRACTION = 25          # percent, split on a hash of the player key

# Present on all six venues. Everything here is computed from days <= cutoff by
# features_lapse.py, so a model using them could have been run on the cutoff day.
CORE = [
    # volume and shape of activity
    "p_hands", "p_days_active", "p_active_minutes",
    "p_hands_per_active_day", "p_hands_per_minute", "p_active_day_share",
    # recency and tenure -- the honest heart of the model
    "p_recency_days", "p_tenure_days", "p_prior_span_days",
    # trend: is their volume decaying into the cutoff, or ramping up?
    "p_hands_w1", "p_hands_w2", "p_hands_last3",
    "p_late_share", "p_w1_w2_ratio", "p_last3_share", "p_weekend_share",
    # playing style
    "p_vpip", "p_pfr", "p_fold_rate", "p_wtsd", "p_vpip_pfr_gap",
    "p_aggression_factor", "p_avg_invested_bb", "p_avg_seats",
    "p_bb_vpip_rate", "p_btn_share",
    # stake mobility
    "p_distinct_stakes", "p_max_stake", "p_min_stake", "p_stake_range",
]

# Recorded on five of the six venues. iPoker writes 'HandHQ' into every table field,
# `inf` into every starting stack, and reconciles winnings on none of its 6.0 M hands
# -- behaviourally complete, economically blind. So this model is run on the five
# venues that do record them, and its coverage is quoted every time it is.
EXTENDED_ONLY = [
    "p_avg_stack_bb", "p_min_stack_bb",
    "p_max_tables", "p_avg_tables", "p_distinct_tables",
    "p_money_hands", "p_money_coverage", "p_bb_per_100",
]

# The old, leaky feature set: the lifetime rates from gold/player_features, which are
# averaged over a player's whole history INCLUDING the week being predicted. Kept only
# to re-run the bake-off's ablation and show the fix worked.
LIFETIME = [
    "vpip", "pfr", "fold_rate", "wtsd", "aggression_factor", "vpip_pfr_gap",
    "avg_invested_bb", "distinct_stakes", "max_stake", "btn_share", "bb_vpip_rate",
]

# Contact budgets to report. A retention team has capacity, not an appetite for
# "flag everyone" -- these are the rows that turn a metric into a decision.
BUDGETS = [0.05, 0.10, 0.20, 0.25, 0.50]

results: dict = {}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def session():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return (SparkSession.builder
            .appName("ecosystem-engine-fork-a")
            .master("local[8]")
            .config("spark.driver.memory", "9g")
            .config("spark.sql.shuffle.partitions", "48")   # 100k rows, not 116 M
            .config("spark.local.dir", str(SCRATCH.resolve()))
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


# ----------------------------------------------------------------------------------
# scoring helpers
# ----------------------------------------------------------------------------------
def auc(df, score_col, label_col="label"):
    """ROC-AUC and PR-AUC. MLlib's evaluator takes a raw score or a probability
    vector; either way it ranks, which is what both metrics measure."""
    ev = lambda m: BinaryClassificationEvaluator(  # noqa: E731
        labelCol=label_col, rawPredictionCol=score_col, metricName=m).evaluate(df)
    return {"roc_auc": float(ev("areaUnderROC")), "pr_auc": float(ev("areaUnderPR"))}


def confusion(y, pred):
    """Counts first, rates second -- the counts are what a retention budget is built
    from, and they make the rates auditable rather than asserted."""
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "accuracy": (tp + tn) / len(y),
    }


def budget_curve(y, score, prevalence):
    """'If the team can only call N% of players, how many of the leavers do we catch?'

    lift = precision within the contacted group / the base rate. Lift 1.0 means the
    model has sorted the list no better than shuffling it.
    """
    order = np.argsort(-score)          # highest risk first
    ys = y[order]
    out = []
    for b in BUDGETS:
        k = max(1, int(round(b * len(ys))))
        hit = int(ys[:k].sum())
        prec = hit / k
        out.append({
            "budget_pct": b, "contacted": k, "true_lapsers_caught": hit,
            "precision": prec, "recall": hit / int(y.sum()),
            "lift_vs_random": prec / prevalence,
        })
    return out


def best_f1(y, score):
    """Sweep the cut-point. The default 0.5 is an arbitrary inheritance from a
    balanced world; on a 62/38 problem the operating point is a choice to be made
    and defended, so the sweep is reported rather than hidden inside a default."""
    grid = np.linspace(0.05, 0.95, 91)
    rows = [(t, confusion(y, (score >= t).astype(int))) for t in grid]
    t, c = max(rows, key=lambda r: r[1]["f1"])
    return {"threshold": float(t), **c}


# ----------------------------------------------------------------------------------
# the model pipelines
# ----------------------------------------------------------------------------------
def build_pipeline(kind, numeric, indicators, dummies, quick):
    """One assembler, three estimators, one decision recorded per stage.

    Imputer(median) runs on the numeric columns only. It is safe HERE and would not
    be safe on the extended set's venue-shaped nulls -- which is exactly why the
    extended model drops the venues that record nothing instead of filling them in.
    Every column imputed also carries a _missing indicator, so 'this was filled in'
    stays visible to the model instead of being laundered into a real value.

    Scaling is applied for logistic regression, which is distance-based in its
    penalty term and would otherwise let p_hands (thousands) drown p_vpip (0-1).
    Trees split on rank, so scaling them is a no-op -- included for LR only, and
    that asymmetry is the point of running both.
    """
    feats = [f"{c}_i" for c in numeric] + indicators + dummies
    stages = [
        Imputer(strategy="median", inputCols=numeric,
                outputCols=[f"{c}_i" for c in numeric]),
        VectorAssembler(inputCols=feats, outputCol="v", handleInvalid="keep"),
    ]
    if kind == "logistic":
        stages += [
            StandardScaler(inputCol="v", outputCol="features",
                           withMean=True, withStd=True),
            LogisticRegression(featuresCol="features", labelCol="label",
                               weightCol="w", maxIter=100, regParam=0.01,
                               elasticNetParam=0.0),
        ]
    elif kind == "gbt":
        stages += [GBTClassifier(featuresCol="v", labelCol="label", weightCol="w",
                                 maxIter=30 if quick else 120, maxDepth=5,
                                 stepSize=0.1, seed=SEED)]
    elif kind == "rf":
        stages += [RandomForestClassifier(featuresCol="v", labelCol="label",
                                          weightCol="w",
                                          numTrees=40 if quick else 150,
                                          maxDepth=8, seed=SEED)]
    return Pipeline(stages=stages), feats


def evaluate(model, test, name, prevalence):
    """Ranking metrics from MLlib, decision metrics from the collected scores.

    The collect is deliberate and safe: this is the Gold grain -- tens of thousands of
    rows, two columns. The rule it must never break is collecting SILVER.
    """
    pred = model.transform(test).withColumn(
        "p1", vector_to_array("probability")[1]).cache()
    m = auc(pred, "probability")

    pdf = pred.select("label", "p1").toPandas()
    y, s = pdf["label"].to_numpy(), pdf["p1"].to_numpy()
    m["at_0.5"] = confusion(y, (s >= 0.5).astype(int))
    m["best_f1"] = best_f1(y, s)
    m["budget_curve"] = budget_curve(y, s, prevalence)

    c = m["at_0.5"]
    print(f"  {name:22s} roc-auc {m['roc_auc']:.4f}  pr-auc {m['pr_auc']:.4f}"
          f"   @0.5: prec {c['precision']:.3f} rec {c['recall']:.3f} f1 {c['f1']:.3f}")
    pred.unpersist()
    return m


def run_variant(spark, df, label, feature_cols, dummies, quick):
    """Train and score every algorithm on one feature set, plus its baselines."""
    hr(f"MODELS · {label}")
    train = df.filter("fold = 'train'").cache()
    test = df.filter("fold = 'test'").cache()
    n_tr, n_te = train.count(), test.count()
    prev_tr = train.select(F.avg("label")).first()[0]
    prev_te = test.select(F.avg("label")).first()[0]
    print(f"  train {n_tr:,} (lapsed {prev_tr:.3%})   test {n_te:,} "
          f"(lapsed {prev_te:.3%})   {len(feature_cols) + len(dummies)} features")

    # Balanced class weights, computed on TRAIN only -- the test set's prevalence is
    # not something a model deployed at the cutoff would know.
    w1 = 1.0 / (2 * prev_tr)
    w0 = 1.0 / (2 * (1 - prev_tr))
    train = train.withColumn("w", F.when(F.col("label") == 1, w1).otherwise(w0))
    test = test.withColumn("w", F.lit(1.0))

    out = {"n_train": n_tr, "n_test": n_te, "prevalence_train": prev_tr,
           "prevalence_test": prev_te, "features": feature_cols + dummies,
           "baselines": {}, "models": {}}

    # ---- baseline 1: flag everyone
    pdf = test.select("label", "p_recency_days").toPandas()
    y = pdf["label"].to_numpy()
    out["baselines"]["majority_class"] = {
        "rule": "predict lapsed for every player",
        "roc_auc": 0.5, "pr_auc": float(prev_te),
        "at_0.5": confusion(y, np.ones_like(y)),
    }
    b = out["baselines"]["majority_class"]
    print(f"  {'majority class':22s} roc-auc 0.5000  pr-auc {b['pr_auc']:.4f}"
          f"   @all: prec {b['at_0.5']['precision']:.3f} rec 1.000 "
          f"f1 {b['at_0.5']['f1']:.3f}")

    # ---- baseline 2: sort the spreadsheet by days-since-last-seen
    rec = pdf["p_recency_days"].to_numpy().astype(float)
    r = auc(test.withColumn("_r", F.col("p_recency_days").cast("double")), "_r")
    r["budget_curve"] = budget_curve(y, rec, prev_te)
    r["rule"] = "rank by p_recency_days, descending -- no model, no training"
    # ties are everywhere (recency is an integer 0-19), so the rule form matters too
    r["hard_rules"] = {
        f">= {n} days quiet": confusion(y, (rec >= n).astype(int)) for n in (1, 2, 3, 5)
    }
    out["baselines"]["recency_rank"] = r
    print(f"  {'recency rank (free)':22s} roc-auc {r['roc_auc']:.4f}  "
          f"pr-auc {r['pr_auc']:.4f}   <- the number MLlib has to beat")
    for k, c in r["hard_rules"].items():
        print(f"      rule {k:16s} prec {c['precision']:.3f} rec {c['recall']:.3f} "
              f"f1 {c['f1']:.3f}")

    # ---- the models
    numeric = feature_cols
    # Only the indicators belonging to THIS feature set. Carrying p_avg_stack_bb_missing
    # into the CORE model would smuggle "this player is on iPoker" in twice.
    indicators = [f"{c}_missing" for c in numeric if f"{c}_missing" in df.columns]
    for kind, name in (("logistic", "logistic_regression"),
                       ("gbt", "gbt_classifier"),
                       ("rf", "random_forest")):
        pipe, feats = build_pipeline(kind, numeric, indicators, dummies, quick)
        t = time.time()
        model = pipe.fit(train)
        fit_s = round(time.time() - t, 1)
        m = evaluate(model, test, name, prev_te)
        m["fit_seconds"] = fit_s

        est = model.stages[-1]
        if kind == "logistic":
            # Standardised, so the magnitudes are comparable and the sign reads
            # directly: positive => pushes a player towards lapsing.
            coef = sorted(zip(feats, est.coefficients.toArray()),
                          key=lambda kv: -abs(kv[1]))
            m["top_coefficients"] = [{"feature": f, "coef": float(c)}
                                     for f, c in coef[:12]]
            print("      strongest standardised coefficients (+ => more likely to lapse):")
            for f, c in coef[:8]:
                print(f"        {f:26s} {c:+.3f}")
        else:
            imp = sorted(zip(feats, est.featureImportances.toArray()),
                         key=lambda kv: -kv[1])
            m["top_importances"] = [{"feature": f, "importance": float(i)}
                                    for f, i in imp[:12]]
            if kind == "gbt":
                print("      strongest GBT feature importances:")
                for f, i in imp[:8]:
                    print(f"        {f:26s} {i:.4f}")
                # per-venue, because a pooled number can hide a venue where it fails
                pred = model.transform(test).withColumn(
                    "p1", vector_to_array("probability")[1]).cache()
                per = {}
                for row in (pred.select("site", "label", "p1")
                            .groupBy("site").count().collect()):
                    sub = pred.filter(F.col("site") == row["site"])
                    a = auc(sub, "probability")
                    per[row["site"]] = {"players": row["count"], **a,
                                        "prevalence": float(
                                            sub.select(F.avg("label")).first()[0])}
                m["per_venue"] = per
                print("      per-venue ROC-AUC (a pooled number can hide a bad venue):")
                for s, v in sorted(per.items(), key=lambda kv: -kv[1]["players"]):
                    print(f"        {s:5s} n={v['players']:6,}  roc {v['roc_auc']:.4f}"
                          f"  pr {v['pr_auc']:.4f}  prevalence {v['prevalence']:.3f}")
                out["_best_model"] = model
        out["models"][name] = m

    train.unpersist()
    return out


# ----------------------------------------------------------------------------------
def leakage_recheck(spark, df, dummies, quick):
    """The bake-off's ablation, re-run now that the honest features exist.

    Step 5 measured, on the SAME GBM: prior-window features 0.798 PR-AUC, lifetime
    rates alone 0.886, both 0.927 -- the leaky set out-predicting the legitimate one,
    which is the signature of leakage rather than of behaviour. The question this
    answers is whether the new prior-window table closes that gap honestly, or merely
    hides it.
    """
    hr("LEAKAGE RE-CHECK · the ablation that made Fork A conditional")
    lifetime = (spark.read.parquet(str(PLAYERS))
                .select("site", "player_id", *LIFETIME))
    j = df.join(lifetime, ["site", "player_id"], "left").cache()

    sets = {
        "old floor (hands_prior + tenure)": ["p_hands", "p_tenure_days"],
        "LIFETIME rates only (leaky)": LIFETIME,
        "PRIOR-WINDOW only (honest, this build)": CORE,
    }
    out = {}
    for name, cols in sets.items():
        train = j.filter("fold = 'train'")
        prev = train.select(F.avg("label")).first()[0]
        train = train.withColumn("w", F.when(F.col("label") == 1, 1 / (2 * prev))
                                 .otherwise(1 / (2 * (1 - prev))))
        test = j.filter("fold = 'test'").withColumn("w", F.lit(1.0))
        pipe, _ = build_pipeline("gbt", cols, [], [], quick)
        model = pipe.fit(train)
        m = auc(model.transform(test), "probability")
        out[name] = {"n_features": len(cols), **m}
        print(f"  {name:42s} ({len(cols):2d} feats)  "
              f"roc {m['roc_auc']:.4f}  pr-auc {m['pr_auc']:.4f}")
    j.unpersist()
    return out


# ----------------------------------------------------------------------------------
def main():
    quick = "--quick" in sys.argv
    t0 = time.time()
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")

    hr("FORK A · short-horizon lapse classification, PySpark MLlib")
    df = spark.read.parquet(str(LAPSE))

    # Split on a hash of the player key, not randomSplit: the same player lands in the
    # same fold on every run and on every machine, so a metric that moves means the
    # model moved. One row per (site, player_id) already, so this is also the
    # "no player in both folds" rule the roadmap asks for -- IDs are never merged
    # across venues, because the same human on two sites is two different codes.
    df = (df.withColumn(
        "fold",
        F.when(F.pmod(F.hash(F.concat_ws(":", "site", "player_id", F.lit(SEED))),
                      F.lit(100)) < TEST_FRACTION, "test").otherwise("train"))
        .withColumnRenamed("lapsed", "label"))

    n = df.count()
    prev = df.select(F.avg("label")).first()[0]
    print(f"  {n:,} players scored at their venue's own cutoff · "
          f"{prev:.2%} played zero hands in the following 7 days")

    # MLlib's Imputer accepts only DoubleType/FloatType -- an int column fails the
    # schema check rather than being promoted, which is a five-minute mystery the
    # first time. Cast once, here, so every downstream stage sees one numeric type.
    for c in CORE + EXTENDED_ONLY:
        df = df.withColumn(c, F.col(c).cast("double"))

    # Missing indicators for anything meaningfully absent, BEFORE imputation, so the
    # fact of the gap survives into the model as information in its own right.
    counts = df.select([F.avg(F.col(c).isNull().cast("double")).alias(c)
                        for c in CORE + EXTENDED_ONLY]).first().asDict()
    gappy = [c for c, r in counts.items() if r > 0.005]
    print(f"  columns >0.5% null, given a _missing indicator: {gappy or 'none'}")
    for c in gappy:
        df = df.withColumn(f"{c}_missing", F.col(c).isNull().cast("double"))

    # One-hot the venue by hand rather than with StringIndexer + OneHotEncoder: six
    # values, and this way every coefficient printed below carries the venue's name
    # instead of an index nobody can read off a slide.
    sites = sorted(r["site"] for r in df.select("site").distinct().collect())
    ref = "PS"     # the largest venue, 38% of the dataset -> the natural reference
    dummies = []
    for s in sites:
        if s == ref:
            continue
        df = df.withColumn(f"site_{s}", (F.col("site") == s).cast("double"))
        dummies.append(f"site_{s}")
    print(f"  venues: {sites}  (one-hot, reference = {ref})")
    df = df.cache()

    results["population"] = {
        "players": n, "prevalence_lapsed": prev, "venues": sites,
        "reference_venue": ref, "test_fraction": TEST_FRACTION / 100,
        "null_rates": {c: r for c, r in counts.items() if r > 0},
    }

    # ---- variant 1: CORE, all six venues
    core_res = run_variant(spark, df, "CORE features · all six venues",
                                      CORE, dummies, quick)
    best_model = core_res.pop("_best_model", None)
    results["core_all_venues"] = core_res

    # ---- variant 2: EXTENDED, only the venues that record stacks/tables/money
    keep = [r["site"] for r in
            df.groupBy("site").agg(F.avg(F.col("tables_recorded").cast("int")).alias("t"),
                                   F.avg(F.col("stacks_recorded").cast("int")).alias("s"))
            .collect() if r["t"] > 0.5 and r["s"] > 0.5]
    ext = df.filter(F.col("site").isin(keep))
    ext_dummies = [d for d in dummies if d.split("_", 1)[1] in keep]
    print(f"\n  EXTENDED set runs on {keep} -- "
          f"{ext.count():,} of {n:,} players ({ext.count() / n:.1%})")
    ext_res = run_variant(spark, ext,
                             f"EXTENDED features · {len(keep)} venues only",
                             CORE + EXTENDED_ONLY, ext_dummies, quick)
    ext_res.pop("_best_model", None)
    results["extended_five_venues"] = ext_res

    # ---- the ablation that made this whole build a precondition
    results["leakage_recheck"] = leakage_recheck(spark, df, dummies, quick)

    # ---- the deliverable: a scored list, which is what a retention team receives
    if best_model is not None:
        hr("THE OUTPUT · a ranked retention list")
        scored = (best_model.transform(df)
                  .withColumn("lapse_risk", vector_to_array("probability")[1])
                  .select("site", "venue", "player_id", "fold", "label",
                          F.col("p_hands").cast("int").alias("p_hands"),
                          F.col("p_recency_days").cast("int").alias("p_recency_days"),
                          F.round("p_late_share", 3).alias("p_late_share"),
                          F.col("p_max_tables").cast("int").alias("p_max_tables"),
                          F.round("p_bb_per_100", 1).alias("p_bb_per_100"),
                          F.round("lapse_risk", 4).alias("lapse_risk")))
        scored.write.mode("overwrite").parquet(str(SCORES))
        print(f"  wrote {SCORES} -- {scored.count():,} scored players")
        print("\n  the 10 highest-risk players the model would put on the call list:")
        (scored.filter("fold = 'test'").orderBy(F.desc("lapse_risk"))
         .show(10, truncate=False))

    # ---- scorecard
    hr("SCORECARD")
    print(f"  {'model':34s} {'roc-auc':>8s} {'pr-auc':>8s} {'f1@0.5':>8s} "
          f"{'prec@10%':>9s} {'recall@10%':>11s}")
    rows = []
    for setname, res in (("core", results["core_all_venues"]),
                         ("ext5", results["extended_five_venues"])):
        b = res["baselines"]
        rows.append((f"{setname} · baseline majority", 0.5, b["majority_class"]["pr_auc"],
                     b["majority_class"]["at_0.5"]["f1"], None, None))
        rc = b["recency_rank"]
        b10 = next(x for x in rc["budget_curve"] if x["budget_pct"] == 0.10)
        rows.append((f"{setname} · baseline recency", rc["roc_auc"], rc["pr_auc"],
                     None, b10["precision"], b10["recall"]))
        for mname, m in res["models"].items():
            m10 = next(x for x in m["budget_curve"] if x["budget_pct"] == 0.10)
            rows.append((f"{setname} · {mname}", m["roc_auc"], m["pr_auc"],
                         m["at_0.5"]["f1"], m10["precision"], m10["recall"]))
    for name, roc, pr, f1, p10, r10 in rows:
        f = lambda v, w=8, d=4: (f"{v:>{w}.{d}f}" if v is not None else " " * (w - 1) + "-")  # noqa: E731
        print(f"  {name:34s} {f(roc)} {f(pr)} {f(f1)} {f(p10, 9, 3)} {f(r10, 11, 3)}")

    results["runtime_seconds"] = round(time.time() - t0, 1)
    results["seed"] = SEED
    results["quick_mode"] = quick
    OUT.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {OUT}")
    print(f"  total {(time.time() - t0) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()
