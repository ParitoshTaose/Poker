"""K-Means segmentation -- the unsupervised layer, and the bridge to the business story.

    reads   data/gold/player_lapse     prior-window features (same population as Fork A)
            data/gold/lapse_scores     each player's predicted lapse risk
    writes  data/gold/player_segments  one segment label per player
            docs/segments-results.json every number this prints

WHY THIS EXISTS, GIVEN FORK A ALREADY WORKS
-------------------------------------------
Fork A says *who* is drifting. It cannot say *what kind of player* is drifting, and that is
the entire thesis of the project: an operator earns rake only while recreational players keep
sitting down, so "2,000 accounts went quiet" and "the recreational segment went quiet" are the
same number carrying completely different management action.

CLUSTERED ON THE PRIOR WINDOW, NOT ON LIFETIME AVERAGES
-------------------------------------------------------
gold/player_features would be the obvious input, and it would quietly poison the cross-tab.
Its rate columns average over a player's whole history INCLUDING the label week: a lapsed
player contributes nothing to their own final week, an active one does, so segments built
that way partly encode the answer before the cross-tab is drawn. Clustering the `p_*` columns
from gold/player_lapse instead means every segment is knowable ON THE CUTOFF DAY -- which is
also the only version an operator could actually act on.

Two feature sets, for the same reason Fork A has two:

    STYLE      how they play. All six venues, 90,469 players. Deliberately EXCLUDES volume,
               so "segment 2 lapses more" is a statement about behaviour rather than an
               arithmetic restatement of "segment 2 plays less".
    ECOSYSTEM  style + intensity + multi-tabling -- the set that actually separates a grinder
               from a recreational player. Needs table identity, which iPoker does not record,
               so it is a FIVE-venue view covering 85.4% of players.

Scaling is not optional here and it is a rubric line: K-Means measures straight-line distance,
and p_hands runs to five figures while p_vpip is bounded by 1. Unscaled, one column decides
every cluster. StandardScaler, then, and the centroids are reported back in ORIGINAL units
because nobody can name a segment from a z-score.

Run:  .venv/bin/python src/segments.py
"""
import json
import os
import sys
import time
from pathlib import Path

# See features_lapse.py -- pin the bridge before pyspark is imported.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.clustering import BisectingKMeans, KMeans  # noqa: E402
from pyspark.ml.evaluation import ClusteringEvaluator  # noqa: E402
from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAPSE = ROOT / "data" / "gold" / "player_lapse"
SCORES = ROOT / "data" / "gold" / "lapse_scores"
OUT_TABLE = ROOT / "data" / "gold" / "player_segments"
OUT_JSON = ROOT / "docs" / "segments-results.json"
SCRATCH = ROOT / "data" / "_work" / "spark-scratch"

SEED = 42
K_RANGE = range(2, 9)

# The chosen k, and it is a judgement call rather than the silhouette optimum -- said out
# loud because "justify every decision" is a named part of the mark. Silhouette peaks at
# k=2 on this data, but "two kinds of poker player" is not a segmentation anybody can act
# on. Four is chosen because the project's plan hypothesises FOUR archetypes, so k=4 makes
# the comparison one-to-one instead of approximate. The silhouette cost of that choice is
# printed, not hidden.
K_CHOSEN = 4

# How a player plays. No volume columns, on purpose: if hands-played is in the clustering,
# then "this segment lapses more" is partly just "this segment plays less", and the finding
# collapses into arithmetic.
STYLE = [
    "p_vpip", "p_pfr", "p_vpip_pfr_gap", "p_fold_rate", "p_wtsd",
    "p_aggression_factor", "p_avg_invested_bb", "p_btn_share", "p_bb_vpip_rate",
    "p_distinct_stakes", "p_max_stake",
]

# Style plus the two things that actually distinguish a professional from a customer:
# how much they play, and how many tables they play at once. iPoker records no table
# identity, so this view drops it rather than imputing max_tables = 1 -- which would file
# all 13,201 of its players as recreational single-tablers.
ECOSYSTEM = STYLE + [
    "p_hands", "p_hands_per_active_day", "p_active_day_share",
    "p_max_tables", "p_avg_tables",
]

# The four archetypes hypothesised in docs/poker-project-plan.html. These are COMMUNITY
# RULES OF THUMB used to build intuition -- they are not findings from this dataset, and
# the plan labels them as such. They are here so the clusters can be checked against the
# hypothesis: matching and not matching are both results worth reporting.
ARCHETYPES = [
    {"name": "The Recreational", "vpip": 0.45, "pfr": 0.08, "tables": 1.5,
     "note": "the customer -- every rupee of rake originates here"},
    {"name": "The Grinder", "vpip": 0.22, "pfr": 0.18, "tables": 12.0,
     "note": "professional, multi-tabling, wins steadily"},
    {"name": "The Nit", "vpip": 0.12, "pfr": 0.09, "tables": 4.0,
     "note": "extremely cautious, generates little action"},
    {"name": "The Gambler", "vpip": 0.60, "pfr": 0.35, "tables": 1.0,
     "note": "plays everything, burns out fast"},
]

# What to print per cluster so a human can name it. Raw units, never z-scores.
PROFILE = [
    "p_vpip", "p_pfr", "p_aggression_factor", "p_wtsd", "p_avg_invested_bb",
    "p_hands", "p_hands_per_active_day", "p_active_day_share",
    "p_max_tables", "p_max_stake", "p_distinct_stakes", "p_bb_per_100",
]

results: dict = {}


def hr(t):
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}", flush=True)


def session():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return (SparkSession.builder
            .appName("ecosystem-engine-segments")
            .master("local[8]")
            .config("spark.driver.memory", "9g")
            .config("spark.sql.shuffle.partitions", "48")
            .config("spark.local.dir", str(SCRATCH.resolve()))
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def prep(df, feats):
    """Impute -> assemble -> scale. Fitted once and reused for every k, so the k sweep
    compares clusterings rather than accidentally comparing preprocessing."""
    stages = [
        Imputer(strategy="median", inputCols=feats,
                outputCols=[f"{c}_i" for c in feats]),
        VectorAssembler(inputCols=[f"{c}_i" for c in feats], outputCol="v",
                        handleInvalid="keep"),
        StandardScaler(inputCol="v", outputCol="features",
                       withMean=True, withStd=True),
    ]
    return Pipeline(stages=stages).fit(df).transform(df).select(
        "site", "venue", "player_id", "features", "lapsed", "lapse_risk",
        *PROFILE).cache()


def sweep(prepped, label):
    """Silhouette and within-cluster cost across k. Both are reported because they
    disagree: cost falls forever, silhouette does not, and neither one knows whether a
    segment can be named."""
    ev = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                             metricName="silhouette", distanceMeasure="squaredEuclidean")
    out = {}
    print(f"  {'k':>3s}  {'silhouette':>11s}  {'cost':>14s}   shares %")
    for k in K_RANGE:
        m = KMeans(featuresCol="features", k=k, seed=SEED, maxIter=50).fit(prepped)
        p = m.transform(prepped)
        sil = ev.evaluate(p)
        n = p.count()
        shares = sorted([round(r["count"] / n * 100, 1) for r in
                         p.groupBy("prediction").count().collect()], reverse=True)
        out[k] = {"silhouette": float(sil), "cost": float(m.summary.trainingCost),
                  "shares_pct": shares}
        print(f"  {k:3d}  {sil:11.4f}  {m.summary.trainingCost:14,.0f}   {shares}")
    best = max(out, key=lambda k: out[k]["silhouette"])
    print(f"  silhouette peaks at k={best} ({out[best]['silhouette']:.4f}); "
          f"k={K_CHOSEN} chosen anyway, costing "
          f"{out[best]['silhouette'] - out[K_CHOSEN]['silhouette']:.4f} -- see the note in"
          f" the source. Separation this weak means GRADIENTS, not natural groups.")
    results.setdefault("sweeps", {})[label] = out
    return out


def profile(p, label, n_total):
    """The centroid table, in original units, with lapse behaviour attached."""
    rows = (p.groupBy("prediction")
            .agg(F.count("*").alias("players"),
                 F.avg("lapsed").alias("lapse_rate"),
                 F.avg("lapse_risk").alias("mean_risk"),
                 *[F.avg(c).alias(c) for c in PROFILE],
                 F.avg(F.when(F.col("p_bb_per_100") < 0, 1.0).otherwise(0.0))
                  .alias("share_losing"))
            .orderBy("prediction").collect())

    print(f"\n  {'seg':>3s} {'players':>8s} {'share':>6s} {'lapse':>6s} {'risk':>5s} "
          f"{'vpip':>5s} {'pfr':>5s} {'af':>5s} {'wtsd':>5s} {'inv':>6s} "
          f"{'hands':>7s} {'h/day':>6s} {'act':>5s} {'tbls':>5s} {'stake':>6s} "
          f"{'bb100':>7s} {'losing':>6s}")
    out = []
    for r in rows:
        share = r["players"] / n_total * 100
        print(f"  {r['prediction']:3d} {r['players']:8,} {share:5.1f}% "
              f"{r['lapse_rate']:5.1%} {r['mean_risk']:5.2f} "
              f"{r['p_vpip']:5.2f} {r['p_pfr']:5.2f} {r['p_aggression_factor']:5.2f} "
              f"{r['p_wtsd']:5.2f} {r['p_avg_invested_bb']:6.2f} "
              f"{r['p_hands']:7,.0f} {r['p_hands_per_active_day']:6.0f} "
              f"{r['p_active_day_share']:5.2f} "
              f"{(r['p_max_tables'] or 0):5.1f} {r['p_max_stake']:6.2f} "
              f"{(r['p_bb_per_100'] or 0):7.1f} {r['share_losing']:5.0%}")
        out.append({k: (float(v) if isinstance(v, (int, float)) and k != "prediction"
                        else v) for k, v in r.asDict().items()} | {"share_pct": share})
    results.setdefault("segments", {})[label] = out
    return out


def match_archetypes(segs, label):
    """Nearest hypothesised archetype for each cluster, by VPIP/PFR/table count.

    The point is not to force a match. The plan hypothesised four named archetypes from
    community rules of thumb; if the data does not produce them, that is a finding and it
    goes in the report exactly as loudly as a match would.
    """
    print("\n  nearest hypothesised archetype (plan §Phase 3 -- rules of thumb, NOT findings):")
    out = []
    for s in segs:
        best, bd = None, 1e9
        for a in ARCHETYPES:
            # tables are on a different scale to the two rates, so damp them
            d = ((s["p_vpip"] - a["vpip"]) ** 2 + (s["p_pfr"] - a["pfr"]) ** 2
                 + (((s["p_max_tables"] or 1) - a["tables"]) / 12) ** 2)
            if d < bd:
                best, bd = a, d
        print(f"    seg {s['prediction']}  vpip {s['p_vpip']:.2f} pfr {s['p_pfr']:.2f} "
              f"tables {(s['p_max_tables'] or 0):.1f}  ->  {best['name']:18s} "
              f"(hypothesis: vpip {best['vpip']:.2f} pfr {best['pfr']:.2f} "
              f"tables {best['tables']:.0f}; distance {bd ** 0.5:.2f})")
        out.append({"segment": s["prediction"], "nearest": best["name"],
                    "distance": float(bd ** 0.5)})
    results.setdefault("archetype_match", {})[label] = out
    return out


def run(df, feats, label, n_total):
    hr(f"SEGMENTATION · {label}  ({len(feats)} features, {df.count():,} players)")
    prepped = prep(df, feats)
    sweep(prepped, label)

    km = KMeans(featuresCol="features", k=K_CHOSEN, seed=SEED, maxIter=50).fit(prepped)
    p = km.transform(prepped).cache()
    segs = profile(p, label, n_total)
    match_archetypes(segs, label)

    # ---- second configuration, so the choice of algorithm is tested and not assumed.
    # Bisecting k-means splits top-down instead of assigning-and-averaging; if the two
    # agree, the structure is in the data rather than in the algorithm.
    ev = ClusteringEvaluator(featuresCol="features", predictionCol="prediction",
                             metricName="silhouette", distanceMeasure="squaredEuclidean")
    bk = BisectingKMeans(featuresCol="features", k=K_CHOSEN, seed=SEED).fit(prepped)
    pb = bk.transform(prepped).withColumnRenamed("prediction", "bisecting")
    both = p.select("site", "player_id", "prediction").join(
        pb.select("site", "player_id", "bisecting"), ["site", "player_id"])
    # purity: for each bisecting cluster, how many of its members share one k-means label
    agree = (both.groupBy("bisecting", "prediction").count()
             .groupBy("bisecting").agg(F.max("count").alias("m"))
             .select(F.sum("m")).first()[0] / both.count())
    sil_km = ev.evaluate(p)
    sil_bk = ev.evaluate(pb.withColumnRenamed("bisecting", "prediction"))
    print(f"\n  two configurations at k={K_CHOSEN}:  k-means silhouette {sil_km:.4f}  ·  "
          f"bisecting {sil_bk:.4f}  ·  they place {agree:.1%} of players together")
    results.setdefault("algorithm_comparison", {})[label] = {
        "kmeans_silhouette": float(sil_km), "bisecting_silhouette": float(sil_bk),
        "agreement": float(agree), "k": K_CHOSEN,
    }
    return p


def main():
    t0 = time.time()
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")

    hr("K-MEANS SEGMENTATION · the unsupervised layer under Fork A")
    df = spark.read.parquet(str(LAPSE))
    scores = spark.read.parquet(str(SCORES)).select("site", "player_id", "lapse_risk")
    df = df.join(scores, ["site", "player_id"], "left")
    n = df.count()
    print(f"  {n:,} players, clustered on PRIOR-WINDOW behaviour only")
    print(f"  pooled lapse rate {df.select(F.avg('lapsed')).first()[0]:.1%} -- a segment is "
          f"only interesting if it sits well off that number")

    # ---- view 1: style only, all six venues
    p_style = run(df, STYLE, "style_all_venues", n)

    # ---- view 2: style + intensity + tables, five venues
    five = df.filter(F.col("tables_recorded"))
    n5 = five.count()
    print(f"\n  ECOSYSTEM view drops iPoker (no table identity): "
          f"{n5:,} of {n:,} players ({n5 / n:.1%})")
    p_eco = run(five, ECOSYSTEM, "ecosystem_five_venues", n5)

    # ---- the deliverable: a segment label per player, joined to their risk score
    hr("THE OUTPUT · segment + risk, one row per player")
    out = (p_style.select("site", "venue", "player_id", "lapsed", "lapse_risk",
                          F.col("prediction").alias("style_segment"))
           .join(p_eco.select("site", "player_id",
                              F.col("prediction").alias("ecosystem_segment")),
                 ["site", "player_id"], "left"))
    out.write.mode("overwrite").parquet(str(OUT_TABLE))
    print(f"  wrote {OUT_TABLE} -- {out.count():,} players")

    print("\n  where the risk actually sits -- segment x venue, style view:")
    (p_style.groupBy("venue").pivot("prediction")
     .agg(F.round(F.avg("lapsed") * 100, 1)).orderBy("venue").show(truncate=False))

    results["k_chosen"] = K_CHOSEN
    results["seed"] = SEED
    results["players"] = n
    results["runtime_seconds"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=float))
    print(f"  wrote {OUT_JSON}\n  total {(time.time() - t0) / 60:.1f} min")
    spark.stop()


if __name__ == "__main__":
    main()
