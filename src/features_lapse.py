"""Fork A, part 1 of 2 -- the prior-window feature table.

    data/gold/player_lapse    one row per (site, player_id), for Fork A only

WHY THIS SCRIPT EXISTS
----------------------
`gold/player_features` averages every rate column -- vpip, pfr, fold_rate, wtsd,
aggression_factor -- over a player's ENTIRE recorded history, final week included.
Fork A predicts the final week. So a model trained on those columns is reading its
own answer, and Step 5's bake-off measured exactly that: the same GBM scored
PR-AUC 0.886 on the lifetime rates alone versus 0.798 on the two features that are
honestly prior-window (hands_prior, tenure_days). The lifetime rates out-predicting
the legitimate ones is the signature of leakage, not of behaviour.

`docs/bakeoff-decision.md` therefore made Fork A conditional: extend the feature
build to compute rates restricted to `day <= cutoff` before Step 8. This is that
extension. Every column it writes with a `p_` prefix is computed from prior-window
rows only, so the resulting model could actually have been run on the cutoff day.

The cutoff is per venue -- `site_last_day - 7` -- never a global calendar date,
because the six venues stop recording on different days (ABS/PS 20, FTP 23, ONG 24,
IPN/PTY 26). A global window marks every ABS and PS player as lapsed by construction.

WHAT IT ADDS BEYOND "the same columns, filtered"
------------------------------------------------
Cutting the window down would have thrown away the strongest honest signal there is.
A retention team does not need a model to tell it that someone who last played 9 days
ago is drifting -- so recency and trend go in as features, and the model script then
has to beat a recency-only ranking to justify itself.

    p_recency_days     days between a player's last prior-window hand and the cutoff
    p_hands_w1 / _w2   volume in the last 7 prior days vs the 7 before that
    p_late_share       share of all prior volume that fell in its final 7 days
    p_active_day_share how many of their tenure days they actually showed up on

Run from the repo root:

    .venv/bin/python src/features_lapse.py              # reuse data/_work/hp_enriched
    .venv/bin/python src/features_lapse.py --rebuild    # rebuild that join from Silver

Reuse is safe because features.py materialises exactly the join this needs and never
mutates it; --rebuild exists for a clean machine, and takes a few minutes longer.
"""
import os
import shutil
import sys
import time
from pathlib import Path

# Pin BOTH ends of the Spark <-> Python bridge to this interpreter before pyspark is
# imported. Without it Spark launches its workers with whatever `python3` the shell
# resolves to, which on this machine is Apple's 3.9 -- and pyspark 4.2 cannot even be
# imported under 3.9 (`TypeError: unsupported operand type(s) for |` in sql/types.py).
# The job then dies inside a Java stack trace that says nothing about Python versions.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

from pyspark import StorageLevel  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

SILVER = Path("data/silver")
GOLD = Path("data/gold")
WORK = Path("data/_work/hp_enriched")
SCRATCH = Path("data/_work/spark-scratch")     # never /tmp -- see CLAUDE.md
OUT = GOLD / "player_lapse"

# Must match features.py, or the label moves under the features.
LAPSE_WINDOW = 7
WEEKEND_DAYS = [4, 5, 11, 12, 18, 19, 25, 26]

# The population filter, and a decision worth defending out loud.
#
# gold/player_features keeps players with >= 100 hands OVER THEIR WHOLE HISTORY.
# Reusing that rule here would leak: a player with 30 prior hands only survives it by
# playing 70+ hands in the label week, i.e. by being definitely-not-lapsed. The filter
# would then be partly built from the answer. Selecting on >= 100 PRIOR hands instead
# is a rule that could actually be applied on the cutoff day, which is the whole test
# for whether a step belongs in this table. The run prints both populations.
MIN_PRIOR_HANDS = 100

INF = float("inf")


def banner(msg, t0):
    print(f"\n{'=' * 74}\n[{time.time() - t0:7.1f}s] {msg}\n{'=' * 74}", flush=True)


def session():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return (SparkSession.builder
            .appName("ecosystem-engine-lapse-features")
            .master("local[8]")
            .config("spark.driver.memory", "9g")
            .config("spark.sql.shuffle.partitions", "96")
            .config("spark.local.dir", str(SCRATCH.resolve()))
            .config("spark.sql.files.maxPartitionBytes", "64m")
            # the console progress bar writes \r-heavy noise into any redirected log,
            # and this script's output is meant to be read after the fact
            .config("spark.ui.showConsoleProgress", "false")
            .getOrCreate())


def finite(col):
    """+-inf -> null.

    iPoker's raw .phhs files carry the literal `starting_stacks = [inf, inf]`. `inf`
    is a valid TOML float, so tomllib parsed it without complaint and it reached Gold
    intact: avg_stack_bb is +inf on all 15,549 iPoker players. An infinity detonates
    StandardScaler and any distance metric on contact; a null is honest and every
    MLlib stage already knows what to do with one. Same treatment, same reason, as the
    table-identity fields -- never impute, because a median stack on a venue that
    recorded none is a number nobody measured.
    """
    return F.when(F.col(col).isNotNull()
                  & ~F.isnan(F.col(col))
                  & (F.col(col) != F.lit(INF))
                  & (F.col(col) != F.lit(-INF)),
                  F.col(col))


def build_join(spark, t0):
    """Rebuild the seat x hand join that features.py materialises. Same logic."""
    banner("0/5  rebuilding data/_work/hp_enriched from Silver (--rebuild)", t0)
    hands = spark.read.parquet(str(SILVER / "hands"))
    hp = spark.read.parquet(str(SILVER / "hand_players"))

    dup = (hands.groupBy("hand_uid").count().filter(F.col("count") > 1)
                .select("hand_uid").cache())
    print(f"  colliding hand_uids dropped: {dup.count():,}")
    hands = hands.join(F.broadcast(dup), "hand_uid", "left_anti")
    hp = hp.join(F.broadcast(dup), "hand_uid", "left_anti")

    hf = (hands.withColumn("minute_of_day",
                           F.substring("time", 1, 2).cast("int") * 60
                           + F.substring("time", 4, 2).cast("int"))
          .select("hand_uid", "site", "day", "minute_of_day", "table_id",
                  "stake", "big_blind", "n_players",
                  F.col("rake_bb").isNotNull().alias("money_ok")))

    joined = (hp.select("hand_uid", "player_id", "vpip", "pfr", "folded", "showed",
                        "postflop_bets", "postflop_calls", "net_bb", "invested_bb",
                        "is_bb", "is_button", "starting_stack")
                .join(hf, "hand_uid")
                .withColumn("stack_bb", F.col("starting_stack") / F.col("big_blind"))
                .withColumn("minute_key", F.col("day") * 1440 + F.col("minute_of_day"))
                .drop("starting_stack", "big_blind", "minute_of_day"))

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    joined.write.mode("overwrite").parquet(str(WORK))


def main():
    t0 = time.time()
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")

    if "--rebuild" in sys.argv or not WORK.exists():
        build_join(spark, t0)

    # ------------------------------------------------------------------ 1
    banner("1/5  load the seat x hand join and set each venue's own cutoff", t0)
    seats = spark.read.parquet(str(WORK))

    # Each venue's last recorded day, and the cutoff seven days before it. Measured,
    # never hard-coded: if the Silver build ever changes, this follows it.
    window = (seats.groupBy("site")
              .agg(F.max("day").alias("site_last_day"),
                   F.min("day").alias("site_first_day"))
              .withColumn("cutoff", F.col("site_last_day") - LAPSE_WINDOW))
    rows = window.orderBy("site").collect()
    print("  site   first  last  cutoff   prior window   label window")
    for r in rows:
        print(f"  {r['site']:5s}  {r['site_first_day']:5d}  {r['site_last_day']:4d}"
              f"  {r['cutoff']:6d}   days {r['site_first_day']}-{r['cutoff']:<2d}"
              f"      days {r['cutoff'] + 1}-{r['site_last_day']}")
    print("  -> prior windows differ in LENGTH by venue (13 to 19 days). `site` is a")
    print("     feature and p_prior_span_days is carried, so a model can calibrate for it.")

    seats = seats.join(F.broadcast(window), "site")
    prior = F.col("day") <= F.col("cutoff")        # everything a model may see
    label_win = F.col("day") > F.col("cutoff")     # the week being predicted
    rd = F.col("cutoff") - F.col("day")            # 0 = cutoff day, larger = older
    is3 = F.col("n_players") >= 3                  # position is undefined heads-up

    def p_avg(expr):
        """Mean of `expr` over prior-window rows only. Null if they have none."""
        return F.avg(F.when(prior, expr))

    def p_sum(expr):
        return F.sum(F.when(prior, expr).otherwise(0))

    # ------------------------------------------------------------------ 2
    banner("2/5  aggregate to one row per player -- prior window only", t0)
    # One pass computes both windows: the p_ columns are conditioned on `prior`, and
    # hands_final7 on `label_win`. Two passes would read 116.6 M rows twice for no
    # reason, and would risk the two windows drifting apart.
    core = seats.groupBy("site", "player_id").agg(
        # ---- volume & recency, all measured to the cutoff
        p_sum(F.lit(1)).alias("p_hands"),
        F.size(F.collect_set(F.when(prior, F.col("day")))).alias("p_days_active"),
        F.min(F.when(prior, F.col("day"))).alias("p_first_day"),
        F.max(F.when(prior, F.col("day"))).alias("p_last_day"),
        p_sum(F.when(rd <= 6, 1).otherwise(0)).alias("p_hands_w1"),      # last 7 days
        p_sum(F.when((rd >= 7) & (rd <= 13), 1).otherwise(0)).alias("p_hands_w2"),
        p_sum(F.when(rd <= 2, 1).otherwise(0)).alias("p_hands_last3"),
        p_sum(F.when(F.col("day").isin(WEEKEND_DAYS), 1).otherwise(0)).alias("p_weekend_hands"),

        # ---- betting behaviour, recomputed inside the window
        p_avg(F.col("vpip").cast("int")).alias("p_vpip"),
        p_avg(F.col("pfr").cast("int")).alias("p_pfr"),
        p_avg(F.col("folded").cast("int")).alias("p_fold_rate"),
        p_avg(F.col("showed").cast("int")).alias("p_wtsd"),
        p_sum(F.col("postflop_bets")).alias("p_postflop_bets"),
        p_sum(F.col("postflop_calls")).alias("p_postflop_calls"),
        p_avg(F.col("invested_bb")).alias("p_avg_invested_bb"),
        p_avg(F.col("n_players").cast("double")).alias("p_avg_seats"),

        # ---- stakes
        F.size(F.collect_set(F.when(prior, F.col("stake")))).alias("p_distinct_stakes"),
        F.max(F.when(prior, F.col("stake"))).alias("p_max_stake"),
        F.min(F.when(prior, F.col("stake"))).alias("p_min_stake"),

        # ---- stack depth. inf-guarded at source: iPoker recorded no stacks at all.
        F.avg(F.when(prior, finite("stack_bb"))).alias("p_avg_stack_bb"),
        F.min(F.when(prior, finite("stack_bb"))).alias("p_min_stack_bb"),

        # ---- money, prior window only. Roughly a third of hands reconcile, and that
        # is a venue property: iPoker reconciles on none of its 6.0 M. Carried with its
        # own denominator so the model script can require a minimum before trusting it.
        p_sum(F.when(F.col("money_ok"), 1).otherwise(0)).alias("p_money_hands"),
        F.avg(F.when(prior & F.col("money_ok"), F.col("net_bb"))).alias("p_avg_net_bb"),

        # ---- position, 3+ handed only
        p_sum(F.when(F.col("is_bb") & is3, 1).otherwise(0)).alias("p_bb_hands"),
        p_sum(F.when(F.col("is_bb") & is3 & F.col("vpip"), 1).otherwise(0)).alias("p_bb_vpip"),
        p_sum(F.when(F.col("is_button") & is3, 1).otherwise(0)).alias("p_btn_hands"),
        p_sum(F.when(is3, 1).otherwise(0)).alias("p_hands_3plus"),

        # ---- the correctness carry-over: pooled BB-VPIP must land near 31%, not 100%
        p_sum(F.when(F.col("is_bb"), 1).otherwise(0)).alias("p_bb_hands_all"),
        p_sum(F.when(F.col("is_bb") & F.col("vpip"), 1).otherwise(0)).alias("p_bb_vpip_all"),

        # ---- THE LABEL, and the only column here allowed to see the final week
        F.sum(F.when(label_win, 1).otherwise(0)).alias("hands_final7"),

        F.first("cutoff").alias("cutoff"),
        F.first("site_last_day").alias("site_last_day"),
        F.first("site_first_day").alias("site_first_day"),
    ).cache()

    # ------------------------------------------------------------------ 3
    banner("3/5  multi-tabling, measured inside the prior window", t0)
    # The showpiece feature: bucket every hand into the wall-clock minute it started,
    # count the distinct tables a player occupied in that minute, take their peak.
    # One human plays one table; a grinder plays a dozen. Collapse to one row per
    # (player, minute, table) FIRST -- a grinder plays several hands per minute per
    # table, so this discards most of the rows before either group-by sees them.
    #
    # PS leaves 3.1% of table ids null. features.py drops those rows from the table
    # counts; the sentinel below keeps them for the minutes count, where they are
    # still evidence the player was sitting somewhere.
    seat_minutes = (seats.filter(prior)
                    .select("site", "player_id", "minute_key",
                            F.coalesce("table_id", F.lit("∅")).alias("tid"))
                    .distinct()
                    .persist(StorageLevel.DISK_ONLY))

    minutes = (seat_minutes.groupBy("site", "player_id")
               .agg(F.countDistinct("minute_key").alias("p_active_minutes")))

    real_tables = seat_minutes.filter(F.col("tid") != "∅")
    multi = (real_tables.groupBy("site", "player_id", "minute_key")
             .agg(F.count("*").alias("t"))
             .groupBy("site", "player_id")
             .agg(F.max("t").alias("p_max_tables"),
                  F.avg("t").alias("p_avg_tables")))
    tables = (real_tables.select("site", "player_id", "tid").distinct()
              .groupBy("site", "player_id")
              .agg(F.count("*").alias("p_distinct_tables")))

    # iPoker writes the literal string 'HandHQ' -- the data vendor's name -- into all
    # 6.0 M of its table fields, so every iPoker player scores exactly max_tables = 1.
    # That is a missing value wearing a plausible number: fed to a model it files all
    # 15,549 of them as recreational single-tablers, the precise opposite of the truth
    # for the grinders among them. Null it and flag it. Never impute 1.
    site_tables = (seats.filter(prior).groupBy("site")
                   .agg(F.countDistinct("table_id").alias("n_tables")))
    blind = [r["site"] for r in site_tables.collect() if (r["n_tables"] or 0) <= 1]
    print(f"  venues that do NOT record table identity: {blind or 'none'}")
    no_tables = F.col("site").isin(blind) if blind else F.lit(False)

    # ------------------------------------------------------------------ 4
    banner("4/5  derive, filter, write", t0)
    venues = (spark.read.parquet(str(GOLD / "hand_features"))
              .select("site", "venue").distinct())

    g = (core
         .join(multi, ["site", "player_id"], "left")
         .join(tables, ["site", "player_id"], "left")
         .join(minutes, ["site", "player_id"], "left")
         .join(F.broadcast(venues), "site", "left")

         .withColumn("tables_recorded", ~no_tables)
         .withColumn("p_max_tables", F.when(~no_tables, F.col("p_max_tables")))
         .withColumn("p_avg_tables", F.when(~no_tables, F.col("p_avg_tables")))
         .withColumn("p_distinct_tables", F.when(~no_tables, F.col("p_distinct_tables")))
         .withColumn("stacks_recorded", F.col("p_avg_stack_bb").isNotNull())

         # ---- recency & tenure. Every one of these is knowable on the cutoff day.
         .withColumn("p_recency_days", F.col("cutoff") - F.col("p_last_day"))
         .withColumn("p_tenure_days", F.col("cutoff") - F.col("p_first_day") + 1)
         .withColumn("p_prior_span_days", F.col("cutoff") - F.col("site_first_day") + 1)
         .withColumn("p_hands_per_active_day", F.col("p_hands") / F.col("p_days_active"))
         .withColumn("p_active_day_share", F.col("p_days_active") / F.col("p_tenure_days"))
         .withColumn("p_hands_per_minute", F.col("p_hands") / F.col("p_active_minutes"))

         # ---- trend. Is their volume decaying into the cutoff or ramping up?
         # p_late_share is span-robust (a share of their own total); the raw w1/w2 pair
         # is left in beside it so a tree can use the levels as well as the ratio.
         .withColumn("p_late_share", F.col("p_hands_w1") / F.col("p_hands"))
         .withColumn("p_w1_w2_ratio",
                     F.col("p_hands_w1") / F.greatest(F.col("p_hands_w2"), F.lit(1)))
         .withColumn("p_last3_share", F.col("p_hands_last3") / F.col("p_hands"))
         .withColumn("p_weekend_share", F.col("p_weekend_hands") / F.col("p_hands"))

         # ---- style
         .withColumn("p_vpip_pfr_gap", F.col("p_vpip") - F.col("p_pfr"))
         .withColumn("p_aggression_factor",
                     F.col("p_postflop_bets")
                     / F.greatest(F.col("p_postflop_calls"), F.lit(1)))
         .withColumn("p_bb_vpip_rate",
                     F.when(F.col("p_bb_hands") > 0,
                            F.col("p_bb_vpip") / F.col("p_bb_hands")))
         .withColumn("p_btn_share",
                     F.when(F.col("p_hands_3plus") > 0,
                            F.col("p_btn_hands") / F.col("p_hands_3plus")))
         .withColumn("p_stake_range", F.col("p_max_stake") - F.col("p_min_stake"))

         # ---- money, with its own denominator alongside it
         .withColumn("p_bb_per_100", F.col("p_avg_net_bb") * 100)
         .withColumn("p_money_coverage", F.col("p_money_hands") / F.col("p_hands"))
         .withColumn("money_recorded", F.col("p_money_hands") > 0)

         # ---- THE LABEL
         .withColumn("lapsed", (F.col("hands_final7") == 0).cast("int")))

    seen = core.filter(F.col("p_hands") > 0).count()
    gold = g.filter(F.col("p_hands") >= MIN_PRIOR_HANDS).cache()
    kept = gold.count()

    # What the alternative filter would have kept, so the decision is visible rather
    # than asserted. `p_hands + hands_final7` is the lifetime count features.py used.
    alt = g.filter((F.col("p_hands") > 0)
                   & (F.col("p_hands") + F.col("hands_final7") >= MIN_PRIOR_HANDS)).count()
    print(f"  players with any prior-window activity : {seen:,}")
    print(f"  kept, >= {MIN_PRIOR_HANDS} PRIOR hands (the honest rule) : {kept:,} "
          f"({kept / seen * 100:.1f}%)")
    print(f"  would have been kept by Gold's lifetime >= {MIN_PRIOR_HANDS} rule : {alt:,}")
    print(f"  the difference, {alt - kept:,} players, is the selection leak this avoids:")
    print("    they clear 100 hands only by playing in the week being predicted,")
    print("    so the filter itself would have been built out of the answer.")

    gold.write.mode("overwrite").parquet(str(OUT))
    print(f"\n  wrote {OUT}  ({len(gold.columns)} cols)")

    # ------------------------------------------------------------------ 5
    banner("5/5  sanity checks -- these must match the parser, or something broke", t0)
    w = core.select(
        F.sum(F.col("p_vpip") * F.col("p_hands")).alias("v"),
        F.sum(F.col("p_pfr") * F.col("p_hands")).alias("p"),
        F.sum("p_hands").alias("n"),
        F.sum("p_bb_vpip_all").alias("bva"), F.sum("p_bb_hands_all").alias("bha"),
        F.sum("hands_final7").alias("hf7"),
    ).first()
    print(f"  prior-window pooled VPIP   {w['v'] / w['n'] * 100:5.1f}%   "
          f"(all 26 days: 27.4%)")
    print(f"  prior-window pooled PFR    {w['p'] / w['n'] * 100:5.1f}%   "
          f"(all 26 days: 14.1%)")
    print(f"  prior-window BB VPIP       {w['bva'] / w['bha'] * 100:5.1f}%   "
          f"(all 26 days: 31.3%)  <- must NOT be ~100%")
    print(f"  seat-rows split            prior {w['n']:,} + label {w['hf7']:,} "
          f"= {w['n'] + w['hf7']:,}")

    print("\n  class balance -- the thing a metric choice depends on:")
    (gold.groupBy("site", "venue")
         .agg(F.count("*").alias("players"),
              F.sum("lapsed").alias("lapsed"),
              F.round(F.avg("lapsed") * 100, 1).alias("lapsed_pct"),
              F.round(F.avg("p_recency_days"), 2).alias("mean_recency"),
              F.round(F.avg("p_hands"), 0).alias("mean_prior_hands"))
         .orderBy(F.desc("players")).show(truncate=False))
    p = gold.select(F.avg("lapsed")).first()[0]
    print(f"  POOLED PREVALENCE: {p:.4f} lapsed -- a majority-class model scores "
          f"PR-AUC {p:.3f} and ROC-AUC 0.500. That is the floor to beat.")

    print("\n  does the honest recency feature separate the classes at all?")
    (gold.groupBy("lapsed")
         .agg(F.count("*").alias("players"),
              F.round(F.avg("p_recency_days"), 2).alias("recency"),
              F.round(F.avg("p_late_share"), 3).alias("late_share"),
              F.round(F.avg("p_hands"), 0).alias("prior_hands"),
              F.round(F.avg("p_vpip"), 3).alias("vpip"),
              F.round(F.avg("p_max_tables"), 2).alias("max_tables"))
         .orderBy("lapsed").show(truncate=False))
    print("  If it separates them here, the models below have to BEAT it, not match it")
    print("  -- a retention team can sort a spreadsheet by recency for free.")

    print("\n  null coverage on every modelling column (nulls are a venue property):")
    n = gold.count()
    nulls = gold.select([
        F.round(F.sum(F.col(c).isNull().cast("int")) / F.lit(n) * 100, 1).alias(c)
        for c in gold.columns if c not in ("site", "player_id", "venue")
    ]).first().asDict()
    for c, pct in sorted(nulls.items(), key=lambda kv: -kv[1]):
        if pct > 0:
            print(f"    {c:26s} {pct:5.1f}% null")

    banner(f"done in {(time.time() - t0) / 60:.1f} min", t0)
    spark.stop()


if __name__ == "__main__":
    main()
