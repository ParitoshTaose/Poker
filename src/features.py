"""Silver -> Gold: the two feature tables every fork shares.

Reads the three Silver Parquet tables and writes the two Gold tables:

    data/gold/player_features   one row per (site, player_id)  -> K-Means, Fork A, Fork B
    data/gold/hand_features     one row per hand               -> Fork C

This is the step where the project becomes small again: 116.6 M seat-rows go in,
a few hundred thousand player-rows come out. Everything after this point (the
models, the deck's numbers) reads Gold, never Silver.

Run from the repo root:   .venv/bin/python src/features.py
Smoke-test on a subset:   .venv/bin/python src/features.py --parts 3
    (reads only the first N of the 44 Silver parts and writes to data/gold_sample/,
     so a test run can never clobber the real Gold tables)

Design notes that are decisions, not details:

* The key is (site, player_id), never player_id alone. The same human on two
  sites is two different codes; merging them would invent a player who does not
  exist.
* hand_uid is unique ACROSS venues but not WITHIN one -- iPoker reissues ids.
  The colliding uids are dropped from both tables before any join, because a
  seat-row cannot be attributed to the right hand once two hands share a uid.
* Money features come only from hands whose money actually reconciles against
  the pot -- 25.0% of hands. "Winnings present" is NOT the same as "winnings
  usable": a further 13.8% of hands carry a complete set of winnings that sums
  to ~1.5x the pot, and averaging those in makes every player look like a heavy
  loser. The shortfall is structural per venue, not random -- iPoker reconciles
  on none of its 6.0 M hands -- so Fork B is a four-venue question, and imputing
  the missing figures would invent a break-even player who never existed.
* Activity windows are per venue, not global. The six venues stop recording on
  different days (ABS/PS day 20, FTP 23, ONG 24, IPN/PTY 26), so a global "played
  in the last week?" test would label every ABS and PS player as lapsed purely
  because their data ends -- 45% of the dataset, silently mislabelled.
* The join is materialised once to data/_work/ and read twice, instead of being
  recomputed for each aggregation. One big shuffle instead of two.
"""
import shutil
import sys
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

SILVER = Path("data/silver")
GOLD = Path("data/gold")
WORK = Path("data/_work/hp_enriched")   # intermediate, not a deliverable

# Spark's shuffle spill directory. NOT /tmp: this job spills many GB, and
# anything that tidies /tmp mid-run (macOS's own cleaner, a sandbox, a CI agent)
# takes the shuffle files with it. The job then dies with a FileNotFoundException
# on blockmgr-*/shuffle_*.data twenty minutes in, which reads like a Spark bug and
# is not one. Keep scratch on the project disk, where nothing else touches it.
SCRATCH = Path("data/_work/spark-scratch")

# A player's VPIP over 12 hands is noise; K-Means will happily build a cluster
# out of it. This threshold is a decision that must be reported, not hidden --
# the run prints how many players it removes.
MIN_HANDS = 100

# The data runs 1-26 July 2009 by the hands' own timestamps, even though every
# source folder is labelled 2009-07-01_2009-07-23. Day 1 was a Wednesday, so
# these are its Saturdays and Sundays. Weekend traffic is a real operator signal.
WEEKEND_DAYS = [4, 5, 11, 12, 18, 19, 25, 26]

# The six venues do NOT share an observation window: ABS and PS stop at day 20,
# FTP at 23, ONG at 24, IPN and PTY at 26. A single global "did they play in the
# last week?" rule would therefore mark every ABS and PS player as lapsed by
# construction -- 45% of the dataset -- because their data simply stops rather
# than because anyone left. Every activity window below is measured backwards
# from each venue's OWN last observed day.
LAPSE_WINDOW = 7


def banner(msg, t0):
    print(f"\n{'=' * 74}\n[{time.time() - t0:7.1f}s] {msg}\n{'=' * 74}", flush=True)


def session():
    """Tuned for THIS machine: M4, 10 cores, 16 GB. The defaults are wrong here."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    return (SparkSession.builder
            .appName("ecosystem-engine-features")
            .master("local[8]")                              # 8 of 10 cores, 2 for macOS
            .config("spark.driver.memory", "9g")             # measured: 1 GB without this
            .config("spark.sql.shuffle.partitions", "96")    # 200 is cluster-tuned, 48 is
            .config("spark.local.dir", str(SCRATCH.resolve()))   # too coarse for 116 M rows
            .config("spark.sql.files.maxPartitionBytes", "64m")
            .getOrCreate())


def silver(name, parts):
    """Full table, or just the first `parts` Parquet parts for a smoke test.

    build_silver.py writes part-{batch} for all three tables from the same batch
    of source files, so taking the first N parts of each yields a consistent
    subset -- the same hands in all three tables, not three unrelated slices.
    """
    if parts is None:
        return [str(SILVER / name)]
    return [str(p) for p in sorted((SILVER / name).glob("*.parquet"))[:parts]]


def main():
    global GOLD, WORK
    parts = None
    if "--parts" in sys.argv:
        parts = int(sys.argv[sys.argv.index("--parts") + 1])
        GOLD, WORK = Path("data/gold_sample"), Path("data/_work/hp_enriched_sample")
        print(f"SUBSET RUN: first {parts} Silver parts -> {GOLD}")

    t0 = time.time()
    spark = session()
    spark.sparkContext.setLogLevel("ERROR")
    GOLD.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1
    banner("1/6  load Silver and drop the colliding hand_uids", t0)
    hands = spark.read.parquet(*silver("hands", parts))
    hp = spark.read.parquet(*silver("hand_players", parts))

    dup = (hands.groupBy("hand_uid").count()
                .filter(F.col("count") > 1)
                .select("hand_uid").cache())
    n_dup_uid = dup.count()
    n_dup_row = hands.join(F.broadcast(dup), "hand_uid").count()
    hands = hands.join(F.broadcast(dup), "hand_uid", "left_anti")
    hp = hp.join(F.broadcast(dup), "hand_uid", "left_anti")
    print(f"  colliding hand_uids: {n_dup_uid:,}  covering {n_dup_row:,} hand rows -> dropped")

    # ------------------------------------------------------------------ 2
    banner("2/6  roll seats up to the hand (stack depth + money usability)", t0)
    # count() skips nulls, so w_present is how many seats actually carry a figure.
    per_hand = hp.groupBy("hand_uid").agg(
        F.count("*").alias("seats"),
        F.avg("starting_stack").alias("avg_stack"),
        F.min("starting_stack").alias("min_stack"),
        F.count("winnings").alias("w_present"),
        F.sum("winnings").alias("w_sum"),
    )

    hx = (hands.join(per_hand, "hand_uid", "left")
          # A four-way data-quality split, not a boolean, because the report has
          # to explain WHICH failure happened. Partial coverage does not occur:
          # winnings are recorded for every seat of a hand or for none of them.
          #   ok           money balances against the pot -> a win-rate is real
          #   unreconciled every seat has a figure, but they sum to ~1.5x the pot
          #   all_zero     every seat recorded as winning exactly nothing
          #   absent       no winnings column at all
          .withColumn("money_status",
                      F.when(F.col("rake_bb").isNotNull(), "ok")
                       .when(F.col("w_present") == 0, "absent")
                       .when(F.col("w_sum") == 0, "all_zero")
                       .otherwise("unreconciled"))
          .withColumn("money_ok", F.col("money_status") == "ok")
          .withColumn("hour", F.substring("time", 1, 2).cast("int"))
          .withColumn("minute_of_day",
                      F.substring("time", 1, 2).cast("int") * 60
                      + F.substring("time", 4, 2).cast("int"))
          .withColumn("is_weekend", F.col("day").isin(WEEKEND_DAYS))
          .withColumn("avg_stack_bb", F.col("avg_stack") / F.col("big_blind"))
          .withColumn("min_stack_bb", F.col("min_stack") / F.col("big_blind")))

    # ------------------------------------------------------------------ 3
    banner("3/6  write gold/hand_features  (Fork C)", t0)
    # Column groups matter here. Everything above the OUTCOME line is known
    # before a card is dealt and is legitimate for predicting pot size.
    # saw_flop / showdown / rake_bb are settled BY the hand, at the same time as
    # the pot -- they are recorded because they are useful for describing the
    # data, but feeding them to a pot-size model is target leakage.
    hand_features = hx.select(
        "hand_uid", "site", "venue", "table_id",                    # identifiers
        "day", "hour", "minute_of_day", "is_weekend",               # ---- pre-hand
        "stake", "big_blind", "n_players", "seat_count",
        F.round("avg_stack_bb", 3).alias("avg_stack_bb"),
        F.round("min_stack_bb", 3).alias("min_stack_bb"),
        "saw_flop", "showdown", "rake_bb",                          # ---- OUTCOME: leaky
        "money_status", "money_ok",                                 # ---- data quality
        "pot_bb",                                                   # ---- target
    )
    hand_features.write.mode("overwrite").parquet(str(GOLD / "hand_features"))
    print(f"  wrote {GOLD / 'hand_features'}")

    # ------------------------------------------------------------------ 4
    banner("4/6  join seats to hands, materialise once", t0)
    # Read hand_features back rather than recomputing hx: the expensive part
    # (the 116 M-row roll-up in step 2) is already on disk.
    hf = spark.read.parquet(str(GOLD / "hand_features")).select(
        "hand_uid", "site", "day", "minute_of_day", "table_id", "stake",
        "big_blind", "n_players", "money_ok")

    joined = (hp.select("hand_uid", "player_id", "vpip", "pfr", "folded", "showed",
                        "postflop_bets", "postflop_calls", "net_bb", "invested_bb",
                        "is_bb", "is_button", "starting_stack")
                .join(hf, "hand_uid")
                .withColumn("stack_bb", F.col("starting_stack") / F.col("big_blind"))
                # one integer per wall-clock minute of the month -> the multi-tabling bucket
                .withColumn("minute_key", F.col("day") * 1440 + F.col("minute_of_day"))
                .drop("starting_stack", "big_blind", "minute_of_day"))

    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.parent.mkdir(parents=True, exist_ok=True)
    joined.write.mode("overwrite").parquet(str(WORK))
    joined = spark.read.parquet(str(WORK))
    print(f"  materialised {joined.count():,} seat-rows to {WORK}")

    # ------------------------------------------------------------------ 5
    banner("5/6  aggregate to one row per player", t0)
    is3 = F.col("n_players") >= 3          # position is only defined 3-handed and up
    money = F.col("money_ok")

    # Each venue's own last day, so "stopped playing" means stopped relative to
    # when that venue's recording stopped -- not relative to a global calendar.
    site_window = hf.groupBy("site").agg(F.max("day").alias("site_last_day"))
    print("  observation window per venue:",
          {r["site"]: r["site_last_day"] for r in site_window.collect()})
    joined = joined.join(F.broadcast(site_window), "site")
    cutoff = F.col("site_last_day") - LAPSE_WINDOW

    # size(collect_set(x)) instead of countDistinct(x): three distinct aggregates
    # in one groupBy make Spark expand the shuffle several times over. These
    # columns have tiny cardinality per player, so a set is far cheaper.
    core = joined.groupBy("site", "player_id").agg(
        F.count("*").alias("hands_played"),

        F.avg(F.col("vpip").cast("int")).alias("vpip"),
        F.avg(F.col("pfr").cast("int")).alias("pfr"),
        F.avg(F.col("folded").cast("int")).alias("fold_rate"),
        F.avg(F.col("showed").cast("int")).alias("wtsd"),
        F.sum("postflop_bets").alias("postflop_bets"),
        F.sum("postflop_calls").alias("postflop_calls"),
        F.avg("invested_bb").alias("avg_invested_bb"),
        F.avg("stack_bb").alias("avg_stack_bb"),

        # money: restricted to hands whose winnings are actually usable
        F.sum(F.when(money, 1).otherwise(0)).alias("money_hands"),
        F.avg(F.when(money, F.col("net_bb"))).alias("avg_net_bb"),

        # activity window -- the raw material for Fork A's lapse label, measured
        # against this venue's own last day rather than a global date
        F.size(F.collect_set("day")).alias("days_active"),
        F.min("day").alias("first_day"),
        F.max("day").alias("last_day"),
        F.first("site_last_day").alias("site_last_day"),
        F.sum(F.when(F.col("day") > cutoff, 1).otherwise(0)).alias("hands_final7"),
        F.sum(F.when(F.col("day") <= cutoff, 1).otherwise(0)).alias("hands_prior"),

        F.size(F.collect_set("stake")).alias("distinct_stakes"),
        F.max("stake").alias("max_stake"),

        # big-blind defence rate: the correctness test, carried per player.
        # ~100% would mean the check/call bug is back.
        F.sum(F.when(F.col("is_bb"), 1).otherwise(0)).alias("bb_hands_all"),
        F.sum(F.when(F.col("is_bb") & F.col("vpip"), 1).otherwise(0)).alias("bb_vpip_all"),
        F.sum(F.when(F.col("is_bb") & is3, 1).otherwise(0)).alias("bb_hands"),
        F.sum(F.when(F.col("is_bb") & is3 & F.col("vpip"), 1).otherwise(0)).alias("bb_vpip"),
        F.sum(F.when(F.col("is_button") & is3, 1).otherwise(0)).alias("btn_hands"),
        F.sum(F.when(is3, 1).otherwise(0)).alias("hands_3plus"),
    ).cache()

    # ---- the showpiece: how many tables at once?
    # Bucket every hand into the minute it started, count how many distinct tables
    # a player occupied inside that minute, then take their peak and average. A
    # human plays one table; a professional grinder plays a dozen. This one number
    # separates them with no labelled data at all.
    #
    # Collapse to one row per (player, minute, table) FIRST. A grinder plays several
    # hands per minute per table, so this throws most of the 116.6 M rows away before
    # either group-by sees them, and reads 4 columns off disk instead of 20. Persisted
    # to disk rather than heap: it is still tens of millions of rows, and the driver
    # only has 9 GB.
    seat_minutes = (joined.filter(F.col("table_id").isNotNull())
                    .select("site", "player_id", "minute_key", "table_id")
                    .distinct()
                    .persist(StorageLevel.DISK_ONLY))

    multi = (seat_minutes
             .groupBy("site", "player_id", "minute_key")
             .agg(F.count("*").alias("tables_this_minute"))
             .groupBy("site", "player_id")
             .agg(F.max("tables_this_minute").alias("max_tables"),
                  F.avg("tables_this_minute").alias("avg_tables")))

    # How many different tables in total -- taken off the same collapsed set, which
    # gives the identical answer to a collect_set over all 116.6 M rows for a small
    # fraction of the shuffle.
    tables = (seat_minutes.select("site", "player_id", "table_id").distinct()
              .groupBy("site", "player_id")
              .agg(F.count("*").alias("distinct_tables")))

    # site -> venue agrees 1:1 on every hand, so this is a 6-row lookup.
    venues = spark.read.parquet(str(GOLD / "hand_features")).select("site", "venue").distinct()

    # Not every venue records which table a hand was played at. iPoker writes the
    # literal string "HandHQ" -- the data vendor's name -- into all 6.0 M of its
    # table fields, so every iPoker player comes out at exactly max_tables = 1.
    # That is not a measurement of single-tabling, it is a missing value wearing a
    # plausible number, and handing it to K-Means would file all 15,549 of them as
    # recreational single-tablers -- the precise opposite of what the ecosystem
    # thesis is looking for. Null the table-derived features there instead, and
    # carry an explicit flag so the omission is visible rather than inferred.
    site_tables = hf.groupBy("site").agg(F.countDistinct("table_id").alias("n_tables"))
    blind = [r["site"] for r in site_tables.collect() if (r["n_tables"] or 0) <= 1]
    print(f"  venues that do NOT record table identity: {blind or 'none'}")
    no_tables = F.col("site").isin(blind) if blind else F.lit(False)

    gold = (core.join(multi, ["site", "player_id"], "left")
            .join(tables, ["site", "player_id"], "left")
            .join(F.broadcast(venues), "site", "left")
            # missing, not 1 -- see the note above
            .withColumn("tables_recorded", ~no_tables)
            .withColumn("max_tables", F.when(~no_tables, F.col("max_tables")))
            .withColumn("avg_tables", F.when(~no_tables, F.col("avg_tables")))
            .withColumn("distinct_tables", F.when(~no_tables, F.col("distinct_tables")))
            .withColumn("vpip_pfr_gap", F.col("vpip") - F.col("pfr"))
            .withColumn("aggression_factor",
                        F.col("postflop_bets") / F.greatest(F.col("postflop_calls"), F.lit(1)))
            .withColumn("bb_per_100", F.col("avg_net_bb") * 100)
            .withColumn("money_coverage", F.col("money_hands") / F.col("hands_played"))
            .withColumn("hands_per_day", F.col("hands_played") / F.col("days_active"))
            .withColumn("span_days", F.col("last_day") - F.col("first_day") + 1)
            # Fork A's raw label material: how many days before this venue stopped
            # recording did the player last show up? 0 = still active at the end.
            .withColumn("days_since_last", F.col("site_last_day") - F.col("last_day"))
            .withColumn("bb_vpip_rate",
                        F.when(F.col("bb_hands") > 0, F.col("bb_vpip") / F.col("bb_hands")))
            .withColumn("btn_share",
                        F.when(F.col("hands_3plus") > 0,
                               F.col("btn_hands") / F.col("hands_3plus"))))

    n_all = core.count()
    gold = gold.filter(F.col("hands_played") >= MIN_HANDS).cache()
    n_keep = gold.count()
    gold.write.mode("overwrite").parquet(str(GOLD / "player_features"))
    print(f"  players seen          : {n_all:,}")
    print(f"  players kept (>= {MIN_HANDS} hands): {n_keep:,}  "
          f"({n_keep / n_all * 100:.1f}%)  -> {GOLD / 'player_features'}")
    print(f"  players dropped       : {n_all - n_keep:,} "
          f"({(n_all - n_keep) / n_all * 100:.1f}%) -- report this, do not hide it")

    # ------------------------------------------------------------------ 6
    banner("6/6  sanity checks -- these must match the parser, or something broke", t0)
    # Weighting each player's rate by their hand count reproduces the pooled
    # per-seat rate the parser prints, so the two are directly comparable.
    w = core.select(
        F.sum(F.col("vpip") * F.col("hands_played")).alias("v"),
        F.sum(F.col("pfr") * F.col("hands_played")).alias("p"),
        F.sum("hands_played").alias("n"),
        F.sum("bb_vpip_all").alias("bva"), F.sum("bb_hands_all").alias("bha"),
        F.sum("bb_vpip").alias("bv"), F.sum("bb_hands").alias("bh"),
        F.sum("money_hands").alias("mh"),
    ).first()
    print(f"  pooled VPIP              {w['v'] / w['n'] * 100:5.1f}%   (parser: 27.4%)")
    print(f"  pooled PFR               {w['p'] / w['n'] * 100:5.1f}%   (parser: 14.1%)")
    print(f"  big-blind VPIP, all      {w['bva'] / w['bha'] * 100:5.1f}%   (parser: 31.3%) "
          f"<- must NOT be ~100%")
    print(f"  big-blind VPIP, 3+ handed{w['bv'] / w['bh'] * 100:5.1f}%")
    print(f"  seat-level money coverage{w['mh'] / w['n'] * 100:5.1f}%")

    # THE money regression test. Poker is zero-sum apart from the rake, so every
    # seat's net across the whole dataset must sum to exactly minus the rake
    # taken. It came out at 18x the rake before the uncalled bet was returned to
    # the player who made it, which made every winning player look like a loser.
    hf_all = spark.read.parquet(str(GOLD / "hand_features"))
    tot_net = core.select(F.sum(F.col("avg_net_bb") * F.col("money_hands"))).first()[0]
    tot_rake = hf_all.filter("money_ok").select(F.sum("rake_bb")).first()[0]
    print(f"\n  money identity   sum(net_bb) = {tot_net:>16,.0f} bb")
    print(f"                  -sum(rake_bb) = {-tot_rake:>16,.0f} bb")
    print(f"                          ratio = {tot_net / -tot_rake:.4f}   <- must be 1.0000")

    print("\n  hand-level money quality -- why Fork B cannot use every hand:")
    (hf_all.groupBy("site").pivot("money_status", ["ok", "unreconciled", "all_zero", "absent"])
           .count().na.fill(0).orderBy("site").show(truncate=False))

    print("  money coverage by venue (Fork B can only use what is covered):")
    (gold.groupBy("site", "venue")
         .agg(F.count("*").alias("players"),
              F.sum("hands_played").alias("seats"),   # seat-rows, NOT distinct hands
              F.avg("money_coverage").alias("mean_cov"),
              F.avg("vpip").alias("vpip"),
              F.avg("max_tables").alias("mean_max_tables"))
         .orderBy(F.desc("seats")).show(truncate=False))

    print("  the 10 highest-volume players -- plausible, or a parsing artifact?")
    (gold.select("site", "player_id", "hands_played", "days_active", "hands_per_day",
                 "max_tables", "vpip", "pfr", "bb_per_100", "days_since_last")
         .orderBy(F.desc("hands_played")).show(10, truncate=False))

    print("  hands_played distribution across the kept players:")
    gold.select("hands_played", "days_active", "max_tables", "vpip", "pfr",
                "aggression_factor").summary(
        "min", "25%", "50%", "75%", "max").show(truncate=False)

    banner(f"done in {(time.time() - t0) / 60:.1f} min", t0)
    spark.stop()


if __name__ == "__main__":
    main()
