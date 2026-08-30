#!/usr/bin/env python3
"""
pipeline_facts.py -- measure the data lake, so the report's engineering numbers are
read off disk rather than remembered.

The model results (Fork A, segments, rake at risk) already write their own JSON. The
*pipeline* numbers -- how many rows survive each layer, how far the Parquet compresses,
and the two correctness identities -- lived only in run logs and in CLAUDE.md until this
script. It recomputes them from data/bronze, data/silver and data/gold and writes
docs/pipeline-facts.json, which src/report_figures.py then injects into the report.

Run:  .venv/bin/python src/pipeline_facts.py          # measure and write
      .venv/bin/python src/pipeline_facts.py --fast   # skip the 23 s zero-sum join

Polars only, no Spark. ~40 s cold, ~15 s with --fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "pipeline-facts.json"

SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
BRONZE = ROOT / "data" / "bronze"

# Recorded by the Silver build's own run log (build_silver.py, 2026-08-07, all 21,782
# files, 27.7 min on 8 cores). These are the only two figures here that cannot be
# recomputed from what is on disk -- a dropped hand leaves no row behind to count.
BUILD_LOG = {
    "published_hands": 21_605_687,   # the dataset's own published count
    "dropped_no_big_blind": 49_119,
    "dropped_no_hand_id": 133,
    "skipped_files": 0,
    "build_minutes": 27.7,
    "build_cores": 8,
}


def du_bytes(path: Path) -> int:
    """Apparent size: the sum of the files' own byte counts, not allocated blocks.

    `du` reports blocks, which inflates a tree of many small files and makes the
    compression ratio depend on the filesystem. Every size quoted in the report is a
    decimal GB (bytes / 1e9) of this figure, so it is reproducible anywhere."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def scan(table: Path) -> pl.LazyFrame:
    return pl.scan_parquet(str(table / "**" / "*.parquet"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip the full-scale zero-sum join -- NOTE: the report quotes it, so a "
                         "--fast run leaves docs/pipeline-facts.json incomplete and "
                         "report_figures.py will refuse to use it")
    args = ap.parse_args()
    t0 = time.time()

    for path in (SILVER, GOLD):
        if not path.exists():
            sys.exit(f"missing {path} -- build the lake first")

    facts: dict = {"build_log": BUILD_LOG}

    # ---- Bronze -------------------------------------------------------------
    print("bronze ...", flush=True)
    files = list(BRONZE.rglob("*.phhs")) if BRONZE.exists() else []
    facts["bronze"] = {
        "files": len(files),
        "bytes": du_bytes(BRONZE),
        "venues": sorted({p.parent.parent.name.split("=")[-1] for p in files}) if files else [],
    }

    # ---- Silver -------------------------------------------------------------
    print("silver ...", flush=True)
    silver = {}
    for name in ("hands", "hand_players", "actions"):
        table = SILVER / name
        rows = scan(table).select(pl.len()).collect().item()
        parts = len(list(table.rglob("*.parquet")))
        silver[name] = {"rows": rows, "bytes": du_bytes(table), "parts": parts}
    facts["silver"] = silver
    silver_bytes = sum(t["bytes"] for t in silver.values())
    facts["silver_total_bytes"] = silver_bytes
    facts["compression_x"] = (facts["bronze"]["bytes"] / silver_bytes) if silver_bytes else None
    facts["seats_per_hand"] = silver["hand_players"]["rows"] / silver["hands"]["rows"]
    facts["actions_per_hand"] = silver["actions"]["rows"] / silver["hands"]["rows"]

    # ---- the reconciliation: every hand parsed, or dropped with a reason -----
    print("reconciliation ...", flush=True)
    h = scan(SILVER / "hands")
    agg = h.select(
        rows=pl.len(),
        distinct=pl.col("hand_uid").n_unique(),
        reconciling=pl.col("rake_bb").is_not_null().sum(),
        rake_bb_sum=pl.col("rake_bb").sum(),
        venues=pl.col("site").n_unique(),
        first_day=pl.col("day").min(),
        last_day=pl.col("day").max(),
    ).collect().to_dicts()[0]
    dropped = BUILD_LOG["dropped_no_big_blind"] + BUILD_LOG["dropped_no_hand_id"]
    facts["reconciliation"] = {
        "silver_rows": agg["rows"],
        "distinct_hand_uid": agg["distinct"],
        "excess_rows_from_reused_ids": agg["rows"] - agg["distinct"],
        "dropped": dropped,
        "accounted_for": agg["distinct"] + dropped,
        "published": BUILD_LOG["published_hands"],
        "exact": agg["distinct"] + dropped == BUILD_LOG["published_hands"],
        "loss_share": dropped / BUILD_LOG["published_hands"],
        "reconciling_hands": agg["reconciling"],
        "venues": agg["venues"],
        "day_min": agg["first_day"],
        "day_max": agg["last_day"],
    }

    # colliding ids -- the ones features.py drops from both tables
    dup = (h.group_by("hand_uid").agg(pl.len().alias("c")).filter(pl.col("c") > 1))
    dup_stat = dup.select(uids=pl.len(), rows=pl.col("c").sum()).collect().to_dicts()[0]
    facts["colliding_ids"] = {
        "uids": dup_stat["uids"],
        "rows": dup_stat["rows"] or 0,
        "share_of_hands": (dup_stat["rows"] or 0) / agg["rows"],
    }

    # ---- per venue ----------------------------------------------------------
    print("per venue ...", flush=True)
    per_venue = (
        h.group_by("site", "venue")
        .agg(
            hands=pl.len(),
            reconciling=pl.col("rake_bb").is_not_null().sum(),
            last_day=pl.col("day").max(),
            first_day=pl.col("day").min(),
            tables=pl.col("table_id").n_unique(),
            median_bb=pl.col("big_blind").median(),
        )
        .sort("hands", descending=True)
        .collect()
    )
    total_hands = agg["rows"]
    facts["by_venue"] = [
        {**row, "share": row["hands"] / total_hands,
         "reconciling_share": row["reconciling"] / row["hands"]}
        for row in per_venue.to_dicts()
    ]

    # ---- correctness test 1: big-blind VPIP is not ~100% --------------------
    print("vpip checks ...", flush=True)
    hp = scan(SILVER / "hand_players")
    vpip = hp.select(
        seats=pl.len(),
        vpip=pl.col("vpip").mean(),
        pfr=pl.col("pfr").mean(),
        bb_seats=pl.col("is_bb").sum(),
        bb_vpip=pl.col("vpip").filter(pl.col("is_bb")).mean(),
    ).collect().to_dicts()[0]
    facts["vpip_check"] = vpip

    # ---- "no flop, no drop": pre-flop-only hands must rake exactly zero -----
    print("no-flop-no-drop ...", flush=True)
    preflop = (
        h.filter(pl.col("rake_bb").is_not_null() & ~pl.col("saw_flop"))
        .group_by("site", "venue")
        .agg(hands=pl.len(), zero=(pl.col("rake_bb") == 0).sum())
        .sort("hands", descending=True)
        .collect()
    )
    rows = preflop.to_dicts()
    facts["no_flop_no_drop"] = {
        "by_venue": [{**r, "zero_share": r["zero"] / r["hands"]} for r in rows],
        "hands": sum(r["hands"] for r in rows),
        "zero": sum(r["zero"] for r in rows),
        "zero_share": sum(r["zero"] for r in rows) / max(sum(r["hands"] for r in rows), 1),
    }

    # ---- correctness test 2: poker is zero-sum apart from the rake ----------
    if args.fast:
        facts["zero_sum_check"] = {"skipped": True}
    else:
        print("zero-sum identity (full scale, ~25 s) ...", flush=True)
        reconciled = (
            h.filter(pl.col("rake_bb").is_not_null())
            .join(dup.select("hand_uid"), on="hand_uid", how="anti")
        )
        rake_sum = reconciled.select(pl.col("rake_bb").sum()).collect().item()
        net_sum = (
            hp.join(reconciled.select("hand_uid"), on="hand_uid", how="semi")
            .select(pl.col("net_bb").sum()).collect().item()
        )
        facts["zero_sum_check"] = {
            "skipped": False,
            "rake_bb": rake_sum,
            "net_bb": net_sum,
            "ratio": (-net_sum / rake_sum) if rake_sum else None,
            "abs_gap_bb": abs(rake_sum + net_sum),
        }

    # ---- Gold ---------------------------------------------------------------
    print("gold ...", flush=True)
    gold = {}
    for name in ("hand_features", "player_features", "player_lapse",
                 "player_segments", "lapse_scores", "rake_at_risk"):
        table = GOLD / name
        if not table.exists():
            continue
        lf = scan(table)
        gold[name] = {
            "rows": lf.select(pl.len()).collect().item(),
            "cols": len(lf.collect_schema().names()),
            "bytes": du_bytes(table),
        }
    facts["gold"] = gold
    facts["gold_total_bytes"] = sum(t["bytes"] for t in gold.values())

    # the sentence the architecture section is built on
    facts["funnel"] = {
        "seat_rows_in": silver["hand_players"]["rows"],
        "model_table_rows": gold["player_lapse"]["rows"],
        "model_table_bytes": gold["player_lapse"]["bytes"],
        "shrink_x": silver["hand_players"]["bytes"] / gold["player_lapse"]["bytes"],
        "excel_row_limit": 1_048_576,
        "seats_vs_excel": silver["hand_players"]["rows"] / 1_048_576,
        "actions_vs_excel": silver["actions"]["rows"] / 1_048_576,
    }

    facts["measured_seconds"] = round(time.time() - t0, 1)
    facts["polars_version"] = pl.__version__
    OUT.write_text(json.dumps(facts, indent=2, default=str))

    # ---- report -------------------------------------------------------------
    rec = facts["reconciliation"]
    print(f"\nBronze  {facts['bronze']['files']:>10,} files   "
          f"{facts['bronze']['bytes'] / 1e9:>6.1f} GB")
    for name, t in silver.items():
        print(f"Silver  {t['rows']:>10,} rows    {t['bytes'] / 1e6:>6.0f} MB  {name}")
    print(f"        compression {facts['compression_x']:.1f}x")
    print(f"Gold    {gold['player_lapse']['rows']:>10,} rows    "
          f"{gold['player_lapse']['bytes'] / 1e6:>6.0f} MB  player_lapse")
    print(f"\nRECONCILIATION  {rec['distinct_hand_uid']:,} distinct + {rec['dropped']:,} dropped "
          f"= {rec['accounted_for']:,}  vs published {rec['published']:,}   "
          f"{'EXACT' if rec['exact'] else 'MISMATCH'}")
    nfd = facts["no_flop_no_drop"]
    print(f"NO FLOP NO DROP {nfd['zero_share']:.4%} of {nfd['hands']:,} pre-flop-only "
          f"reconciling hands rake exactly $0.00")
    print(f"BB-VPIP CHECK   {vpip['bb_vpip']:.1%} (must be ~31%, never ~100%)   "
          f"pooled VPIP {vpip['vpip']:.1%} · PFR {vpip['pfr']:.1%}")
    z = facts["zero_sum_check"]
    if not z["skipped"]:
        print(f"ZERO-SUM CHECK  rake {z['rake_bb']:,.0f} bb  vs  net {z['net_bb']:,.0f} bb   "
              f"ratio {z['ratio']:.7f}")
    if not rec["exact"]:
        sys.exit("reconciliation is NOT exact -- do not quote it")
    if args.fast:
        print("\nNOTE: --fast skipped the zero-sum identity, so this file is incomplete. "
              "Re-run without --fast before rebuilding the report.")
    print(f"\nwrote {OUT.relative_to(ROOT)}  ({facts['measured_seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
