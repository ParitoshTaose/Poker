#!/usr/bin/env python3
"""Gold -> the dashboard payload.

Builds everything `docs/dashboard.html` renders and injects it into that file between
the `/*DATA:START*/` and `/*DATA:END*/` markers, so no figure on the page is ever
hand-typed (delivery gate A5).

Design notes, because two of them are decisions rather than mechanics:

*   **No Spark.** The three Gold tables this reads are 9 MB + 21 MB + 2.5 MB. Polars
    reads them in under a second, which also side-steps the `PYSPARK_PYTHON` trap.
    The one big input is the 2.1 GB seat join, and that is a projected 4-column scan.
*   **The whole population travels with the page.** All 77,268 players are encoded as
    columnar typed arrays in base64 (~1.1 MB) rather than pre-aggregated into a cube.
    The page therefore recomputes every headline, curve and ranking in the browser from
    real Gold rows, so any filter combination is exact instead of interpolated.
*   **Stake is the player's MODAL stake** -- the one they played the most hands at inside
    their venue's own prior window -- not `p_max_stake`. One hand at 1000NL should not
    file a 25NL regular as a high-roller; it differs for 24.8% of players. Cached in
    `data/_work/player_stake.parquet` because it costs a scan of the seat join.

Run:
    .venv/bin/python src/dashboard_data.py                 # reuses the stake cache
    .venv/bin/python src/dashboard_data.py --rebuild-stake # rescans hp_enriched (~6 s)
    .venv/bin/python src/dashboard_data.py --dump out.json # also write the payload out
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "gold"
WORK = ROOT / "data" / "_work"
DOCS = ROOT / "docs"

RAKE = str(GOLD / "rake_at_risk" / "*.parquet")
LAPSE = str(GOLD / "player_lapse" / "*.parquet")
SEGMENTS = str(GOLD / "player_segments" / "*.parquet")
HP = str(WORK / "hp_enriched" / "*.parquet")
STAKE_CACHE = WORK / "player_stake.parquet"

PAGE = DOCS / "dashboard.html"
MARK_A, MARK_B = "/*DATA:START*/", "/*DATA:END*/"

# The venue's own last day of collection. Verified 2026-08-07; every venue is labelled
# 2009-07-01_2009-07-23 in its folder name and three of them overrun it.
VENUE_NAMES = {
    "PS": "PokerStars",
    "PTY": "PartyPoker",
    "ONG": "Ongame Network",
    "ABS": "Absolute Poker",
    "FTP": "Full Tilt Poker",
    "IPN": "iPoker",
}


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------------------------
# modal stake
# ---------------------------------------------------------------------------------
def build_modal_stake(rebuild: bool) -> pl.DataFrame:
    """Stake each player played most inside their venue's own prior window.

    The regression check is the point of doing it here rather than eyeballing it: the
    prior-window hand counts this produces must equal `p_hands` in `gold/player_lapse`
    player by player. If they do not, the stake was measured over different rows than
    the label and the filter would be quietly wrong.
    """
    if STAKE_CACHE.exists() and not rebuild:
        df = pl.read_parquet(STAKE_CACHE)
        say(f"  modal stake      : reused {STAKE_CACHE.relative_to(ROOT)} ({df.height:,} players)")
        return df

    t0 = time.time()
    cut = (
        pl.read_parquet(LAPSE, columns=["site", "cutoff"])
        .group_by("site")
        .agg(pl.first("cutoff"))
    )
    per_stake = (
        pl.scan_parquet(HP)
        .select("site", "player_id", "day", "stake")
        .join(cut.lazy(), on="site", how="inner")
        .filter(pl.col("day") <= pl.col("cutoff"))
        .group_by("site", "player_id", "stake")
        .agg(pl.len().alias("hands"))
        .collect(engine="streaming")
    )
    # most hands wins; ties break to the HIGHER stake, deterministically
    modal = (
        per_stake.sort(["hands", "stake"], descending=[True, True])
        .group_by("site", "player_id")
        .agg(
            pl.first("stake").alias("modal_stake"),
            pl.first("hands").alias("modal_hands"),
            pl.sum("hands").alias("prior_hands"),
            pl.len().alias("n_stakes"),
        )
    )
    STAKE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    modal.write_parquet(STAKE_CACHE)
    say(
        f"  modal stake      : rebuilt from {per_stake['hands'].sum():,} prior-window seat-rows "
        f"-> {modal.height:,} players [{time.time() - t0:.1f}s]"
    )
    return modal


def check_stake_against_gold(modal: pl.DataFrame) -> dict:
    gold = pl.read_parquet(LAPSE, columns=["site", "player_id", "p_hands", "p_distinct_stakes"])
    chk = gold.join(modal, on=["site", "player_id"], how="left")
    out = {
        "players": gold.height,
        "missing": int(chk["prior_hands"].null_count()),
        "hand_count_mismatches": int(chk.filter(pl.col("prior_hands") != pl.col("p_hands")).height),
        "distinct_stake_mismatches": int(
            chk.filter(pl.col("n_stakes") != pl.col("p_distinct_stakes")).height
        ),
    }
    ok = out["missing"] == out["hand_count_mismatches"] == out["distinct_stake_mismatches"] == 0
    say(
        f"  REGRESSION       : {out['players']:,} players · missing {out['missing']} · "
        f"hand-count mismatches {out['hand_count_mismatches']} · "
        f"distinct-stake mismatches {out['distinct_stake_mismatches']}  "
        f"{'PASS' if ok else '*** FAIL ***'}"
    )
    if not ok:
        raise SystemExit("modal stake does not reconcile with gold/player_lapse -- stopping.")
    return out


# ---------------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------------
def b64(arr: np.ndarray) -> str:
    return base64.b64encode(arr.tobytes()).decode("ascii")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild-stake", action="store_true", help="rescan hp_enriched for modal stake")
    ap.add_argument("--dump", metavar="PATH", help="also write the payload as readable JSON")
    ap.add_argument("--no-inject", action="store_true", help="skip writing into docs/dashboard.html")
    args = ap.parse_args()

    t0 = time.time()
    say("READ")

    rake = pl.read_parquet(RAKE)
    say(f"  gold/rake_at_risk: {rake.height:,} players x {rake.width} cols")

    lapse = pl.read_parquet(
        LAPSE,
        columns=[
            "site", "player_id", "p_hands", "p_hands_w1", "p_recency_days", "p_max_tables",
            "p_avg_tables", "tables_recorded", "p_vpip", "p_pfr", "p_money_hands",
            "cutoff", "site_last_day", "site_first_day", "p_tenure_days",
        ],
    )
    say(f"  gold/player_lapse: {lapse.height:,} players")

    modal = build_modal_stake(args.rebuild_stake)
    stake_check = check_stake_against_gold(modal)

    run = json.loads((DOCS / "rake-at-risk-results.json").read_text())
    forka = json.loads((DOCS / "fork-a-results.json").read_text())
    segjson = json.loads((DOCS / "segments-results.json").read_text())
    say("  run logs         : rake-at-risk · fork-a · segments results JSON")

    # ---------------------------------------------------------------- join
    df = (
        rake.join(lapse, on=["site", "player_id"], how="left")
        .join(
            modal.select("site", "player_id", "modal_stake", "modal_hands", "n_stakes"),
            on=["site", "player_id"],
            how="left",
        )
        .sort("expected_rake_at_risk", descending=True)
    )
    if df.height != rake.height:
        raise SystemExit(f"join fanned out: {rake.height:,} -> {df.height:,}")
    for c in ("modal_stake", "segment_id", "venue", "p_recency_days", "p_hands_w1"):
        if df[c].null_count():
            raise SystemExit(f"{df[c].null_count()} nulls in {c} after the join")
    say(f"\nJOIN  {df.height:,} rows · 0 nulls in venue / segment / stake · sorted by expected loss")

    # ---------------------------------------------------------------- dimensions
    # Ordered by share of the money at risk, so index 0 is the venue/segment that
    # matters most rather than whichever sorted first alphabetically.
    v_order = (
        df.group_by("site", "venue")
        .agg(pl.sum("expected_rake_at_risk").alias("r"))
        .sort("r", descending=True)
    )
    venue_codes = v_order["site"].to_list()
    seg_names = {int(k): v for k, v in run["segment_names"].items()}
    s_order = (
        df.group_by("segment_id")
        .agg(pl.sum("expected_rake_at_risk").alias("r"))
        .sort("r", descending=True)
    )
    seg_ids = s_order["segment_id"].to_list()
    stake_levels = sorted(df["modal_stake"].unique().to_list())

    v_idx = {c: i for i, c in enumerate(venue_codes)}
    s_idx = {int(s): i for i, s in enumerate(seg_ids)}
    k_idx = {float(s): i for i, s in enumerate(stake_levels)}
    say(
        "      venues  " + " ".join(f"{i}={c}" for c, i in v_idx.items())
        + "\n      segments " + " ".join(f"{i}={seg_names[s]}" for s, i in s_idx.items())
        + "\n      stakes  " + " ".join(f"{i}={int(bb * 100)}NL" for bb, i in k_idx.items())
    )

    # ---------------------------------------------------------------- encode
    n = df.height
    vi = df["site"].replace_strict(v_idx, return_dtype=pl.UInt8).to_numpy()
    si = df["segment_id"].replace_strict(s_idx, return_dtype=pl.UInt8).to_numpy()
    ki = df["modal_stake"].replace_strict(k_idx, return_dtype=pl.UInt8).to_numpy()
    key = ((vi << 5) | (si << 3) | ki).astype(np.uint8)

    flags = (
        df["lapsed"].to_numpy().astype(np.uint8)
        | ((df["fold"] == "test").to_numpy().astype(np.uint8) << 1)
        | ((df["p_money_hands"].to_numpy() > 0).astype(np.uint8) << 2)
    ).astype(np.uint8)

    # float32 for the three columns that get multiplied and summed. An earlier version
    # stored probability at 1e-4 and dollars in whole cents, which is the natural
    # precision of each -- but 77,268 half-cent roundings walk the headline off by
    # $0.59, and the page would then read $604,162 against the report's $604,163.
    # float32 keeps ~7 significant digits, costs 154 KB, and the drift falls to cents.
    risk = df["risk_cal"].to_numpy().astype(np.float32)
    # Nine players carry a cent-scale negative weekly rake (max -$0.077) and twenty-five
    # a measured figure a few cents above their total. Both are float noise on players
    # with 1-25 hands in the week; carried as-is rather than clamped away.
    rake_c = df["weekly_rake_usd"].to_numpy().astype(np.float32)
    rakem_c = df["weekly_rake_usd_measured"].to_numpy().astype(np.float32)
    # Ship the Gold column rather than recomputing risk x rake in the browser. Gold
    # stores `risk_cal` rounded to 4 dp but `expected_rake_at_risk` was computed from
    # the unrounded probability, so multiplying the two shipped columns back together
    # is off by up to $0.16 a player. A5 names this column as the source; use it.
    exp = df["expected_rake_at_risk"].to_numpy().astype(np.float32)
    rec = np.clip(df["p_recency_days"].to_numpy(), 0, 255).astype(np.uint8)
    hw1 = np.clip(df["p_hands_w1"].to_numpy(), 0, 65_535).astype(np.uint16)
    tabs_raw = df["p_max_tables"].fill_null(255).to_numpy()
    tabs = np.clip(tabs_raw, 0, 255).astype(np.uint8)  # 255 == not recorded
    n_tab_null = int(df["p_max_tables"].null_count())

    cols = {
        "key": b64(key), "flags": b64(flags), "risk": b64(risk),
        "rake": b64(rake_c), "rakem": b64(rakem_c), "exp": b64(exp),
        "rec": b64(rec), "hw1": b64(hw1), "tabs": b64(tabs),
    }
    ids = "\n".join(df["player_id"].to_list())

    # ---------------------------------------------------------------- self-check
    # Decode straight back out of the payload and reconcile against Gold and the run
    # log. This is the check that the page's arithmetic starts from the right numbers.
    d_risk = np.frombuffer(base64.b64decode(cols["risk"]), dtype=np.float32).astype(np.float64)
    d_rake = np.frombuffer(base64.b64decode(cols["rake"]), dtype=np.float32).astype(np.float64)
    d_rakem = np.frombuffer(base64.b64decode(cols["rakem"]), dtype=np.float32).astype(np.float64)
    d_exp = np.frombuffer(base64.b64decode(cols["exp"]), dtype=np.float32).astype(np.float64)

    gold_weekly = float(df["weekly_rake_usd"].sum())
    gold_risk = float(df["expected_rake_at_risk"].sum())
    head = run["headline"]

    order = np.argsort(-d_exp, kind="stable")
    budget = int(round(0.10 * n))
    reach = float(d_exp[order[:budget]].sum())

    checks = [
        ("weekly rake", float(d_rake.sum()), gold_weekly, head["weekly_rake_usd"]),
        ("measured rake", float(d_rakem.sum()), float(df["weekly_rake_usd_measured"].sum()),
         head["weekly_rake_measured_usd"]),
        ("rake at risk", float(d_exp.sum()), gold_risk, head["expected_rake_at_risk_usd"]),
        ("top-10% reach", reach, reach, head["top_budget_reach_usd"]),
    ]
    say("\nSELF-CHECK  decoded payload vs Gold vs the run log")
    say(f"  {'':<14} {'from payload':>15} {'from Gold':>15} {'run log':>15} {'delta':>9}")
    worst = 0.0
    for label, dec, gold, ref in checks:
        delta = dec - ref
        worst = max(worst, abs(delta))
        say(f"  {label:<14} {dec:>15,.2f} {gold:>15,.2f} {ref:>15,.2f} {delta:>9,.2f}")
    say(f"  budget players {budget:>15,} {'':>15} {head['top_budget_players']:>15,}")
    if budget != head["top_budget_players"]:
        raise SystemExit("top-10% contact count does not match the run log")
    if worst > 0.50:
        raise SystemExit(f"payload rounding drifts by ${worst:,.2f} -- raise the encoded precision")
    say(f"  worst drift ${worst:,.4f} on ${head['weekly_rake_usd']:,.0f} (float32 encoding); "
        f"every figure rounds to the run log's dollar -- PASS")

    # The one number that legitimately differs from the run log, stated rather than hidden.
    at_risk = int((d_risk >= 0.50).sum())
    if at_risk != head["at_risk_players"]:
        say(f"\n  NOTE  players at calibrated P >= 0.50: {at_risk:,} re-derived from Gold vs "
            f"{head['at_risk_players']:,} in the run log.")
        say(f"        Gold stores `risk_cal` rounded to 4 dp, so "
            f"{at_risk - head['at_risk_players']} player(s) sitting just under the threshold "
            f"round onto it. {at_risk:,} is the reproducible figure -- it is what anyone re-deriving "
            f"from the published table gets -- so the page shows it.")

    # per-venue and per-segment reconciliation against the run log
    say("\n  per-venue rake at risk, payload vs run log")
    ref_v = {r["venue"]: r["expected_at_risk"] for r in run["by_venue"]}
    for code in venue_codes:
        m = vi == v_idx[code]
        name = VENUE_NAMES[code]
        got, want = float(d_exp[m].sum()), ref_v[name]
        flag = "ok" if abs(got - want) < 5 else "MISMATCH"
        say(f"    {name:<16} {got:>12,.2f}  vs {want:>12,.2f}   {flag}")
        if flag != "ok":
            raise SystemExit(f"{name} does not reconcile")
    ref_s = {r["segment"]: r["expected_at_risk"] for r in run["by_segment"]}
    say("  per-segment rake at risk, payload vs run log")
    for sid in seg_ids:
        m = si == s_idx[int(sid)]
        name = seg_names[int(sid)]
        got, want = float(d_exp[m].sum()), ref_s[name]
        flag = "ok" if abs(got - want) < 5 else "MISMATCH"
        say(f"    {name:<16} {got:>12,.2f}  vs {want:>12,.2f}   {flag}")
        if flag != "ok":
            raise SystemExit(f"{name} does not reconcile")

    # ---------------------------------------------------------------- dimensions out
    win = (
        lapse.group_by("site")
        .agg(pl.first("site_first_day"), pl.first("site_last_day"), pl.first("cutoff"))
        .to_dicts()
    )
    win = {w["site"]: w for w in win}
    cover = {c["site"]: c for c in run["money_coverage_by_venue"]}
    share = {s["site"]: s for s in run["rake_share_by_venue"]}

    venues = []
    for code in venue_codes:
        sub = df.filter(pl.col("site") == code)
        venues.append({
            "code": code, "name": VENUE_NAMES[code], "players": sub.height,
            "first_day": win[code]["site_first_day"], "last_day": win[code]["site_last_day"],
            "cutoff": win[code]["cutoff"],
            "hands": cover[code]["hands"], "coverage": cover[code]["coverage"],
            "median_share": share.get(code, {}).get("median_share"),
            "p90_share": share.get(code, {}).get("p90_share"),
        })

    seg_centroids = {int(s["prediction"]): s for s in segjson["segments"]["ecosystem_five_venues"]}
    segments = []
    for sid in seg_ids:
        c = seg_centroids.get(int(sid), {})
        segments.append({
            "id": int(sid), "name": seg_names[int(sid)],
            "hands": c.get("p_hands"), "tables": c.get("p_max_tables"),
            "vpip": c.get("p_vpip"), "pfr": c.get("p_pfr"),
            "bb100": c.get("p_bb_per_100"), "share_pct": c.get("share_pct"),
        })

    stakes = [{"bb": float(bb), "label": f"{int(round(bb * 100))}NL"} for bb in stake_levels]

    ipn = cover["IPN"]
    excluded = {
        "code": "IPN", "name": "iPoker",
        "players": int(lapse.filter(pl.col("site") == "IPN").height),
        "hands": ipn["hands"], "reconciling": ipn["reconciling"],
        "table_ids": 1,
        "first_day": win["IPN"]["site_first_day"], "last_day": win["IPN"]["site_last_day"],
    }

    payload = {
        "meta": {
            "generated": date.today().isoformat(),
            "script": "src/dashboard_data.py",
            "n": n,
            "sources": [
                {"path": "data/gold/rake_at_risk", "rows": rake.height, "cols": rake.width},
                {"path": "data/gold/player_lapse", "rows": lapse.height},
                {"path": "data/gold/player_segments", "rows": pl.read_parquet(SEGMENTS, columns=["site"]).height},
                {"path": "data/_work/hp_enriched", "rows": stake_check["players"], "note": "modal stake"},
                {"path": "docs/rake-at-risk-results.json"},
                {"path": "docs/fork-a-results.json"},
                {"path": "docs/segments-results.json"},
            ],
            "stake_check": stake_check,
            "tables_null": n_tab_null,
            "worst_drift_usd": round(worst, 4),
            "encoding": "little-endian typed arrays, base64; risk and dollars as float32",
        },
        "cols": cols,
        "ids": ids,
        "venues": venues,
        "segments": segments,
        "stakes": stakes,
        "excluded": excluded,
        "run": run,
        "forkA": {"population": forka["population"], "core": forka["core_all_venues"]},
        "seg": {
            "algorithm_comparison": segjson["algorithm_comparison"],
            "sweeps": segjson["sweeps"],
            "k_chosen": segjson["k_chosen"],
            "archetype_match": segjson["archetype_match"],
        },
    }

    blob = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    say(f"\nPAYLOAD  {len(blob) / 1e6:.2f} MB  "
        f"(arrays {sum(len(v) for v in cols.values()) / 1e6:.2f} MB · "
        f"ids {len(ids) / 1e6:.2f} MB · "
        f"aggregates + run logs {(len(blob) - sum(len(v) for v in cols.values()) - len(ids)) / 1e3:.0f} KB)")

    if args.dump:
        Path(args.dump).write_text(blob)
        say(f"  dumped -> {args.dump}")

    if not args.no_inject:
        if not PAGE.exists():
            raise SystemExit(f"{PAGE} does not exist yet -- create it with the {MARK_A} markers first")
        html = PAGE.read_text()
        i, j = html.find(MARK_A), html.find(MARK_B)
        if i < 0 or j < 0 or j < i:
            raise SystemExit(f"{PAGE.name} is missing the {MARK_A} ... {MARK_B} markers")
        before, after = html[: i + len(MARK_A)], html[j:]
        PAGE.write_text(f"{before}\nconst DASH = {blob};\n{after}")
        size = PAGE.stat().st_size
        say(f"  injected -> docs/{PAGE.name}  ({size / 1e6:.2f} MB on disk)")

    say(f"\ndone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main())
