"""Bronze -> Silver: parse every .phhs file into three Parquet tables.

Batches files instead of holding the whole dataset in memory at once.
Measured on this machine: parsing 200 real files keeps ~1 GB of Python
objects in RAM before they're written out. data/bronze/ holds 21,782 files
(the full real-money dataset, not a small sample) -- doing it in one shot
would need on the order of 100+ GB of RAM. BATCH=500 keeps peak memory to a
few GB regardless of how many files there are in total.

Each table is written as many small Parquet "part" files under its own
folder (data/silver/hands/part-0000.parquet, part-0001.parquet, ...) rather
than one single file. Polars/Spark read a folder of part files exactly like
one table -- this is the same partitioning idea already used for Bronze.
"""
import shutil
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import polars as pl
from parse_phh import parse_file

SRC = Path("data/bronze/handhq")
OUT = Path("data/silver")
BATCH = 500          # ~2.5-3 GB peak RSS per batch, measured

TABLES = ["hands", "hand_players", "actions"]

# Explicit schemas — do NOT let Polars infer these.
# TOML stores `25` as int and `25.0` as float, so the same field arrives as both
# types depending on the hand. Inference samples the first rows, picks one type,
# then dies mid-run when the other shows up (this cost two full-length runs).
# Declaring the schema makes the write deterministic and fails loudly at the source.
S = pl.String, pl.Float64, pl.Int32, pl.Boolean
_STR, _F64, _I32, _BOOL = S

SCHEMAS = {
    "hands": {
        "hand_uid": _STR, "hand_id": _STR, "site": _STR, "venue": _STR,
        "venue_dir": _STR, "stake": _F64, "big_blind": _F64, "table_id": _STR,
        "year": _I32, "month": _I32, "day": _I32, "time": _STR,
        "n_players": _I32, "seat_count": _I32, "saw_flop": _BOOL, "showdown": _BOOL,
        "pot": _F64, "pot_bb": _F64, "rake": _F64, "rake_bb": _F64,
    },
    "hand_players": {
        "hand_uid": _STR, "hand_id": _STR, "player_id": _STR, "position_idx": _I32,
        "is_sb": _BOOL, "is_bb": _BOOL, "is_button": _BOOL, "starting_stack": _F64,
        "invested": _F64, "invested_bb": _F64, "winnings": _F64, "net_bb": _F64,
        "vpip": _BOOL, "pfr": _BOOL, "folded": _BOOL, "showed": _BOOL,
        "postflop_bets": _I32, "postflop_calls": _I32,
    },
    "actions": {
        "hand_uid": _STR, "hand_id": _STR, "player_id": _STR, "street": _STR,
        "action": _STR, "amount": _F64, "amount_bb": _F64, "action_order": _I32,
    },
}


def safe(p):
    try:
        return parse_file(p)
    except Exception as e:
        print("SKIP", p.name, e)
        return None


if __name__ == "__main__":
    files = sorted(SRC.rglob("*.phhs"))
    print(len(files), "files")
    # Part names are deterministic (part-{batch}), so a re-run that dies halfway
    # would leave new parts 0..k next to stale parts k+1..n in the same folder --
    # a table that is silently half one parser version and half another. Clear
    # first: a run that fails loudly beats a table that lies quietly.
    for name in TABLES:
        if (OUT / name).exists():
            shutil.rmtree(OUT / name)
        (OUT / name).mkdir(parents=True)

    skipped = 0
    row_totals = {name: 0 for name in TABLES}
    drops = Counter()

    with Pool(8) as pool:                      # 8 of the 10 cores
        for b in range(0, len(files), BATCH):
            batch = files[b:b + BATCH]
            results = pool.map(safe, batch, chunksize=8)
            skipped += sum(1 for r in results if r is None)
            ok = [r for r in results if r is not None]
            for res in ok:
                drops.update(res[3])
            for i, name in enumerate(TABLES):
                rows = [row for res in ok for row in res[i]]
                if rows:
                    part = b // BATCH
                    (pl.DataFrame(rows, schema=SCHEMAS[name])
                       .write_parquet(OUT / name / f"part-{part:04d}.parquet"))
                row_totals[name] += len(rows)
            done = b + len(batch)
            print(f"batch {b // BATCH}: {done}/{len(files)} files", flush=True)

    print("SKIPPED FILES:", skipped)
    print("DROPPED HANDS:", dict(drops) or "none")
    for name in TABLES:
        print(name, row_totals[name], "rows")
