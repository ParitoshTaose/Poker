# `data/` — the data lake

**Nothing in here is committed to git.** The folder structure is (via `.gitkeep`); the contents
are not. The dataset is 1.71 GB for the GitHub subset and 20,289,230,983 bytes for the full
Zenodo archive — far past anything that belongs in a repo.

## Layout

| Folder | Contents | Written by |
|---|---|---|
| `bronze/` | `.phhs` files copied **untouched**, `handhq/venue=<SITE>/stake=<BB>/` | download step |
| `silver/` | `hands.parquet`, `hand_players.parquet`, `actions.parquet` | `src/parse_phh.py` → `build_silver.py` |
| `gold/` | `player_features.parquet` — one row per player, ~200 MB | `features.py` (PySpark) |

Bronze is never edited. If a parsing bug is found, Silver and Gold are rebuilt from Bronze —
that is the entire reason Bronze exists.

## Getting the data

**Route A — GitHub subset (1.71 GB). Use this for development.**

```bash
git clone --filter=blob:none --sparse https://github.com/uoftcprg/phh-dataset
cd phh-dataset
git sparse-checkout set data/handhq
```

**Route B — full Zenodo archive (20.3 GB). Only when running at full scale.**

DOI `10.5281/zenodo.17136841`. Extract **selectively** — never blind:

```bash
unzip <archive>.zip '*/data/handhq/*'
```

> **Use `data/handhq/` only.** The archive also contains 619 M Annual Computer Poker Competition
> hands — bots playing bots. Enormous, and useless for a business project about human players.

## Folder names carry free metadata

```
{SITE}-{start}_{end}_{stake}NLH_OBFU/{big_blind}/*.phhs
```

Parse the path and you get venue, date range, and stake without opening a single file. This is
why Bronze is partitioned by `venue` and `stake` — partition pruning means a query for one site
never reads the other five.
