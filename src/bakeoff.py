"""
Step 5 — the fork bake-off.

Five candidate ML questions, prototyped cheaply on the Gold tables, each scored
against its OWN stated baseline. Nothing here is graded: the winner gets rebuilt
properly in PySpark MLlib in Step 8. This script exists to make one decision well.

    K-Means segmentation      player grain   silhouette vs. cluster readability
    Fork A  lapse             player grain   vs. majority-class predictor
    Fork B  money value       player grain   vs. per-venue mean
    Fork B  next-week volume  player grain   vs. per-venue mean
    Fork C  hand economics    hand   grain   vs. per-stake mean

Every supervised candidate gets TWO algorithms — a linear model and a gradient-boosted
tree — so the "signal" axis is comparable across forks, and so Step 8's "at least two
approaches, compare metrics" requirement is already de-risked here.

Run:  .venv/bin/python src/bakeoff.py
      .venv/bin/python src/bakeoff.py --hand-sample 200000    (faster Fork C)

Writes docs/bakeoff-results.json and prints the scorecard.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
PLAYERS = ROOT / "data" / "gold" / "player_features" / "*.parquet"
HANDS = ROOT / "data" / "gold" / "hand_features" / "*.parquet"
OUT = ROOT / "docs" / "bakeoff-results.json"

# features.py: cutoff = site_last_day - LAPSE_WINDOW
LAPSE_WINDOW = 7

# Fork B money variant: the threshold chosen in this run. Measured alternatives are
# in the scorecard's `threshold_sweep` — >=50 is the knee, halving the small-sample
# tail (6.4% -> 2.8% of players beyond +-200 bb/100) while keeping 74% of the
# money-bearing players. This is a judgement call, recorded on purpose.
MONEY_HANDS_MIN = 50

# Behavioural columns. Every one of these is a LIFETIME average in Gold — it includes
# the final week. Legitimate for the money variant (a contemporaneous description of
# realised value); a known leakage risk for anything forward-looking. Flagged, not hidden.
#
# The split below is by WHICH VENUES ACTUALLY RECORD THE FIELD, which is the only
# honest way to group them on this dataset:
#
#   CORE  — recorded on all six venues. Safe for any all-venue model.
#   STACK — avg_stack_bb. iPoker writes the TOML literal `inf` for every starting
#           stack (verified in the raw .phhs), so all 15,549 of its players carry
#           +inf. Not a big stack: no stack at all.
#   TABLE — max/avg/distinct_tables. features.py already nulls these where a venue
#           records <=1 distinct table id, which is iPoker and only iPoker.
#
# So iPoker is behaviourally complete and economically blind: no stacks, no table
# identity, and no reconciling winnings on any of its 6.0M hands. Models that need
# money or stacks are five-venue questions; models that need only betting behaviour
# keep all six. Imputing any of these would file 15,549 real players under a number
# that was never measured.
CORE = [
    "vpip",
    "pfr",
    "fold_rate",
    "wtsd",
    "aggression_factor",
    "vpip_pfr_gap",
    "avg_invested_bb",
    "distinct_stakes",
    "max_stake",
    "btn_share",
    "bb_vpip_rate",
]
STACK = ["avg_stack_bb"]
TABLE = ["max_tables", "avg_tables", "distinct_tables"]

# Legal ONLY for the money variant, which describes realised value over a player's
# whole recorded history and is not forward-looking. For Fork A and the volume
# variant these are an algebraic leak, not merely a lifetime-average one — see
# FORWARD_LEAKS below.
LIFETIME_VOLUME = ["hands_per_day"]

# Columns that ARE the label for the forward-looking forks, or are arithmetically
# built from it. hands_played = hands_prior + hands_final7, so it is a direct leak.
#
# The subtle ones cost a run to find. hands_per_day = hands_played / days_active, so
# with hands_prior ALSO in the feature set a model can simply recover the label:
#   hands_final7 = hands_per_day * days_active - hands_prior
# That is not a lifetime-average caveat, it is arithmetic. The first pass of this
# script scored Fork A at PR-AUC 0.960 against a 0.625 baseline on exactly that.
# money_coverage = money_hands / hands_played leaks the same denominator.
FORWARD_LEAKS = [
    "hands_final7",
    "hands_played",
    "days_since_last",
    "last_day",
    "days_active",
    "span_days",
    "site_last_day",
    "hands_per_day",
    "money_coverage",
    "money_hands",
    "avg_net_bb",
    "bb_per_100",
]

results: dict = {}


def hr(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


# ----------------------------------------------------------------------------------
# shared scorers
# ----------------------------------------------------------------------------------
def score_regression(name, y_tr, y_te, base_te, preds: dict) -> dict:
    """Every model is scored against `base_te`, the stated dumb baseline.

    skill = 1 - SSE(model)/SSE(baseline). This is the number that matters: plain R2
    measures against the GLOBAL mean, which flatters any model when the baseline is
    already per-group. A skill score at or below 0 means the model adds nothing.
    """
    out = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "baseline": {"rmse": rmse(y_te, base_te), "mae": float(mean_absolute_error(y_te, base_te))},
        "models": {},
    }
    sse_base = float(np.sum((y_te - base_te) ** 2))
    for mname, p in preds.items():
        sse = float(np.sum((y_te - p) ** 2))
        out["models"][mname] = {
            "rmse": rmse(y_te, p),
            "mae": float(mean_absolute_error(y_te, p)),
            "skill_vs_baseline": 1.0 - sse / sse_base if sse_base else float("nan"),
        }
    print(f"  baseline           rmse {out['baseline']['rmse']:9.3f}  mae {out['baseline']['mae']:9.3f}")
    for mname, m in out["models"].items():
        print(
            f"  {mname:18s} rmse {m['rmse']:9.3f}  mae {m['mae']:9.3f}"
            f"  skill vs baseline {m['skill_vs_baseline']:+.4f}"
        )
    return out


def per_venue_mean_baseline(train_df, test_df, site_col, target_col):
    """Baseline = the mean of the target within each venue, learned on TRAIN only."""
    means = train_df.group_by(site_col).agg(pl.col(target_col).mean().alias("_m"))
    grand = float(train_df[target_col].mean())
    joined = test_df.select(site_col).join(means, on=site_col, how="left")
    return joined["_m"].fill_null(grand).to_numpy()


# ----------------------------------------------------------------------------------
# load the shared Gold table
# ----------------------------------------------------------------------------------
def deinf(df: pl.DataFrame, label: str) -> pl.DataFrame:
    """Turn +-inf into null in every float column, and say how much was hit.

    Gold carries +inf wherever iPoker's `starting_stacks = [inf, inf]` was parsed —
    tomllib accepts `inf` as a valid TOML float, so nothing upstream ever raised.
    A null is honest; an infinity silently detonates StandardScaler, Ridge and any
    distance-based method. This is a SYMPTOM PATCH: the real fix belongs in
    features.py, next to the tables_recorded flag it exactly parallels.
    """
    floats = [c for c, t in df.schema.items() if t in (pl.Float64, pl.Float32)]
    hits = {
        c: int(df[c].is_infinite().sum())
        for c in floats
        if df[c].is_infinite().any()
    }
    if hits:
        print(f"  [{label}] +-inf -> null: " + ", ".join(f"{c}={n:,}" for c, n in hits.items()))
    results.setdefault("infinities_found", {})[label] = hits
    return df.with_columns(
        [pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c) for c in floats]
    )


def load_players() -> pl.DataFrame:
    df = deinf(pl.read_parquet(PLAYERS), "player_features")
    # tenure up to the cutoff — legitimately known before the label window opens
    df = df.with_columns(
        (pl.col("site_last_day") - LAPSE_WINDOW - pl.col("first_day")).alias("tenure_days"),
        pl.col("avg_stack_bb").is_not_null().alias("stacks_recorded"),
    )
    return df


# ----------------------------------------------------------------------------------
# candidate 1 — K-Means segmentation (the unsupervised base, always in)
# ----------------------------------------------------------------------------------
def run_kmeans(df: pl.DataFrame) -> dict:
    hr("CANDIDATE 1 · K-Means segmentation  (player grain, unsupervised)")

    # Two runs, because iPoker records neither stacks nor table identity. Imputing
    # either would file 15,549 real players as recreational single-tablers on a median
    # stack — the exact opposite of the truth for the grinders among them, and the
    # multi-tabling signal is the whole point of the ecosystem thesis. So: one run on
    # all six venues using only fields everyone records, and one richer run on the five
    # venues that record stacks and tables.
    feats_all = CORE
    feats_rich = CORE + STACK + TABLE

    out = {"grain": "player", "runs": {}}
    for label, feats, sub in [
        ("all_venues_core_only", feats_all, df),
        ("five_venues_stacks_and_tables", feats_rich, df.filter(pl.col("tables_recorded"))),
    ]:
        X = sub.select(feats).to_numpy()
        X = SimpleImputer(strategy="median").fit_transform(X)
        Xs = StandardScaler().fit_transform(X)
        print(f"\n  {label}: n={len(Xs):,}  features={len(feats)}")
        run = {"n": int(len(Xs)), "features": feats, "k": {}}
        for k in (2, 3, 4, 5, 6, 7, 8):
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(Xs)
            sil = float(
                silhouette_score(Xs, km.labels_, sample_size=10_000, random_state=SEED)
            )
            sizes = np.bincount(km.labels_, minlength=k)
            shares = (sizes / sizes.sum() * 100).round(1).tolist()
            run["k"][k] = {"silhouette": sil, "inertia": float(km.inertia_), "shares_pct": shares}
            print(f"    k={k}  silhouette {sil:.4f}  inertia {km.inertia_:12.0f}  sizes% {shares}")
        out["runs"][label] = run

    # readable centroids at the k the team is most likely to present
    sub = df
    X = SimpleImputer(strategy="median").fit_transform(sub.select(feats_all).to_numpy())
    sc = StandardScaler().fit(X)
    km = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit(sc.transform(X))
    cent = pl.DataFrame(sc.inverse_transform(km.cluster_centers_), schema=feats_all)
    cent = cent.with_columns(
        pl.Series("cluster", range(4)),
        pl.Series("share_pct", (np.bincount(km.labels_, minlength=4) / len(X) * 100).round(1)),
    ).select(["cluster", "share_pct"] + feats_all)
    print("\n  k=4 centroids in real units (the readability test):")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=2):
        print(cent)
    out["k4_centroids"] = cent.to_dicts()
    return out



def leakage_ablation(kind, sub, y, groups, seed=SEED):
    """Split the feature set into 'legitimately prior-window' vs 'lifetime rates'
    and fit the same GBM on each, plus both together.

    Why this matters more than the headline score: every rate column in Gold
    (vpip, pfr, fold_rate, ...) is averaged over a player's WHOLE history, final
    week included. They are not literally known before the cutoff. If the rates
    alone predict the label about as well as the full set, the fork is scoring on
    information it would not have at decision time, and the headline number is
    fiction. hands_prior and tenure_days are the only two features here that are
    honestly prior-window.
    """
    sets = {
        "prior_window_only": ["hands_prior", "tenure_days"],
        "lifetime_rates_only": CORE,
        "both (headline)": CORE + ["hands_prior", "tenure_days"],
    }
    print("\n  LEAKAGE ABLATION — same GBM, three feature sets:")
    out = {}
    for label, feats in sets.items():
        X = sub.select(feats).to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.25, random_state=seed,
            stratify=y if kind == "clf" else groups,
        )
        if kind == "clf":
            m = HistGradientBoostingClassifier(
                class_weight="balanced", random_state=seed, max_iter=200
            ).fit(Xtr, ytr)
            score = float(average_precision_score(yte, m.predict_proba(Xte)[:, 1]))
            out[label] = {"features": len(feats), "pr_auc": score}
            print(f"    {label:22s} ({len(feats):2d} feats)  pr-auc {score:.4f}")
        else:
            m = HistGradientBoostingRegressor(random_state=seed, max_iter=300).fit(Xtr, ytr)
            p = m.predict(Xte)
            sse_b = float(np.sum((yte - ytr.mean()) ** 2))
            score = 1.0 - float(np.sum((yte - p) ** 2)) / sse_b
            out[label] = {"features": len(feats), "r2_vs_global_mean": score}
            print(f"    {label:22s} ({len(feats):2d} feats)  r2 {score:+.4f}")
    return out


# ----------------------------------------------------------------------------------
# candidate 2 — Fork A · lapse classification
# ----------------------------------------------------------------------------------
def run_fork_a(df: pl.DataFrame) -> dict:
    hr("CANDIDATE 2 · Fork A — lapse classification  (player grain)")

    # Restrict to players who existed BEFORE the label window. Someone who first
    # appears inside the final week has not lapsed, they are new — including them
    # would label arrivals as departures.
    sub = df.filter(pl.col("hands_prior") > 0)
    y = (sub["hands_final7"] == 0).cast(pl.Int8).to_numpy()
    # CORE only: Fork A's selling point is ~100% coverage, so it must not depend on
    # fields two of the six venues never wrote.
    feats = CORE + ["hands_prior", "tenure_days"]
    assert not set(feats) & set(FORWARD_LEAKS), "leaky feature in Fork A"

    prevalence = float(y.mean())
    print(f"  n={len(y):,}  lapsed={y.sum():,} ({prevalence:.1%})  active={len(y) - y.sum():,}")
    print("  -> class balance MEASURED, not assumed (fork-specs listed this as open)")

    X = sub.select(feats).to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y
    )

    models = {
        "logistic_l2": Pipeline(
            [
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
                ("m", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED)),
            ]
        ),
        "hist_gbm": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED, max_iter=200
        ),
    }

    # Baseline: the majority-class predictor. It flags everyone as lapsed, so its
    # recall is a perfect 1.0 and its precision is just the prevalence — which is
    # exactly why accuracy alone is not the criterion here.
    base_pred = np.ones_like(yte)
    out = {
        "grain": "player",
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "prevalence_lapsed": prevalence,
        "features": feats,
        "baseline": {
            "rule": "majority class (predict lapsed for everyone)",
            "precision": float(precision_score(yte, base_pred)),
            "recall": float(recall_score(yte, base_pred)),
            "f1": float(f1_score(yte, base_pred)),
            "roc_auc": 0.5,
            "pr_auc": float(yte.mean()),
        },
        "models": {},
    }
    b = out["baseline"]
    print(
        f"\n  baseline (all-lapsed)  prec {b['precision']:.4f}  rec {b['recall']:.4f}"
        f"  f1 {b['f1']:.4f}  roc-auc 0.5000  pr-auc {b['pr_auc']:.4f}"
    )
    for name, m in models.items():
        t = time.time()
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        prob = m.predict_proba(Xte)[:, 1]
        out["models"][name] = {
            "precision": float(precision_score(yte, p)),
            "recall": float(recall_score(yte, p)),
            "f1": float(f1_score(yte, p)),
            "roc_auc": float(roc_auc_score(yte, prob)),
            "pr_auc": float(average_precision_score(yte, prob)),
            "fit_seconds": round(time.time() - t, 1),
        }
        r = out["models"][name]
        print(
            f"  {name:18s}     prec {r['precision']:.4f}  rec {r['recall']:.4f}"
            f"  f1 {r['f1']:.4f}  roc-auc {r['roc_auc']:.4f}  pr-auc {r['pr_auc']:.4f}"
        )

    # the coefficient that goes on a slide
    lr = models["logistic_l2"].named_steps["m"]
    coef = sorted(zip(feats, lr.coef_[0]), key=lambda t: -abs(t[1]))
    out["top_coefficients"] = [{"feature": f, "coef": float(c)} for f, c in coef[:8]]
    print("\n  strongest logistic coefficients (standardised, + => more likely to lapse):")
    for f, c in coef[:8]:
        print(f"    {f:20s} {c:+.3f}")

    out["ablation"] = leakage_ablation("clf", sub, y, sub["site"].to_numpy())
    return out


# ----------------------------------------------------------------------------------
# candidate 3 — Fork B · money value
# ----------------------------------------------------------------------------------
def run_fork_b_money(df: pl.DataFrame) -> dict:
    hr("CANDIDATE 3 · Fork B — player value, MONEY variant  (player grain)")

    # Record what the threshold costs before choosing it.
    sweep = {}
    for thr in (1, 30, 50, 100, 200):
        s = df.filter((pl.col("site") != "IPN") & (pl.col("money_hands") >= thr))
        q = s["bb_per_100"]
        sweep[thr] = {
            "n": int(s.height),
            "median_bb_per_100": float(q.median()),
            "pct_beyond_200": float((q.abs() > 200).mean() * 100),
        }
    print("  threshold sweep on money_hands (iPoker already excluded):")
    for thr, v in sweep.items():
        print(
            f"    >={thr:4d}  n={v['n']:7,}  median {v['median_bb_per_100']:7.2f}"
            f"  beyond +-200: {v['pct_beyond_200']:.1f}%"
        )

    # iPoker reconciles on NONE of its 6.0M hands, so it is excluded outright —
    # this is a five-venue question, and saying so is part of the honesty story.
    sub = df.filter((pl.col("site") != "IPN") & (pl.col("money_hands") >= MONEY_HANDS_MIN))
    dropped = df.height - sub.height
    print(
        f"\n  threshold in use: money_hands >= {MONEY_HANDS_MIN}"
        f"  ->  n={sub.height:,}  (dropped {dropped:,} of {df.height:,} = {dropped / df.height:.1%})"
    )
    print(f"  venues: {sorted(sub['site'].unique().to_list())}")

    # iPoker is already excluded, so stack and table features are fully populated here.
    feats = CORE + STACK + TABLE + LIFETIME_VOLUME
    target = "bb_per_100"

    idx = np.arange(sub.height)
    tr_i, te_i = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=sub["site"].to_numpy())
    train, test = sub[tr_i], sub[te_i]
    ytr, yte = train[target].to_numpy(), test[target].to_numpy()

    base = per_venue_mean_baseline(train, test, "site", target)

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
        ]
    )
    Xtr = train.select(feats + ["site"]).to_pandas()
    Xte = test.select(feats + ["site"]).to_pandas()

    preds = {}
    ridge = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0, random_state=SEED))]).fit(Xtr, ytr)
    preds["ridge"] = ridge.predict(Xte)
    gbm = Pipeline(
        [("pre", pre), ("m", HistGradientBoostingRegressor(random_state=SEED, max_iter=300))]
    ).fit(Xtr, ytr)
    preds["hist_gbm"] = gbm.predict(Xte)

    out = score_regression("fork_b_money", ytr, yte, base, preds)
    out.update(
        {
            "grain": "player",
            "target": target,
            "money_hands_min": MONEY_HANDS_MIN,
            "venues_excluded": ["IPN"],
            "threshold_sweep": sweep,
            "features": feats + ["site"],
            "target_sd": float(np.std(yte)),
            "coverage_pct_of_gold": round(sub.height / df.height * 100, 1),
        }
    )
    print(f"  (target sd on test = {out['target_sd']:.2f} bb/100 — compare against the RMSEs above)")

    # per venue, because coverage and quality differ sharply by venue
    out["per_venue"] = {}
    print("\n  per venue (a single pooled metric would hide the ONG/ABS vs PS/FTP gap):")
    sites = test["site"].to_numpy()
    for s in sorted(set(sites.tolist())):
        m = sites == s
        sse_b = float(np.sum((yte[m] - base[m]) ** 2))
        row = {
            "n": int(m.sum()),
            "baseline_rmse": rmse(yte[m], base[m]),
            "ridge_rmse": rmse(yte[m], preds["ridge"][m]),
            "hist_gbm_rmse": rmse(yte[m], preds["hist_gbm"][m]),
            "hist_gbm_skill": 1.0 - float(np.sum((yte[m] - preds["hist_gbm"][m]) ** 2)) / sse_b,
        }
        out["per_venue"][s] = row
        print(
            f"    {s:4s} n={row['n']:6,}  baseline {row['baseline_rmse']:8.2f}"
            f"  ridge {row['ridge_rmse']:8.2f}  gbm {row['hist_gbm_rmse']:8.2f}"
            f"  gbm skill {row['hist_gbm_skill']:+.4f}"
        )
    return out


# ----------------------------------------------------------------------------------
# candidate 4 — Fork B · next-week volume
# ----------------------------------------------------------------------------------
def run_fork_b_volume(df: pl.DataFrame) -> dict:
    hr("CANDIDATE 4 · Fork B — player value, VOLUME variant  (player grain)")

    sub = df.filter(pl.col("hands_prior") > 0)
    raw = sub["hands_final7"].to_numpy().astype(float)
    zero_share = float((raw == 0).mean())
    print(
        f"  n={sub.height:,}  target hands_final7:  zero on {(raw == 0).sum():,} players"
        f" ({zero_share:.1%})  median {np.median(raw):.0f}  mean {raw.mean():.0f}  max {raw.max():.0f}"
    )
    print("  -> heavily zero-inflated. Modelled on log1p; RMSE also reported in raw hands.")

    # log1p because the raw target spans 0 to ~39k: squared error on the raw scale is
    # entirely decided by a handful of whales, and every model collapses to the mean.
    y = np.log1p(raw)
    feats = CORE + ["hands_prior", "tenure_days"]
    assert not set(feats) & set(FORWARD_LEAKS), "leaky feature in Fork B volume"

    idx = np.arange(sub.height)
    tr_i, te_i = train_test_split(idx, test_size=0.25, random_state=SEED, stratify=sub["site"].to_numpy())
    train, test = sub[tr_i], sub[te_i]
    ytr, yte = y[tr_i], y[te_i]

    train_log = train.with_columns(pl.col("hands_final7").log1p().alias("_y"))
    base = per_venue_mean_baseline(train_log, test, "site", "_y")

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
        ]
    )
    Xtr = train.select(feats + ["site"]).to_pandas()
    Xte = test.select(feats + ["site"]).to_pandas()

    preds = {}
    preds["ridge"] = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0, random_state=SEED))]).fit(Xtr, ytr).predict(Xte)
    preds["hist_gbm"] = (
        Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=SEED, max_iter=300))])
        .fit(Xtr, ytr)
        .predict(Xte)
    )

    out = score_regression("fork_b_volume", ytr, yte, base, preds)
    out.update(
        {
            "grain": "player",
            "target": "log1p(hands_final7)",
            "zero_share": zero_share,
            "features": feats + ["site"],
            "target_sd": float(np.std(yte)),
            "coverage_pct_of_gold": round(sub.height / df.height * 100, 1),
            "leakage_note": (
                "BEHAVIOUR columns are lifetime averages that include the final week, so "
                "they leak the label window. Closing this means extending features.py to "
                "compute rate columns restricted to day <= cutoff. NOT closed in this run."
            ),
        }
    )
    print(f"  (target sd on test = {out['target_sd']:.3f} log-hands)")

    # expm1 is explosive: an unclipped Ridge prediction of ~12.6 log-hands becomes
    # 300k hands and single-handedly dictates the raw-scale RMSE. Clip to the range
    # actually observed in training before inverting — otherwise this diagnostic
    # measures the exponential, not the model.
    lo, hi = float(ytr.min()), float(ytr.max())
    raw_te = raw[te_i]
    print(f"\n  back on the raw hands scale (expm1, predictions clipped to [{lo:.2f}, {hi:.2f}] log-hands):")
    out["raw_scale_rmse"] = {"baseline": rmse(raw_te, np.expm1(np.clip(base, lo, hi)))}
    print(f"    baseline           rmse {out['raw_scale_rmse']['baseline']:10.1f} hands")
    for name, p in preds.items():
        r = rmse(raw_te, np.expm1(np.clip(p, lo, hi)))
        out["raw_scale_rmse"][name] = r
        print(f"    {name:18s} rmse {r:10.1f} hands")

    out["ablation"] = leakage_ablation("reg", sub, y, sub["site"].to_numpy())
    print("  !! LEAKAGE NOT CLOSED — these scores are an optimistic ceiling, not a fair result.")
    return out


# ----------------------------------------------------------------------------------
# candidate 5 — Fork C · hand economics
# ----------------------------------------------------------------------------------
def run_fork_c(n_sample: int) -> dict:
    hr("CANDIDATE 5 · Fork C — hand economics, pot_bb  (hand grain)")

    cols = [
        "site", "day", "hour", "minute_of_day", "is_weekend", "stake", "big_blind",
        "n_players", "seat_count", "avg_stack_bb", "min_stack_bb", "pot_bb",
    ]
    # Excluded on purpose — these settle at the same time as the pot, so they are
    # target leakage: saw_flop, showdown, rake_bb, money_status, money_ok.
    lf = pl.scan_parquet(HANDS).select(cols).filter(pl.col("pot_bb").is_not_null())
    total = int(lf.select(pl.len()).collect().item())
    hands = deinf(lf.collect().sample(n=min(n_sample, total), seed=SEED), "hand_features_sample")
    print(f"  {total:,} hands with a pot; sampled {hands.height:,} ({hands.height / total:.1%}) for the prototype")

    # Split by TIME, not at random — a random shuffle would let the model see the same
    # table-session on both sides. Each venue stops recording on a different day, so the
    # cutoff is measured per venue from that venue's own last observed day.
    last = hands.group_by("site").agg(pl.col("day").max().alias("_last"))
    hands = hands.join(last, on="site").with_columns(
        (pl.col("day") > pl.col("_last") - 4).alias("_test")
    )
    train, test = hands.filter(~pl.col("_test")), hands.filter(pl.col("_test"))
    print(f"  time split: train {train.height:,} (early days)  test {test.height:,} (last 4 days per venue)")

    target = "pot_bb"
    feats = ["day", "hour", "minute_of_day", "is_weekend", "stake", "big_blind",
             "n_players", "seat_count", "avg_stack_bb", "min_stack_bb"]
    ytr, yte = train[target].to_numpy(), test[target].to_numpy()

    # Baseline = mean pot within each stake level. Pot size scales with stake by
    # construction, so a single global mean would make any model look artificially good.
    base = per_venue_mean_baseline(train, test, "stake", target)

    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feats),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site"]),
        ]
    )
    Xtr = train.select(feats + ["site"]).to_pandas()
    Xte = test.select(feats + ["site"]).to_pandas()

    preds = {}
    preds["ridge"] = Pipeline([("pre", pre), ("m", Ridge(alpha=1.0, random_state=SEED))]).fit(Xtr, ytr).predict(Xte)
    t = time.time()
    preds["hist_gbm"] = (
        Pipeline([("pre", pre), ("m", HistGradientBoostingRegressor(random_state=SEED, max_iter=300))])
        .fit(Xtr, ytr)
        .predict(Xte)
    )
    print(f"  (gbm fit {time.time() - t:.1f}s)")

    out = score_regression("fork_c", ytr, yte, base, preds)
    out.update(
        {
            "grain": "hand",
            "target": target,
            "hands_with_pot": total,
            "sampled": int(hands.height),
            "split": "by time — last 4 days per venue held out",
            "features": feats + ["site"],
            "target_sd": float(np.std(yte)),
            "excluded_as_leakage": ["saw_flop", "showdown", "rake_bb", "money_status", "money_ok"],
        }
    )
    print(f"  (target sd on test = {out['target_sd']:.2f} bb)")

    out["per_stake"] = {}
    print("\n  per stake (higher stakes carry more variance):")
    stakes = test["stake"].to_numpy()
    for s in sorted(set(stakes.tolist())):
        m = stakes == s
        if m.sum() < 200:
            continue
        row = {
            "n": int(m.sum()),
            "baseline_rmse": rmse(yte[m], base[m]),
            "hist_gbm_rmse": rmse(yte[m], preds["hist_gbm"][m]),
        }
        out["per_stake"][str(s)] = row
        print(f"    stake {s:6.2f}  n={row['n']:7,}  baseline {row['baseline_rmse']:8.2f}  gbm {row['hist_gbm_rmse']:8.2f}")
    return out


# ----------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand-sample", type=int, default=600_000, help="rows sampled for Fork C")
    ap.add_argument("--skip", nargs="*", default=[], help="candidate keys to skip")
    args = ap.parse_args()

    t0 = time.time()
    hr("STEP 5 · FORK BAKE-OFF — five candidates, each against its own baseline")
    df = load_players()
    print(f"gold/player_features: {df.height:,} rows x {df.width} cols")
    results["gold_players"] = df.height

    jobs = {
        "kmeans": lambda: run_kmeans(df),
        "fork_a_lapse": lambda: run_fork_a(df),
        "fork_b_money": lambda: run_fork_b_money(df),
        "fork_b_volume": lambda: run_fork_b_volume(df),
        "fork_c_pot": lambda: run_fork_c(args.hand_sample),
    }
    for key, fn in jobs.items():
        if key in args.skip:
            print(f"\n(skipping {key})")
            continue
        results[key] = fn()

    # ------------------------------------------------------------------ scorecard
    hr("SCORECARD — measured axes only. Business story & rubric fit are human calls.")
    print(f"{'candidate':26s} {'coverage':>22s}  {'signal vs baseline':>38s}")
    print("-" * 90)

    def line(name, cov, sig):
        print(f"{name:26s} {cov:>22s}  {sig:>38s}")

    if "fork_a_lapse" in results:
        r = results["fork_a_lapse"]
        best = max(r["models"].items(), key=lambda kv: kv[1]["pr_auc"])
        line(
            "Fork A · lapse",
            f"{r['n_train'] + r['n_test']:,} players",
            f"pr-auc {best[1]['pr_auc']:.3f} vs {r['baseline']['pr_auc']:.3f} ({best[0]})",
        )
    if "fork_b_money" in results:
        r = results["fork_b_money"]
        best = max(r["models"].items(), key=lambda kv: kv[1]["skill_vs_baseline"])
        line(
            "Fork B · money",
            f"{r['n_train'] + r['n_test']:,} players / 5 venues",
            f"skill {best[1]['skill_vs_baseline']:+.4f} ({best[0]})",
        )
    if "fork_b_volume" in results:
        r = results["fork_b_volume"]
        best = max(r["models"].items(), key=lambda kv: kv[1]["skill_vs_baseline"])
        line(
            "Fork B · volume",
            f"{r['n_train'] + r['n_test']:,} players (LEAKY)",
            f"skill {best[1]['skill_vs_baseline']:+.4f} ({best[0]})",
        )
    if "fork_c_pot" in results:
        r = results["fork_c_pot"]
        best = max(r["models"].items(), key=lambda kv: kv[1]["skill_vs_baseline"])
        line(
            "Fork C · pot_bb",
            f"{r['hands_with_pot']:,} hands",
            f"skill {best[1]['skill_vs_baseline']:+.4f} ({best[0]})",
        )
    if "kmeans" in results:
        run = results["kmeans"]["runs"]["all_venues_core_only"]
        bk = max(run["k"].items(), key=lambda kv: kv[1]["silhouette"])
        line("K-Means segmentation", f"{run['n']:,} players", f"best silhouette {bk[1]['silhouette']:.3f} at k={bk[0]}")

    print(
        "\nReminder: accuracy is explicitly NOT the criterion. A fork that wins on a third\n"
        "of the data can still lose the scorecard — coverage and business story carry equal weight."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {OUT.relative_to(ROOT)}   ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
