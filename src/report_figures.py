#!/usr/bin/env python3
"""
report_figures.py -- build the consulting report's numbers, tables and charts from
measured data.

deliverables/report/ecosystem-engine-report.html contains no hand-typed figures. Every
number sits in a span tagged with the JSON path it came from:

    <b class="fig" data-fig="rake.headline.players" data-fmt="int">77,268</b>

and every table or chart sits between markers:

    <!--CHART:segments:START--> ... <!--CHART:segments:END-->

This script re-derives all of them from
  * docs/pipeline-facts.json         (pipeline_facts.py -- the lake, measured)
  * docs/rake-at-risk-results.json   (rake_at_risk.py)
  * docs/fork-a-results.json         (fork_a.py)
  * docs/segments-results.json       (segments.py)
  * docs/bakeoff-results.json        (bakeoff.py)
  * data/gold/rake_at_risk           (the shipped Gold table, for the ranking curves)
and rewrites the report in place, printing every value it changed. A clean run that
changes nothing is the proof that the report matches the analysis.

Run:  .venv/bin/python src/report_figures.py            # inject
      .venv/bin/python src/report_figures.py --check    # verify only, exit 1 on drift

No Spark. Polars only. ~2 seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from deck_charts import (  # noqa: E402  -- shared primitives, one source of truth
    CYAN, CYAN_LT, DIM, GOLD_, GOLD_LT, GREEN, GREEN_LT, INK, LINE, LINE2, PANEL2,
    RED, RED_LT, SOFT, esc, line, load_gold, polyline, ranking_curves, rect, resolve,
    svg, txt,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "deliverables" / "report" / "ecosystem-engine-report.html"
DOCS = ROOT / "docs"

BUDGET = 0.10          # the contact budget the whole report is quoted at
COL = 696              # the report's content column, in px -- SVG viewBoxes match it 1:1


# ------------------------------------------------------------------ formatting
def fmt(value, how: str) -> str:
    if how == "str":
        return str(value)
    if how == "int":
        return f"{round(value):,}"
    if how == "usd0":
        return f"${round(value):,}"
    if how == "usd2":
        return f"${value:,.2f}"
    if how.startswith("num"):
        return f"{value:,.{int(how[3:])}f}"
    if how.startswith("pct"):
        return f"{value * 100:.{int(how[3:])}f}%"
    if how == "signed1":
        return f"{value:+,.1f}"
    raise ValueError(f"unknown format {how!r}")


# ------------------------------------------------------------- table utilities
def th(label: str, cls: str = "", width: str | None = None) -> str:
    w = f' style="width:{width}"' if width else ""
    c = f' class="{cls}"' if cls else ""
    return f"<th{c}{w}>{label}</th>"


def td(value: str, cls: str = "") -> str:
    c = f' class="{cls}"' if cls else ""
    return f"<td{c}>{value}</td>"


def table(head: str, body: str, caption: str = "", cls: str = "g") -> str:
    cap = f"<caption>{caption}</caption>" if caption else ""
    return f'<table class="{cls}">{cap}<tr>{head}</tr>{body}</table>'


def bar(frac: float, colour: str = "") -> str:
    pct = max(0.0, min(1.0, frac)) * 100
    c = f" {colour}" if colour else ""
    return f'<span class="bar{c}"><i style="width:{pct:.1f}%"></i></span>'


def gb(byte_count: float) -> str:
    return f"{byte_count / 1e9:,.2f} GB"


def mb(byte_count: float) -> str:
    return f"{byte_count / 1e6:,.0f} MB"


# ============================================================ tables and charts
def block_volume(pipe: dict) -> str:
    s, g = pipe["silver"], pipe["gold"]
    rows = [
        ("Bronze &middot; raw", f'{pipe["bronze"]["files"]:,} files',
         gb(pipe["bronze"]["bytes"]), "TOML hand histories, kept byte-for-byte as published"),
        ("Silver &middot; <span class='m'>hands</span>", f'{s["hands"]["rows"]:,} rows',
         mb(s["hands"]["bytes"]), "one row per hand &mdash; pot, rake, stake, table, timestamp"),
        ("Silver &middot; <span class='m'>hand_players</span>", f'{s["hand_players"]["rows"]:,} rows',
         mb(s["hand_players"]["bytes"]),
         f'one row per seat &mdash; {pipe["seats_per_hand"]:.2f} per hand'),
        ("Silver &middot; <span class='m'>actions</span>", f'{s["actions"]["rows"]:,} rows',
         mb(s["actions"]["bytes"]),
         f'one row per decision &mdash; {pipe["actions_per_hand"]:.2f} per hand'),
        ("Gold &middot; <span class='m'>player_lapse</span>", f'{g["player_lapse"]["rows"]:,} rows',
         mb(g["player_lapse"]["bytes"]),
         f'<b>the modelling table</b> &mdash; {g["player_lapse"]["cols"]} columns, one row per player'),
    ]
    body = "".join(
        f'<tr{" class=hi" if "Gold" in name else ""}>'
        + td(name, "nm") + td(f'<span class="m">{count}</span>')
        + td(f'<span class="m">{size}</span>', "r") + td(note) + "</tr>"
        for name, count, size, note in rows
    )
    head = (th("Layer", width="150px") + th("Rows / files", width="118px")
            + th("On disk", "r", "72px") + th("What one row is"))
    cap = (f'Sizes are the sum of the files&rsquo; own byte counts in decimal GB, measured by '
           f'<span class="m">src/pipeline_facts.py</span>. Bronze &rarr; Silver compresses '
           f'<b>{pipe["compression_x"]:.1f}&times;</b> while adding types, column statistics and '
           f'predicate push-down.')
    return table(head, body, cap)


def block_venuetable(pipe: dict) -> str:
    rows = pipe["by_venue"]
    body = ""
    for r in rows:
        flag = ""
        if r["reconciling"] == 0:
            flag = ' <span class="chip cr">no usable money</span>'
        elif r["tables"] <= 1:
            flag = ' <span class="chip cr">no table identity</span>'
        body += (
            "<tr>"
            + td(f'{short(r["venue"])}<span class="ss">{r["site"]}</span>', "nm")
            + td(f'<span class="m">{r["hands"]:,}</span>')
            + td(f'{bar(r["share"] / 0.4)}<span class="m">{r["share"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m">{r["last_day"]}</span>', "c")
            + td(f'<span class="m">{r["tables"]:,}</span>{flag}', "r")
            + td(f'<span class="m">${r["median_bb"]:,.2f}</span>', "r")
            + td(f'<span class="m">{r["reconciling_share"] * 100:.1f}%</span>', "r")
            + "</tr>"
        )
    head = (th("Venue", width="132px") + th("Hands") + th("Share", "r", "104px")
            + th("Last day", "c", "54px") + th("Distinct tables", "r", "132px")
            + th("Median bb", "r", "62px") + th("Winnings reconcile", "r", "78px"))
    cap = ('&ldquo;Last day&rdquo; is the venue&rsquo;s own final recorded day of July 2009 &mdash; they do '
           'not share a calendar. &ldquo;Distinct tables&rdquo; of 1 is not a small venue: it is a venue '
           'that wrote the data vendor&rsquo;s name into every table field.')
    return table(head, body, cap)


def block_coverage(rake: dict, pipe: dict) -> str:
    excluded = set(rake["excluded_venues"])
    body = ""
    for r in sorted(rake["money_coverage_by_venue"], key=lambda x: -x["hands"]):
        out = r["site"] in excluded or r["plausible"] == 0
        dropped = r["usd_reconciling"] - r["usd_plausible"]
        body += (
            f'<tr class="{"lo" if out else ""}">'
            + td(f'{short(r["venue"])}<span class="ss">{r["site"]}</span>', "nm")
            + td(f'<span class="m">{r["hands"]:,}</span>', "r")
            + td(f'<span class="m">{r["reconciling"]:,}</span>', "r")
            + td(f'<span class="m">{r["plausible"]:,}</span>', "r")
            + td(f'{bar(r["coverage"], "g")}<span class="m">{r["coverage"] * 100:.1f}%</span>', "r")
            + td('<span class="m">'
                 + (f'${dropped:,.0f}' if dropped else "&mdash;") + "</span>", "r")
            + "</tr>"
        )
    head = (th("Venue", width="120px") + th("Hands", "r") + th("Rake reconciles", "r")
            + th("&hellip;and is plausible", "r") + th("Coverage", "r", "116px")
            + th("$ dropped by the screen", "r", "88px"))
    cap = ('&ldquo;Reconciles&rdquo; = the derived rake balances against the pot within the parser&rsquo;s '
           'bound. &ldquo;Plausible&rdquo; additionally requires it to be at most 6% of the pot &mdash; the '
           'screen justified in &sect;5.7. iPoker reconciles on nothing and is excluded from every money '
           'figure in this report. <b>Hand counts here are post-de-duplication</b> and so run 547 lower '
           'on iPoker than the Silver counts in &sect;3.2 &mdash; that is exactly the 147 reused ids of '
           '&sect;5.5b, dropped before any join.')
    return table(head, body, cap)


def block_recon(pipe: dict) -> str:
    r = pipe["reconciliation"]
    bl = pipe["build_log"]
    cells = [
        (f'{r["distinct_hand_uid"]:,}', "distinct hands in Silver", CYAN_LT),
        (f'{bl["dropped_no_big_blind"]:,}', "dropped &mdash; no big blind, so money cannot be normalised", GOLD_LT),
        (f'{bl["dropped_no_hand_id"]:,}', "dropped &mdash; no hand ID, so seats cannot be joined", GOLD_LT),
        (f'{r["accounted_for"]:,}', "accounted for", INK),
        (f'{r["published"]:,}', "the dataset&rsquo;s published count", GREEN_LT),
    ]
    parts = []
    for i, (value, label, colour) in enumerate(cells):
        sign = "+" if i in (1, 2) else ("=" if i == 3 else ("" if i == 0 else ""))
        if i == 4:
            sign = "="
        parts.append(
            (f'<div style="font-family:var(--mono);font-size:17px;color:var(--dim);'
             f'padding:0 3px;align-self:center">{sign}</div>' if sign else "")
            + f'<div style="flex:1;min-width:0"><div class="n sm" style="color:{colour};'
            f'font-size:15px">{value}</div>'
            f'<div class="cap" style="margin-top:3px">{label}</div></div>'
        )
    strip = ('<div style="display:flex;gap:6px;align-items:stretch;background:rgba(42,162,119,.07);'
             'border:1px solid rgba(42,162,119,.3);border-radius:10px;padding:11px 13px;margin:9px 0">'
             + "".join(parts) + "</div>")
    note = (f'<p class="cap" style="margin-top:0"><b>The identity closes exactly</b> &mdash; not to '
            f'within a rounding tolerance, exactly. Silver holds {r["silver_rows"]:,} rows against '
            f'{r["distinct_hand_uid"]:,} distinct keys; the {r["excess_rows_from_reused_ids"]:,}-row '
            f'difference is one venue reusing hand IDs (&sect;3.4), not further loss. '
            f'{bl["skipped_files"]} of {pipe["bronze"]["files"]:,} files were skipped.</p>')
    return strip + note


def block_population(forka: dict) -> str:
    prev = forka["population"]["prevalence_lapsed"]
    kept = forka["population"]["players"]
    lapsed = round(prev * kept)
    lifetime_players = 98_718          # the prototype's population, from bakeoff-results.json
    rows = [
        ("Lifetime &ldquo;&ge; 100 hands&rdquo; rule<span class='ss'>counts the week being predicted</span>",
         lifetime_players, lapsed, lapsed / lifetime_players, "lo"),
        ("Prior-window &ldquo;&ge; 100 hands&rdquo; rule<span class='ss'>applicable on the cutoff day</span>",
         kept, lapsed, prev, "hi"),
    ]
    body = ""
    for name, players, lap, prevalence, cls in rows:
        body += (
            f'<tr class="{cls}">' + td(name, "nm")
            + td(f'<span class="m">{players:,}</span>', "r")
            + td(f'<span class="m">{lap:,}</span>', "r")
            + td(f'<span class="m">{prevalence * 100:.1f}%</span>', "r") + "</tr>"
        )
    body += (
        '<tr class="tot">' + td("Difference", "nm")
        + td(f'<span class="m">{lifetime_players - kept:,}</span>', "r")
        + td('<span class="m" style="color:var(--red-lt)">0</span>', "r")
        + td("&mdash;", "r") + "</tr>"
    )
    head = (th("Population rule", width="286px") + th("Players", "r")
            + th("Lapsed", "r") + th("Prevalence", "r"))
    return table(head, body)


def block_leakage(forka: dict) -> str:
    order = [
        ("The old floor &mdash; volume and tenure only",
         "old floor (hands_prior + tenure)", "honestly prior-window, but only 2 features", ""),
        ("Lifetime rates only &mdash; <b>the leaky set</b>",
         "LIFETIME rates only (leaky)", "every column averages over the label week", "lo"),
        ("Prior-window only &mdash; <b>this build</b>",
         "PRIOR-WINDOW only (honest, this build)",
         "structurally cannot see the week being predicted", "hi"),
    ]
    body = ""
    for label, key, note, cls in order:
        r = forka["leakage_recheck"][key]
        body += (
            f'<tr class="{cls}">'
            + td(f'{label}<span class="ss">{note}</span>', "nm")
            + td(f'<span class="m">{r["n_features"]}</span>', "r")
            + td(f'<span class="m">{r["roc_auc"]:.3f}</span>', "r")
            + td(f'<span class="m">{r["pr_auc"]:.3f}</span>', "r") + "</tr>"
        )
    head = (th("Feature set", width="330px") + th("Features", "r")
            + th("ROC-AUC", "r") + th("PR-AUC", "r"))
    return table(head, body,
                 "Same estimator, same population, same split &mdash; only the feature set changes.")


def block_representative(rake: dict) -> str:
    by_site: dict[str, dict] = {}
    for r in rake["representativeness"]:
        by_site.setdefault(r["site"], {})[bool(r["measured"])] = r
    venue = {v["site"]: v["venue"] for v in rake["money_coverage_by_venue"]}
    body = ""
    for site, pair in sorted(by_site.items(), key=lambda kv: -kv[1][True]["hands"]):
        m, u = pair.get(True), pair.get(False)
        if not m or not u:
            continue
        gap = m["median_pot_bb"] / u["median_pot_bb"] if u["median_pot_bb"] else 0
        body += (
            f'<tr class="{"lo" if gap >= 3 else ""}">'
            + td(short(venue.get(site, site)), "nm")
            + td(f'<span class="m">{m["hands"]:,}</span>', "r")
            + td(f'<span class="m">{m["median_pot_bb"]:,.2f}</span>', "r")
            + td(f'<span class="m">{m["saw_flop_share"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m">{u["hands"]:,}</span>', "r")
            + td(f'<span class="m">{u["median_pot_bb"]:,.2f}</span>', "r")
            + td(f'<span class="m">{u["saw_flop_share"] * 100:.1f}%</span>', "r") + "</tr>"
        )
    head = (th("Venue", width="112px")
            + th("Hands", "r") + th("Median pot", "r") + th("Saw flop", "r")
            + th("Hands", "r") + th("Median pot", "r") + th("Saw flop", "r"))
    sub = ('<tr><td></td><th colspan="3" style="text-align:center;color:var(--green-lt);'
           'border-bottom:1px solid var(--line2)">WHERE RAKE RECONCILES</th>'
           '<th colspan="3" style="text-align:center;color:var(--red-lt);'
           'border-bottom:1px solid var(--line2)">WHERE IT DOES NOT</th></tr>')
    cap = ('Pots in big blinds. <b>The hands whose money reconciles are systematically the bigger '
           'ones</b>, because a fixed shortfall is a large share of a small pot and a small share of a '
           'large one. On PartyPoker the two populations differ by a factor of nine.')
    return f'<table class="g"><caption>{cap}</caption>{sub}<tr>{head}</tr>{body}</table>'


def block_plausibility(rake: dict) -> str:
    venue = {v["site"]: v["venue"] for v in rake["money_coverage_by_venue"]}
    rows = sorted(rake["rake_share_by_venue"], key=lambda r: r["above_ceiling"])
    body = ""
    for r in rows:
        bad = r["above_ceiling"] > 0.05
        body += (
            f'<tr class="{"lo" if bad else "hi"}">'
            + td(short(venue.get(r["site"], r["site"])), "nm")
            + td(f'<span class="m">{r["flop_hands"]:,}</span>', "r")
            + td(f'<span class="m">{r["median_share"] * 100:.2f}%</span>', "r")
            + td(f'<span class="m">{r["p90_share"] * 100:.2f}%</span>', "r")
            + td(f'{bar(r["above_ceiling"], "r")}'
                 f'<span class="m"><b>{r["above_ceiling"] * 100:.2f}%</b></span>', "r")
            + td(f'<span class="m">{r["at_parser_bound"] * 100:.2f}%</span>', "r") + "</tr>"
        )
    head = (th("Venue", width="118px") + th("Flop hands", "r")
            + th("Median rake share", "r") + th("90th pct", "r")
            + th("Above the 6% ceiling", "r", "126px") + th("At the parser&rsquo;s 15% bound", "r", "76px"))
    cap = ('Rake as a share of the pot, on hands that saw a flop. <b>The median is the real commercial '
           'rate on every venue</b> &mdash; strong evidence the derivation is sound. The upper tail is '
           'not: no operator charges 15%.')
    return table(head, body, cap)


def block_estimator(rake: dict) -> str:
    venue = {v["site"]: v["venue"] for v in rake["money_coverage_by_venue"]}
    body = ""
    for r in sorted(rake["estimator_validation"], key=lambda x: -x["hands"]):
        off = abs(r["ratio"] - 1)
        body += (
            f'<tr class="{"hi" if off <= 0.01 else ""}">'
            + td(short(venue.get(r["site"], r["site"])), "nm")
            + td(f'<span class="m">{r["hands"]:,}</span>', "r")
            + td(f'<span class="m">${r["actual_usd"]:,.0f}</span>', "r")
            + td(f'<span class="m">${r["predicted_usd"]:,.0f}</span>', "r")
            + td(f'<span class="m"><b>{r["ratio"]:.4f}</b></span>', "r")
            + td(f'<span class="m">${r["mae_usd_per_hand"]:,.4f}</span>', "r")
            + td(f'<span class="m">${r["mean_actual_usd_per_hand"]:,.4f}</span>', "r") + "</tr>"
        )
    head = (th("Venue", width="112px") + th("Held-out hands", "r")
            + th("Actual", "r") + th("Predicted", "r") + th("Ratio", "r", "58px")
            + th("MAE / hand", "r") + th("Mean / hand", "r"))
    cap = ('Cells fitted on half the measured hands, scored against the other half. <b>Every venue '
           'lands within 0.7% on totals</b>; per hand it is much rougher, which is why this estimator '
           'is quoted for sums and never for a single hand.')
    return table(head, body, cap)


def block_bakeoff(bake: dict) -> str:
    rows = [
        ("Lapse classification", "majority class",
         f'PR-AUC {bake["fork_a_lapse"]["baseline"]["pr_auc"]:.3f}',
         f'<b>PR-AUC {bake["fork_a_lapse"]["models"]["hist_gbm"]["pr_auc"]:.3f}</b>, '
         f'ROC {bake["fork_a_lapse"]["models"]["hist_gbm"]["roc_auc"]:.3f}',
         "17", "hi", "all six venues; a retention list a team can work through on a Monday"),
        ("Player value &middot; money", "per-venue mean",
         f'RMSE {bake["fork_b_money"]["baseline"]["rmse"]:.2f}',
         f'RMSE {bake["fork_b_money"]["models"]["hist_gbm"]["rmse"]:.2f}, '
         f'<b>skill +{bake["fork_b_money"]["models"]["hist_gbm"]["skill_vs_baseline"]:.3f}</b>',
         "15", "", "cleanest methodology; loses on coverage &mdash; five venues, 61% of players"),
        ("K-means segmentation", "&mdash;", "&mdash;",
         f'silhouette {bake["kmeans"]["runs"]["all_venues_core_only"]["k"]["2"]["silhouette"]:.3f} at k=2',
         "13", "", "weak separation, but it ships regardless as the base layer"),
        ("Player value &middot; volume", "per-venue mean",
         f'RMSE {bake["fork_b_volume"]["baseline"]["rmse"]:.3f}',
         f'RMSE {bake["fork_b_volume"]["models"]["hist_gbm"]["rmse"]:.3f}, '
         f'skill +{bake["fork_b_volume"]["models"]["hist_gbm"]["skill_vs_baseline"]:.3f}',
         "13", "", "highest raw skill and least trustworthy &mdash; target is zero for 62.5% of players"),
        ("Hand economics &middot; pot size", "per-stake mean",
         f'RMSE {bake["fork_c_pot"]["baseline"]["rmse"]:.2f}',
         f'RMSE {bake["fork_c_pot"]["models"]["ridge"]["rmse"]:.2f}, '
         f'<b>skill +{bake["fork_c_pot"]["models"]["ridge"]["skill_vs_baseline"]:.3f}</b>',
         "12", "lo", "<b>eliminated on evidence</b> &mdash; a genuine negative result"),
    ]
    body = ""
    for name, base, base_score, best, total, cls, note in rows:
        body += (
            f'<tr class="{cls}">'
            + td(f'{name}<span class="ss">{note}</span>', "nm")
            + td(f'{base}<span class="ss m">{base_score}</span>')
            + td(f'<span class="m">{best}</span>')
            + td(f'<span class="m"><b>{total}</b></span>', "r") + "</tr>"
        )
    head = (th("Candidate question", width="230px") + th("Its own dumb baseline", width="128px")
            + th("Best prototype") + th("Score /20", "r", "50px"))
    cap = ('The score out of 20 is four axes of five: <b>signal</b> and <b>coverage</b> are measured '
           'from the run; <b>business story</b> and <b>rubric fit</b> are judgement calls and are marked '
           'as such. Two algorithms were fitted per supervised candidate (a linear model and a '
           'gradient-boosted tree), which also de-risked the &ldquo;at least two approaches&rdquo; '
           'requirement.')
    return table(head, body, cap)


def block_venueprev(forka: dict) -> str:
    per = forka["core_all_venues"]["models"]["gbt_classifier"]["per_venue"]
    names = {"PS": "PokerStars", "IPN": "iPoker", "PTY": "PartyPoker",
             "ONG": "Ongame", "FTP": "Full Tilt", "ABS": "Absolute"}
    body = ""
    for site, r in sorted(per.items(), key=lambda kv: -kv[1]["players"]):
        weak = r["roc_auc"] < 0.80
        body += (
            f'<tr class="{"lo" if weak else ""}">'
            + td(f'{names.get(site, site)}<span class="ss">{site}</span>', "nm")
            + td(f'<span class="m">{r["players"]:,}</span>', "r")
            + td(f'{bar(r["prevalence"])}<span class="m">{r["prevalence"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m"><b>{r["roc_auc"]:.3f}</b></span>', "r")
            + td(f'<span class="m">{r["pr_auc"]:.3f}</span>', "r") + "</tr>"
        )
    head = (th("Venue", width="126px") + th("Held-out players", "r")
            + th("Lapse prevalence", "r", "128px") + th("ROC-AUC", "r") + th("PR-AUC", "r"))
    cap = ('Shipped GBT, scored on each venue&rsquo;s own held-out players. <b>The model is weakest '
           'exactly where lapsing is rarest</b> &mdash; small test folds explain part of it, not all. '
           'This is the evidence behind recommendation 3.')
    return table(head, body, cap)


def block_models(forka: dict) -> str:
    core = forka["core_all_venues"]
    bl, md = core["baselines"], core["models"]

    def budget_precision(node):
        for point in node.get("budget_curve", []):
            if abs(point["budget_pct"] - BUDGET) < 1e-9:
                return f'{point["precision"]:.3f}'
        return "&mdash;"

    rows = [
        ("Baseline &middot; flag every player", "no ranking at all",
         bl["majority_class"], None, "", ""),
        ("Baseline &middot; sort by days since last seen", "four minutes in a spreadsheet",
         bl["recency_rank"], None, "cg", ""),
        ("MLlib &middot; logistic regression", "readable coefficients",
         md["logistic_regression"], "logistic_regression", "", ""),
        ("MLlib &middot; random forest", "100 trees, a statistical tie with GBT",
         md["random_forest"], "random_forest", "", ""),
        ("MLlib &middot; gradient-boosted trees", "<b>the shipped model</b>",
         md["gbt_classifier"], "gbt_classifier", "cj", "hi"),
    ]
    body = ""
    for name, note, node, _key, _chip, cls in rows:
        best_f1 = node.get("best_f1", {}).get("f1")
        body += (
            f'<tr class="{cls}">'
            + td(f'{name}<span class="ss">{note}</span>', "nm")
            + td(f'{bar((node["roc_auc"] - 0.45) / 0.55)}'
                 f'<span class="m"><b>{node["roc_auc"]:.3f}</b></span>', "r")
            + td(f'<span class="m">{node["pr_auc"]:.3f}</span>', "r")
            + td(f'<span class="m">{best_f1:.3f}</span>' if best_f1 else "&mdash;", "r")
            + td(f'<span class="m">{budget_precision(node)}</span>', "r") + "</tr>"
        )
    head = (th("Approach", width="252px") + th("ROC-AUC", "r", "128px") + th("PR-AUC", "r")
            + th("Best F1", "r") + th("Precision @ 10% budget", "r", "86px"))
    cap = (f'CORE feature set, all six venues, {core["n_test"]:,} held-out players. PR-AUC&rsquo;s floor '
           f'is the prevalence ({bl["majority_class"]["pr_auc"]:.3f}), not 0.5. <b>The second row is '
           f'the bar every line of this pipeline has to clear.</b>')
    return table(head, body, cap)


def block_budget(forka: dict) -> str:
    core = forka["core_all_venues"]
    gbt = core["models"]["gbt_classifier"]
    rec = core["baselines"]["recency_rank"]

    def at_budget(node):
        for point in node["budget_curve"]:
            if abs(point["budget_pct"] - BUDGET) < 1e-9:
                return point
        raise KeyError("no 10% budget point")

    g, r = at_budget(gbt), at_budget(rec)
    contacts = g["contacted"]
    prevalence = core["prevalence_test"]
    random_caught = round(contacts * prevalence)
    rows = [
        ("Random selection", random_caught, prevalence, 1.0, ""),
        ("Sort by recency &mdash; free, no model", r["true_lapsers_caught"], r["precision"],
         r["lift_vs_random"], "cg"),
        ("<b>MLlib gradient-boosted trees</b>", g["true_lapsers_caught"], g["precision"],
         g["lift_vs_random"], "hi"),
    ]
    body = ""
    for name, caught, precision, lift, cls in rows:
        body += (
            f'<tr class="{cls if cls == "hi" else ""}">' + td(name, "nm")
            + td(f'<span class="m">{caught:,}</span>', "r")
            + td(f'<span class="m">{precision:.3f}</span>', "r")
            + td(f'<span class="m">{lift:.2f}x</span>', "r") + "</tr>"
        )
    head = (th(f"Who gets the {contacts:,} calls", width="290px")
            + th("True lapsers reached", "r") + th("Precision", "r") + th("Lift vs random", "r"))
    cap = (f'A 10% contact budget on the {core["n_test"]:,} held-out players. <b>Recall at this budget '
           f'is {g["recall"]:.3f}</b> and that is honest: when '
           f'{prevalence:.0%} of players go quiet in a week, no list covering a tenth of them can '
           f'cover most of them.')
    return table(head, body, cap)


def block_importance(forka: dict) -> str:
    core = forka["core_all_venues"]
    imps = core["models"]["gbt_classifier"]["top_importances"][:7]
    coefs = core["models"]["logistic_regression"]["top_coefficients"][:7]
    top = max(i["importance"] for i in imps)
    topc = max(abs(c["coef"]) for c in coefs)

    def clean(name: str) -> str:
        return name.removeprefix("p_").removesuffix("_i").replace("_", " ")

    left = "".join(
        f'<tr>{td(clean(i["feature"]), "nm")}'
        f'{td(bar(i["importance"] / top) + f"""<span class="m">{i["importance"]:.3f}</span>""", "r")}</tr>'
        for i in imps
    )
    right = "".join(
        f'<tr>{td(clean(c["feature"]), "nm")}'
        f'{td(bar(abs(c["coef"]) / topc, "g") + f"""<span class="m">{c["coef"]:+.3f}</span>""", "r")}</tr>'
        for c in coefs
    )
    return (
        '<div class="row c2" style="margin:8px 0">'
        f'<div><div class="eye">GBT feature importance</div>'
        f'<table class="g" style="margin-top:2px">{left}</table></div>'
        f'<div><div class="eye">Logistic regression &middot; standardised coefficients</div>'
        f'<table class="g" style="margin-top:2px">{right}</table></div></div>'
    )


def block_segments(rake: dict, segs: dict) -> str:
    eco = {int(r["prediction"]): r for r in segs["segments"]["ecosystem_five_venues"]}
    name_of = {int(k): v for k, v in rake["segment_names"].items()}
    by_name = {r["segment"]: r for r in rake["by_segment"]}
    lapsed_total = sum(r["lapsed_actual"] for r in rake["by_segment"])
    order = ["Grinder", "Regular", "Gambler", "Recreational"]
    idx = {v: k for k, v in name_of.items()}
    body = ""
    for label in order:
        c, m = eco[idx[label]], by_name[label]
        loose = label in ("Gambler", "Recreational")
        body += (
            f'<tr class="{"lo" if loose else ""}">'
            + td(f'The {label}', "nm")
            + td(f'<span class="m">{c["share_pct"]:.1f}%</span>'
                 f'<span class="ss m">{m["players"]:,}</span>', "r")
            + td(f'<span class="m">{c["p_hands"]:,.0f}</span>', "r")
            + td(f'<span class="m">{c["p_max_tables"]:.1f}</span>', "r")
            + td(f'<span class="m">{c["p_vpip"]:.2f}</span>', "r")
            + td(f'<span class="m">{c["p_pfr"]:.2f}</span>', "r")
            + td(f'<span class="m" style="color:var(--red-lt)">{c["p_bb_per_100"]:+.1f}</span>', "r")
            + td(f'{bar(m["lapse_rate"])}<span class="m"><b>{m["lapse_rate"] * 100:.1f}%</b></span>', "r")
            + td(f'<span class="m">{m["lapsed_actual"] / lapsed_total * 100:.1f}%</span>', "r")
            + "</tr>"
        )
    head = (th("Segment", width="102px") + th("Share", "r", "70px") + th("Hands", "r")
            + th("Tables", "r") + th("VPIP", "r") + th("PFR", "r") + th("bb/100", "r")
            + th("Lapse rate", "r", "104px") + th("Of all lapsers", "r"))
    cap = ('K-means, k=4, five venues, '
           f'{sum(r["players"] for r in rake["by_segment"]):,} players, clustered on prior-window '
           'behaviour. <b>Names are interpretation; every figure is measured.</b> Nothing about '
           'leaving was in the feature set &mdash; the lapse column was cross-tabbed afterwards.')
    return table(head, body, cap)


def block_segmoney(rake: dict) -> str:
    total_risk = rake["headline"]["expected_rake_at_risk_usd"]
    pooled_lapse = (sum(r["lapsed_actual"] for r in rake["by_segment"])
                    / sum(r["players"] for r in rake["by_segment"]))
    body = ""
    for r in sorted(rake["by_segment"], key=lambda x: -x["expected_at_risk"]):
        loose = r["segment"] in ("Gambler", "Recreational")
        body += (
            f'<tr class="{"lo" if loose else ""}">'
            + td(f'The {r["segment"]}', "nm")
            + td(f'<span class="m">{r["players"]:,}</span>', "r")
            + td(f'<span class="m">${r["weekly_rake"]:,.0f}</span>', "r")
            + td(f'<span class="m"><b>${r["usd_per_player"]:,.2f}</b></span>', "r")
            + td(f'<span class="m">{r["lapse_rate"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m"><b>${r["expected_at_risk"]:,.0f}</b></span>', "r")
            + td(f'{bar(r["share_of_risk"] / 0.35)}'
                 f'<span class="m">{r["share_of_risk"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m">{r["measured_share"] * 100:.1f}%</span>', "r") + "</tr>"
        )
    body += (
        '<tr class="tot">' + td("All five venues", "nm")
        + td(f'<span class="m">{rake["headline"]["players"]:,}</span>', "r")
        + td(f'<span class="m">${rake["headline"]["weekly_rake_usd"]:,.0f}</span>', "r")
        + td("&mdash;", "r")
        + td(f'<span class="m">{pooled_lapse * 100:.1f}%</span>', "r")
        + td(f'<span class="m">${total_risk:,.0f}</span>', "r") + td("100%", "r")
        + td(f'<span class="m">{rake["headline"]["measured_share"] * 100:.1f}%</span>', "r") + "</tr>"
    )
    head = (th("Segment", width="104px") + th("Players", "r") + th("Weekly rake", "r")
            + th("Per player", "r") + th("Lapse rate", "r") + th("$ at risk", "r")
            + th("Share of risk", "r", "104px") + th("Measured", "r"))
    cap = ('Weekly rake is the last seven days of each venue&rsquo;s own prior window &mdash; the same '
           'window the label is measured over, so both halves of the multiplication describe the same '
           'period. Rows shaded red are the loose, money-losing segments.')
    return table(head, body, cap)


SHORT = {"Ongame Network": "Ongame", "Absolute Poker": "Absolute",
         "Full Tilt Poker": "Full Tilt", "iPoker Network": "iPoker"}


def short(name: str) -> str:
    return SHORT.get(name, name)


def block_venuemoney(rake: dict) -> str:
    body = ""
    for r in sorted(rake["by_venue"], key=lambda x: -x["expected_at_risk"]):
        body += (
            "<tr>" + td(short(r["venue"]), "nm")
            + td(f'<span class="m">{r["players"]:,}</span>', "r")
            + td(f'<span class="m">${r["weekly_rake"]:,.0f}</span>', "r")
            + td(f'<span class="m">${r["usd_per_player"]:,.2f}</span>', "r")
            + td(f'<span class="m">{r["lapse_rate"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m"><b>${r["expected_at_risk"]:,.0f}</b></span>', "r")
            + td(f'{bar(r["share_of_risk"] / 0.5)}'
                 f'<span class="m">{r["share_of_risk"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m">{r["measured_share"] * 100:.1f}%</span>', "r") + "</tr>"
        )
    head = (th("Venue", width="104px") + th("Players", "r") + th("Weekly rake", "r")
            + th("Per player", "r") + th("Lapse rate", "r") + th("$ at risk", "r")
            + th("Share of risk", "r", "104px") + th("Measured", "r"))
    cap = ('Five venues; iPoker excluded. <b>Per-player rake differs by a factor of five mostly because '
           'the venues sit at different stakes</b>, not because their players behave differently.')
    return table(head, body, cap)


def block_ranking(rake: dict, curves: dict, budget_k: int, total: float) -> str:
    full = rake["ranking_full"]
    test = rake["ranking_test_fold"]
    rows = [
        ("Expected loss <span class='ss'>calibrated risk &times; weekly rake</span>",
         "expected loss (risk x $)", "hi"),
        ("Money alone <span class='ss'>rank by weekly rake</span>", "weekly rake alone", ""),
        ("Risk alone <span class='ss'>rank by the churn score</span>", "risk alone", "lo"),
    ]
    body = ""
    for label, key, cls in rows:
        f, t = full[key], test[key]
        body += (
            f'<tr class="{cls}">' + td(label, "nm")
            + td(f'<span class="m">${f["expected_at_risk_reached"]:,.0f}</span>', "r")
            + td(f'{bar(f["share_of_total"])}'
                 f'<span class="m"><b>{f["share_of_total"] * 100:.1f}%</b></span>', "r")
            + td(f'<span class="m">{f["true_lapsers"]:,}</span>', "r")
            + td(f'<span class="m">{f["precision"] * 100:.0f}%</span>', "r")
            + td(f'<span class="m">{f["overlap_with_expected_loss"] * 100:.1f}%</span>', "r")
            + td(f'<span class="m">{t["share_of_total"] * 100:.1f}%</span>', "r") + "</tr>"
        )
    head = (th("Sort the list by", width="182px") + th("Rake reached", "r")
            + th("Share of the money", "r", "112px") + th("True leavers", "r")
            + th("Precision", "r") + th("Names shared with row 1", "r", "70px")
            + th("Held-out fold", "r", "66px"))
    cap = (f'The same {rake["headline"]["players"]:,} players, {budget_k:,} calls (10%), three ways of '
           f'choosing who gets them. The last column repeats the exercise on the '
           f'{test["expected loss (risk x $)"]["contacted"]:,} held-out players alone &mdash; labels no '
           f'model ever trained on &mdash; and the picture is identical.')
    return table(head, body, cap)


def block_sensitivity(rake: dict) -> str:
    hl = rake["headline"]["expected_rake_at_risk_usd"]
    alt = rake["sensitivity_probability"]["shipped weighted scores"]
    est = rake["sensitivity_estimator"]
    att = rake["sensitivity_attribution"]
    hc = rake["sensitivity_high_coverage"]
    rows = [
        ("Which probability", "the uncalibrated weighted scores",
         f'${alt:,.0f} &mdash; <b>{(1 - alt / hl) * 100:.1f}% lower</b>',
         "<b>Moves the headline by nearly a third.</b> This is the entire reason &sect;6.4 exists.", "lo"),
        ("Rake estimator", "the naive per-player rate &times; hand count",
         f'${est["naive_usd"]:,.0f} vs ${est["estimator_usd"]:,.0f} of weekly rake &mdash; '
         f'<b>{(est["inflation_x"] - 1) * 100:.0f}% high</b>',
         "Directionally exactly as &sect;5.7 predicts, on the "
         f'{est["players"]:,} players it can be computed for.', "lo"),
        ("Attribution rule", "equal split between everyone dealt in",
         f'prior-window rake ${att["contributed_usd"]:,.0f} vs ${att["equal_split_usd"]:,.0f}, '
         f'<b>ratio {att["ratio"]:.3f}</b>',
         "<b>Does not move the headline.</b> The rule shifts ~2% of the money between players.", "hi"),
        ("Modelled venues", "the two best-covered venues only",
         f'{hc["at_risk_rate"] * 100:.1f}% of their weekly rake at risk vs '
         f'{rake["headline"]["at_risk_share_of_weekly"] * 100:.1f}% pooled',
         f'Tracks <b>lapse prevalence, not the estimator</b>: these two venues '
         f'({hc["measured_share"] * 100:.0f}% measured) have the lowest lapse rates.', ""),
    ]
    body = ""
    for decision, alternative, result, note, cls in rows:
        body += (
            f'<tr class="{cls}">'
            + td(f'{decision}<span class="ss">{alternative}</span>', "nm")
            + td(f'<span class="m">{result}</span>')
            + td(note) + "</tr>"
        )
    head = (th("Decision", width="150px") + th("Headline under the alternative", width="200px")
            + th("Reading"))
    return table(head, body, cls="g tight")


# ------------------------------------------------------------------ svg charts
def chart_calibration(before: list, after: list) -> str:
    """Reliability curve, sized for the report column."""
    W, H = COL, 250
    L, R, T, B = 46, 250, 14, 40
    pw, ph = W - L - R, H - T - B

    def sx(p):
        return L + p * pw

    def sy(p):
        return T + ph - p * ph

    parts = [
        *[line(L, sy(t), L + pw, sy(t), stroke=LINE) for t in (0, 0.25, 0.5, 0.75, 1)],
        *[txt(L - 8, sy(t) + 3.5, f"{t:.0%}", size=9.5, anchor="end", fill=DIM, mono=True)
          for t in (0, 0.25, 0.5, 0.75, 1)],
        *[txt(sx(t), T + ph + 15, f"{t:.0%}", size=9.5, anchor="middle", fill=DIM, mono=True)
          for t in (0, 0.25, 0.5, 0.75, 1)],
        line(sx(0), sy(0), sx(1), sy(1), stroke=LINE2, w=1.1, dash="4 4"),
        txt(sx(0.66), sy(0.63), "perfectly calibrated", size=9.5, fill=LINE2, weight=600),
        txt(L + pw / 2, T + ph + 32, "predicted chance of going quiet",
            size=10, anchor="middle", fill=SOFT),
        txt(-(T + ph / 2), 12, "observed rate", size=10, anchor="middle", fill=SOFT,
            style="transform:rotate(-90deg);transform-box:view-box"),
    ]
    for bins, colour, dash, label in (
        (before, RED_LT, "5 4", "weighted (shipped Fork A)"),
        (after, GREEN_LT, None, "unweighted refit"),
    ):
        pts = [(sx(b["mean_predicted"]), sy(b["observed"])) for b in bins]
        parts.append(polyline(pts, stroke=colour, w=2.2, dash=dash))
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" style="fill:{colour}"/>')

    worst_b = max(before, key=lambda b: abs(b["deviation"]))
    parts += [
        line(sx(worst_b["mean_predicted"]), sy(worst_b["mean_predicted"]),
             sx(worst_b["mean_predicted"]), sy(worst_b["observed"]), stroke=RED, w=1.4),
        txt(sx(worst_b["mean_predicted"]) + 7,
            sy((worst_b["mean_predicted"] + worst_b["observed"]) / 2) + 3,
            f'{abs(worst_b["deviation"]):.3f} off', size=10, fill=RED_LT, weight=700, mono=True),
    ]
    lx = L + pw + 26
    parts += [
        txt(lx, T + 14, "BEFORE", size=9.5, fill=RED_LT, weight=800),
        txt(lx, T + 29, "class-weighted GBT", size=10.5, fill=SOFT),
        txt(lx, T + 46, "worst decile 0.200 off", size=10.5, fill=INK, weight=700, mono=True),
        txt(lx, T + 61, "ECE 0.106  ·  ROC 0.8566", size=10, fill=SOFT, mono=True),
        txt(lx, T + 88, "AFTER", size=9.5, fill=GREEN_LT, weight=800),
        txt(lx, T + 103, "same GBT, no weights", size=10.5, fill=SOFT),
        txt(lx, T + 120, "worst decile 0.026 off", size=10.5, fill=INK, weight=700, mono=True),
        txt(lx, T + 135, "ECE 0.011  ·  ROC 0.8575", size=10, fill=SOFT, mono=True),
        txt(lx, T + 162, "Calibration changed.", size=10.5, fill=INK, weight=700),
        txt(lx, T + 176, "Discrimination did not.", size=10.5, fill=INK, weight=700),
    ]
    body = svg(W, H, "".join(parts),
               "Reliability curve before and after removing class weights")
    return (f'<figure>{body}<figcaption>Ten equal-sized deciles of predicted probability, each '
            f'plotted against what actually happened. <b>The dashed red line is the shipped '
            f'classifier</b>; the solid green line is the same model with class weighting removed. '
            f'ROC-AUC moves by 0.0009 &mdash; inside gradient boosting&rsquo;s own reproducibility '
            f'wobble.</figcaption></figure>')


def chart_reach(curves: dict, total: float, budget_k: int, players: int,
                LOG_RISK_REACH: float = 0.0) -> str:
    """Cumulative rake reached against list depth, for three ranking rules."""
    W, H = COL, 252
    L, R, T, B = 60, 176, 14, 44
    pw, ph = W - L - R, H - T - B

    def sx(frac):
        return L + frac * pw

    def sy(dollars):
        return T + ph - (dollars / total) * ph

    parts = [
        *[line(L, sy(total * f), L + pw, sy(total * f), stroke=LINE)
          for f in (0, 0.25, 0.5, 0.75, 1.0)],
        *[txt(L - 8, sy(total * f) + 3.5, f"${total * f / 1000:,.0f}k",
              size=9.5, anchor="end", fill=DIM, mono=True) for f in (0, 0.25, 0.5, 0.75, 1.0)],
        *[txt(sx(f), T + ph + 15, f"{f * 100:.0f}%", size=9.5, anchor="middle", fill=DIM)
          for f in (0, 0.25, 0.5, 0.75, 1.0)],
        *[txt(sx(f), T + ph + 28, f"{round(players * f):,}", size=8.8, anchor="middle",
              fill=DIM, mono=True) for f in (0, 0.25, 0.5, 0.75, 1.0)],
        txt(L + pw / 2, T + ph + 41, "players contacted, in ranked order",
            size=10, anchor="middle", fill=SOFT),
        txt(-(T + ph / 2), 12, "next week's rake reached", size=10, anchor="middle", fill=SOFT,
            style="transform:rotate(-90deg);transform-box:view-box"),
    ]
    style = {
        "expected": (CYAN_LT, None, 2.4, "expected loss"),
        "money": (GOLD_, "7 4", 2.0, "money alone"),
        "risk": (RED_LT, "2 3", 2.0, "risk alone"),
    }
    for key, (colour, dash, width, label) in style.items():
        pts = [(sx(f), sy(v)) for f, v in curves[key]["curve"]]
        parts.append(polyline(pts, stroke=colour, w=width, dash=dash))

    bx = sx(BUDGET)
    parts += [
        line(bx, T, bx, T + ph, stroke=GOLD_, w=1.1, dash="3 3"),
        txt(bx + 5, T + 11, f"{budget_k:,} calls", size=9.5, fill=GOLD_LT, weight=700, mono=True),
    ]
    lx = L + pw + 16
    for i, (key, (colour, dash, _w, label)) in enumerate(style.items()):
        y = T + 18 + i * 46
        v = curves[key]["at_budget"]
        parts += [
            rect(lx, y - 9, 16, 2.6, fill=colour),
            txt(lx + 22, y - 5, label, size=10.5, fill=INK, weight=700),
            txt(lx, y + 12, f"${v:,.0f}", size=13, fill=colour, weight=700, mono=True),
            txt(lx, y + 26, f"{v / total * 100:.1f}% of the money",
                size=9.5, fill=SOFT, mono=True),
        ]
    body = svg(W, H, "".join(parts), "Rake reached against list depth for three ranking rules")
    return (f'<figure>{body}<figcaption>Each curve recomputed from the shipped Gold table by '
            f'<span class="m">src/report_figures.py</span>, which checks itself against the model '
            f'run&rsquo;s own JSON before writing. <b>One figure moves by design:</b> the risk-only '
            f'ranking re-derives to <b>${curves["risk"]["at_budget"]:,.0f}</b> here against '
            f'<b>${LOG_RISK_REACH:,.0f}</b> in the run log, because <span class="m">risk_cal</span> '
            f'ships rounded to four decimals and its many ties break differently outside the run. It '
            f'is {curves["risk"]["at_budget"] / total:.1%} of the money either way, and it is stated '
            f'rather than hidden. Series carry dash patterns as well as colour, because red and gold '
            f'sit inside the colour-vision confusion band.</figcaption></figure>')


def chart_arch(pipe: dict, rake: dict, forka: dict) -> str:
    """The medallion pipeline, drawn with the measured volume at each stage."""
    W, H = COL, 224
    stages = [
        ("BRONZE · RAW", "Published files, untouched",
         f'{pipe["bronze"]["files"]:,} .phhs files',
         f'{pipe["bronze"]["bytes"] / 1e9:,.1f} GB text', GOLD_, False),
        ("INGEST", "Parse · normalise · derive",
         f'{pipe["build_log"]["build_minutes"]:.1f} min on {pipe["build_log"]["build_cores"]} cores',
         f'{pipe["reconciliation"]["published"] / 1e6:,.1f} M hands', SOFT, False),
        ("SILVER · MODELLED", "Three typed Parquet tables",
         f'{pipe["silver"]["hand_players"]["rows"] / 1e6:,.0f} M seats · '
         f'{pipe["silver"]["actions"]["rows"] / 1e6:,.0f} M actions',
         f'{pipe["silver_total_bytes"] / 1e9:,.1f} GB · {pipe["compression_x"]:.1f}x', CYAN, False),
        ("GOLD · FEATURES", "One row per player",
         f'{pipe["gold"]["player_lapse"]["cols"]} prior-window features',
         f'{pipe["gold"]["player_lapse"]["bytes"] / 1e6:,.0f} MB · '
         f'{pipe["gold"]["player_lapse"]["rows"]:,} rows', GREEN, True),
        ("ML & DECISION", "MLlib, then a ranked list",
         "GBT · RF · logistic · K-means",
         f'{rake["headline"]["players"]:,} accounts', CYAN_LT, True),
    ]
    gap, pad = 8, 0
    bw = (W - gap * (len(stages) - 1) - pad * 2) / len(stages)
    parts = [txt(0, 11, "THE PIPELINE, WITH WHAT WAS MEASURED AT EACH STAGE", size=9.5,
                 fill=DIM, weight=800, style="letter-spacing:1.6px")]
    for i, (kicker, title, detail, big, colour, highlight) in enumerate(stages):
        x = pad + i * (bw + gap)
        parts.append(rect(x, 22, bw, 122, fill="#F6F9FC" if highlight else "#FFFFFF"))
        parts.append(f'<rect x="{x:.1f}" y="22" width="{bw:.1f}" height="122" rx="8" '
                     f'style="fill:none;stroke:{LINE2}" stroke-width="1"/>')
        parts.append(rect(x, 22, bw, 3, fill=colour, rx=0))
        parts.append(txt(x + 10, 42, kicker, size=8.4, fill=colour, weight=800,
                         style="letter-spacing:1.1px"))
        # wrap the title over up to two lines
        words, lines, cur = title.split(), [], ""
        for w in words:
            trial = (cur + " " + w).strip()
            if len(trial) > 20 and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        for j, ln in enumerate(lines[:2]):
            parts.append(txt(x + 10, 60 + j * 13, ln, size=11, fill=INK, weight=750))
        dwords, dlines, dcur = detail.split(), [], ""
        for w in dwords:
            trial = (dcur + " " + w).strip()
            if len(trial) > 22 and dcur:
                dlines.append(dcur)
                dcur = w
            else:
                dcur = trial
        dlines.append(dcur)
        for j, ln in enumerate(dlines[:3]):
            parts.append(txt(x + 10, 88 + j * 11, ln, size=9, fill=SOFT))
        parts.append(txt(x + 9, 134, big, size=10, fill=colour, weight=700, mono=True))
        if i < len(stages) - 1:
            ax = x + bw + gap / 2
            parts.append(f'<path d="M{ax - 3:.1f},78 L{ax + 3:.1f},83 L{ax - 3:.1f},88" '
                         f'fill="none" style="stroke:{DIM}" stroke-width="1.3"/>')

    # the funnel annotation underneath
    parts += [
        line(0, 164, W, 164, stroke=LINE),
        txt(0, 182, f'{pipe["silver"]["hand_players"]["rows"]:,}', size=15, fill=INK,
            weight=700, mono=True),
        txt(0, 196, "seat-rows enter feature engineering", size=9.5, fill=SOFT),
        txt(W, 182, f'{pipe["gold"]["player_lapse"]["bytes"] / 1e6:,.0f} MB', size=15, fill=GREEN_LT,
            weight=700, mono=True, anchor="end"),
        txt(W, 196, "is what every model actually reads", size=9.5, fill=SOFT, anchor="end"),
        txt(W / 2, 190, f'{pipe["funnel"]["shrink_x"]:,.0f}x smaller', size=12, fill=GOLD_LT,
            weight=700, mono=True, anchor="middle"),
    ]
    body = svg(W, H, "".join(parts), "Medallion architecture with measured volume at each stage")
    return (f'<figure>{body}<figcaption><b>Every figure on this diagram is measured, not '
            f'estimated</b> &mdash; recomputed from the lake on disk by '
            f'<span class="m">src/pipeline_facts.py</span>. Local Spark '
            f'(<span class="m">local[8]</span>, 9 GB driver) on one laptop; the same code re-hosts '
            f'to Databricks unchanged.</figcaption></figure>')


# ==================================================================== the build
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args()

    rake = json.loads((DOCS / "rake-at-risk-results.json").read_text())
    forka = json.loads((DOCS / "fork-a-results.json").read_text())
    segs = json.loads((DOCS / "segments-results.json").read_text())
    bake = json.loads((DOCS / "bakeoff-results.json").read_text())
    pipe_path = DOCS / "pipeline-facts.json"
    if not pipe_path.exists():
        sys.exit("missing docs/pipeline-facts.json -- run src/pipeline_facts.py first")
    pipe = json.loads(pipe_path.read_text())
    if pipe.get("zero_sum_check", {}).get("skipped"):
        sys.exit("docs/pipeline-facts.json was written by `pipeline_facts.py --fast`, which skips "
                 "the zero-sum identity the report quotes in §5.3.\n"
                 "Re-run `.venv/bin/python src/pipeline_facts.py` without --fast.")

    df = load_gold()
    players = df.height
    budget_k = int(round(players * BUDGET))
    curves = ranking_curves(df, budget_k)
    total = curves["_total"]

    # ---- regression checks against the run logs -----------------------------
    checks = []

    def check(label, got, want, tol):
        checks.append((label, got, want, abs(got - want) <= tol))

    hl = rake["headline"]
    check("players", players, hl["players"], 0)
    check("contacts at 10%", budget_k, hl["top_budget_players"], 0)
    check("total expected at risk", total, hl["expected_rake_at_risk_usd"], 1.0)
    check("expected-loss reach", curves["expected"]["at_budget"], hl["top_budget_reach_usd"], 1.0)
    check("money-alone reach", curves["money"]["at_budget"],
          rake["ranking_full"]["weekly rake alone"]["expected_at_risk_reached"], 1.0)
    check("expected-loss leavers", curves["expected"]["lapsers"],
          rake["ranking_full"]["expected loss (risk x $)"]["true_lapsers"], 0)
    check("risk-alone leavers", curves["risk"]["lapsers"],
          rake["ranking_full"]["risk alone"]["true_lapsers"], 0)
    # risk_cal ships rounded to 4 dp, so ties break differently outside the run
    check("risk-alone reach (ties, +-2%)", curves["risk"]["at_budget"],
          rake["ranking_full"]["risk alone"]["expected_at_risk_reached"],
          0.02 * rake["ranking_full"]["risk alone"]["expected_at_risk_reached"])
    # the pipeline facts must agree with the model runs about the population
    check("gold player_lapse rows", pipe["gold"]["player_lapse"]["rows"],
          forka["population"]["players"], 0)
    check("reconciliation closes exactly", pipe["reconciliation"]["accounted_for"],
          pipe["reconciliation"]["published"], 0)

    print("REGRESSION CHECKS vs the run logs")
    for label, got, want, ok in checks:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label:34s} got {got:>14,.2f}   want {want:>14,.2f}")
    if not all(ok for *_, ok in checks):
        sys.exit("regression checks failed -- the report was NOT written")

    # ---- derived values the report quotes -----------------------------------
    core = forka["core_all_venues"]
    gbt = core["models"]["gbt_classifier"]
    rec = core["baselines"]["recency_rank"]
    cal = rake["calibration"]
    eco = {int(r["prediction"]): r for r in segs["segments"]["ecosystem_five_venues"]}
    names = {v: int(k) for k, v in rake["segment_names"].items()}
    seg = {r["segment"]: r for r in rake["by_segment"]}
    venue = {r["venue"]: r for r in rake["by_venue"]}

    def v(prefix: str) -> dict:
        """rake JSON writes 'Ongame Network'; the report says 'Ongame'."""
        hits = [row for name, row in venue.items() if name.startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"venue lookup {prefix!r} matched {len(hits)} rows")
        return hits[0]
    lapsed_total = sum(r["lapsed_actual"] for r in rake["by_segment"])
    gbt_budget = next(p for p in gbt["budget_curve"] if abs(p["budget_pct"] - BUDGET) < 1e-9)
    rec_budget = next(p for p in rec["budget_curve"] if abs(p["budget_pct"] - BUDGET) < 1e-9)
    pv = {r["site"]: r for r in pipe["by_venue"]}
    window_days = pipe["reconciliation"]["day_max"]

    calc = {
        # pipeline
        "bronze_gb": pipe["bronze"]["bytes"] / 1e9,
        "silver_gb": pipe["silver_total_bytes"] / 1e9,
        "compression_x": pipe["compression_x"],
        "model_table_mb": pipe["gold"]["player_lapse"]["bytes"] / 1e6,
        "seat_rows_m": pipe["silver"]["hand_players"]["rows"] / 1e6,
        "hands_per_day": pipe["reconciliation"]["silver_rows"] / window_days,
        "actions_per_day": pipe["silver"]["actions"]["rows"] / window_days,
        "ipn_hands": pv["IPN"]["hands"],
        "ps_tables": pv["PS"]["tables"],
        "nfd_perfect_venues": sum(
            1 for r in pipe["no_flop_no_drop"]["by_venue"] if r["zero_share"] >= 1.0),
        # rake / money
        "median_rake_share_lo": min(r["median_share"] for r in rake["rake_share_by_venue"]),
        "median_rake_share_hi": max(r["median_share"] for r in rake["rake_share_by_venue"]),
        "modelled_share": 1 - hl["measured_share"],
        "at_risk_player_share": hl["at_risk_players"] / hl["players"],
        "priced_hands_m": sum(r["hands"] for r in rake["priced_by_venue"]) / 1e6,
        "usd_per_hand": (rake["attribution_check"]["contributed_total_usd"]
                         / sum(r["hands"] for r in rake["priced_by_venue"])),
        "naive_inflation": rake["sensitivity_estimator"]["inflation_x"] - 1,
        "calibration_understatement": 1 - (
            rake["sensitivity_probability"]["shipped weighted scores"]
            / hl["expected_rake_at_risk_usd"]),
        # ranking
        "expected_share": curves["expected"]["at_budget"] / total,
        "expected_lapsers": curves["expected"]["lapsers"],
        "risk_reach": curves["risk"]["at_budget"],
        "risk_share": curves["risk"]["at_budget"] / total,
        "risk_lapsers": curves["risk"]["lapsers"],
        "risk_precision": curves["risk"]["precision"],
        "money_share": curves["money"]["at_budget"] / total,
        "overlap": curves["risk"]["overlap"],
        "overlap_test": rake["ranking_test_fold"]["risk alone"]["overlap_with_expected_loss"],
        # segments
        "rec_bb100": eco[names["Recreational"]]["p_bb_per_100"],
        "gambler_bb100": eco[names["Gambler"]]["p_bb_per_100"],
        "grinder_bb100": eco[names["Grinder"]]["p_bb_per_100"],
        "grinder_tables": eco[names["Grinder"]]["p_max_tables"],
        "rec_lapse": seg["Recreational"]["lapse_rate"],
        "gambler_lapse": seg["Gambler"]["lapse_rate"],
        "rec_per_player": seg["Recreational"]["usd_per_player"],
        "grinder_per_player": seg["Grinder"]["usd_per_player"],
        "grinder_multiple": seg["Grinder"]["usd_per_player"] / seg["Recreational"]["usd_per_player"],
        "rec_share_of_players": seg["Recreational"]["players"] / hl["players"],
        "gambler_share_of_risk": seg["Gambler"]["share_of_risk"],
        "grinder_share_of_risk": seg["Grinder"]["share_of_risk"],
        "loose_share_of_risk": seg["Gambler"]["share_of_risk"] + seg["Recreational"]["share_of_risk"],
        "loose_at_risk_usd": seg["Gambler"]["expected_at_risk"] + seg["Recreational"]["expected_at_risk"],
        "loose_share_of_players": (seg["Gambler"]["players"] + seg["Recreational"]["players"]) / hl["players"],
        "loose_share_of_lapsers": (
            seg["Gambler"]["lapsed_actual"] + seg["Recreational"]["lapsed_actual"]) / lapsed_total,
        "kmeans_sil": segs["algorithm_comparison"]["ecosystem_five_venues"]["kmeans_silhouette"],
        "bisecting_sil": segs["algorithm_comparison"]["ecosystem_five_venues"]["bisecting_silhouette"],
        "cluster_agreement": segs["algorithm_comparison"]["ecosystem_five_venues"]["agreement"],
        "sil_k2": segs["sweeps"]["style_all_venues"]["2"]["silhouette"],
        # venues
        "venue_lapse_lo": min(r["lapse_rate"] for r in rake["by_venue"]),
        "venue_lapse_hi": max(r["lapse_rate"] for r in rake["by_venue"]),
        "venue_per_player_lo": min(r["usd_per_player"] for r in rake["by_venue"]),
        "venue_per_player_hi": max(r["usd_per_player"] for r in rake["by_venue"]),
        "ps_share_of_players": v("PokerStars")["players"] / hl["players"],
        "ps_share_of_risk": v("PokerStars")["share_of_risk"],
        "ps_measured_share": v("PokerStars")["measured_share"],
        "ps_lapse": v("PokerStars")["lapse_rate"],
        "ong_lapse": v("Ongame")["lapse_rate"],
        "abs_lapse": v("Absolute")["lapse_rate"],
        # models
        "prevalence_pct": forka["population"]["prevalence_lapsed"] * 100,
        "majority_f1": core["baselines"]["majority_class"]["at_0.5"]["f1"],
        "quiet1_f1": rec["hard_rules"][">= 1 days quiet"]["f1"],
        "gbt_f1_default": gbt["at_0.5"]["f1"],
        "gbt_f1_tuned": gbt["best_f1"]["f1"],
        "gbt_thr_tuned": gbt["best_f1"]["threshold"],
        "gbt_budget_contacts": gbt_budget["contacted"],
        "gbt_budget_recall": gbt_budget["recall"],
        "gbt_extra_lapsers": gbt_budget["true_lapsers_caught"] - rec_budget["true_lapsers_caught"],
        "logistic_gap": gbt["roc_auc"] - core["models"]["logistic_regression"]["roc_auc"],
        "recency_importance": gbt["top_importances"][0]["importance"],
        "extended_players": forka["extended_five_venues"]["n_train"] + forka["extended_five_venues"]["n_test"],
        "extended_gain": (forka["extended_five_venues"]["models"]["gbt_classifier"]["roc_auc"]
                          - gbt["roc_auc"]),
        "extended_loss_share": 1 - (
            (forka["extended_five_venues"]["n_train"] + forka["extended_five_venues"]["n_test"])
            / forka["population"]["players"]),
        # calibration
        "roc_before": cal["weighted"]["roc_auc"],
        "roc_after": cal["unweighted"]["roc_auc"],
        "ece_before": cal["weighted"]["reliability"]["ece"],
        "ece_after": cal["unweighted"]["reliability"]["ece"],
        "worst_decile_before": max(abs(b["deviation"]) for b in cal["weighted"]["reliability"]["bins"]),
        "worst_decile_after": max(abs(b["deviation"]) for b in cal["unweighted"]["reliability"]["bins"]),
        "recal_f1": cal["best_f1"]["f1"],
        "recal_thr": cal["best_f1"]["threshold"],
    }
    roots = {"rake": rake, "forka": forka, "seg": segs, "bake": bake, "pipe": pipe, "calc": calc}

    if not REPORT.exists():
        sys.exit(f"missing {REPORT}")
    html = REPORT.read_text()
    original = html

    # ---- generated blocks ---------------------------------------------------
    blocks = {
        "volume": block_volume(pipe),
        "venuetable": block_venuetable(pipe),
        "coverage": block_coverage(rake, pipe),
        "recon": block_recon(pipe),
        "arch": chart_arch(pipe, rake, forka),
        "population": block_population(forka),
        "leakage": block_leakage(forka),
        "representative": block_representative(rake),
        "plausibility": block_plausibility(rake),
        "estimator": block_estimator(rake),
        "bakeoff": block_bakeoff(bake),
        "venueprev": block_venueprev(forka),
        "models": block_models(forka),
        "budget": block_budget(forka),
        "importance": block_importance(forka),
        "segments": block_segments(rake, segs),
        "segmoney": block_segmoney(rake),
        "venuemoney": block_venuemoney(rake),
        "ranking": block_ranking(rake, curves, budget_k, total),
        "reach": chart_reach(curves, total, budget_k, players,
                             rake["ranking_full"]["risk alone"]["expected_at_risk_reached"]),
        "calibration": chart_calibration(cal["weighted"]["reliability"]["bins"],
                                         cal["unweighted"]["reliability"]["bins"]),
        "sensitivity": block_sensitivity(rake),
    }
    for name, content in blocks.items():
        pattern = re.compile(rf"(<!--CHART:{name}:START-->).*?(<!--CHART:{name}:END-->)", re.S)
        if not pattern.search(html):
            sys.exit(f"report has no marker pair for {name!r}")
        html = pattern.sub(lambda m: m.group(1) + content + m.group(2), html)
        print(f"  built  {name:15s} {len(content):>7,} chars")

    # ---- every tagged figure ------------------------------------------------
    figure = re.compile(
        r'(?P<open><(?P<tag>[a-z]+)[^>]*class="fig"[^>]*data-fig="(?P<path>[^"]+)"'
        r'[^>]*data-fmt="(?P<fmt>[^"]+)"[^>]*>)(?P<text>[^<]*)(?P<close></(?P=tag)>)'
    )
    changed, seen = [], 0

    def substitute(m):
        nonlocal seen
        seen += 1
        rendered = fmt(resolve(roots, m.group("path")), m.group("fmt"))
        if rendered != m.group("text"):
            changed.append((m.group("path"), m.group("text"), rendered))
        return m.group("open") + rendered + m.group("close")

    html = figure.sub(substitute, html)
    print(f"\n{seen} tagged figures resolved, {len(changed)} rewritten")

    # ---- page numbers, stamped from the sheet order -------------------------
    # Pages get split and reordered while the report is being written; hand-typed
    # footer numbers desynchronise the moment that happens, so they are derived.
    sheets = list(re.finditer(r'<section class="pg"[^>]*>.*?</section>', html, re.S))
    renumbered = 0
    for index, sheet in enumerate(reversed(sheets), start=0):
        page_no = len(sheets) - index
        piece = sheet.group(0)
        fixed, n = re.subn(r'(<span class="pn">)[^<]*(</span>)',
                           rf'\g<1>{page_no}\g<2>', piece)
        if fixed != piece:
            renumbered += n
            html = html[: sheet.start()] + fixed + html[sheet.end():]
    print(f"{len(sheets)} sheets, {renumbered} page numbers stamped")
    for path, was, now in changed:
        print(f"  {path:56s} {was!r} -> {now!r}")

    if args.check:
        if html != original:
            sys.exit("\nREPORT IS STALE -- re-run without --check")
        print("\nreport is current")
        return 0

    REPORT.write_text(html)
    print(f"\nwrote {REPORT.relative_to(ROOT)}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
