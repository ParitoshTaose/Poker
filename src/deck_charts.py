#!/usr/bin/env python3
"""
deck_charts.py — build the executive deck's numbers and charts from measured data.

The deck (deliverables/presentation/ecosystem-engine-deck.html) contains no hand-typed
figures. Every number sits in a span tagged with the JSON path it came from:

    <b class="fig" data-fig="rake.headline.players" data-fmt="int">77,268</b>

and every chart sits between markers:

    <!--CHART:reach:START--> ... <!--CHART:reach:END-->

This script re-derives all of them from
  * docs/rake-at-risk-results.json   (rake_at_risk.py)
  * docs/fork-a-results.json         (fork_a.py)
  * docs/segments-results.json       (segments.py)
  * data/gold/rake_at_risk           (the shipped Gold table, for the ranking curves)
and rewrites the deck in place, printing every value it changed. A clean run that
changes nothing is the proof that the deck matches the analysis.

Run:  .venv/bin/python src/deck_charts.py            # inject
      .venv/bin/python src/deck_charts.py --check    # verify only, exit 1 on drift

No Spark. Polars only. ~2 seconds.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "deliverables" / "presentation" / "ecosystem-engine-deck.html"
GOLD = ROOT / "data" / "gold" / "rake_at_risk"

# ---- house palette (docs/poker-project-plan.html, docs/dashboard.html) ---------
INK, BODY, SOFT, DIM = "#14263C", "#2B3D52", "#546A82", "#7E92A8"
LINE, LINE2, PANEL2 = "#D5E0ED", "#C6D4E4", "#E9F0F8"
CYAN, CYAN_LT = "#2E8BC0", "#176390"
GOLD_, GOLD_LT = "#C08A2E", "#8A5E17"
RED, RED_LT = "#DB4A40", "#C0392E"
GREEN, GREEN_LT = "#2AA277", "#1C8560"
SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
MONO = "'SF Mono',ui-monospace,Menlo,Consolas,monospace"

BUDGET = 0.10  # the contact budget the whole deck is quoted at


# ------------------------------------------------------------------ formatting
def fmt(value, how: str) -> str:
    if how == "int":
        return f"{round(value):,}"
    if how == "usd0":
        return f"${round(value):,}"
    if how == "usd2":
        return f"${value:,.2f}"
    if how == "usdk":
        return f"${round(value / 1000):,}k"
    if how == "pct0":
        return f"{value * 100:.0f}%"
    if how == "pct1":
        return f"{value * 100:.1f}%"
    if how == "pct2":
        return f"{value * 100:.2f}%"
    if how == "num1":
        return f"{value:,.1f}"
    if how == "num2":
        return f"{value:,.2f}"
    if how == "num3":
        return f"{value:,.3f}"
    if how == "x2":
        return f"{value:.2f}x"
    if how == "signed1":
        return f"{value:+,.1f}"
    raise ValueError(f"unknown format {how!r}")


# ------------------------------------------------------------- path resolution
STEP = re.compile(r"([^.\[\]]+)|\[([^\]]+)\]")


def resolve(root: dict, path: str):
    """rake.by_segment[Recreational].share_of_risk  ->  the float."""
    node = root
    for name, key in STEP.findall(path):
        token = name or key
        if isinstance(node, dict):
            if token not in node:
                raise KeyError(f"{path}: no key {token!r} (have {list(node)[:8]})")
            node = node[token]
        elif isinstance(node, list):
            if token.lstrip("-").isdigit():
                node = node[int(token)]
            else:
                hits = [
                    row
                    for row in node
                    if isinstance(row, dict)
                    and any(row.get(f) == token for f in ("segment", "venue", "site", "name"))
                ]
                if len(hits) != 1:
                    raise KeyError(f"{path}: {len(hits)} rows match {token!r}")
                node = hits[0]
        else:
            raise KeyError(f"{path}: cannot index {type(node).__name__} with {token!r}")
    return node


# --------------------------------------------------------------- svg utilities
def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def txt(x, y, s, *, size=12, fill=SOFT, anchor="start", weight=400, mono=False, style=""):
    family = MONO if mono else SANS
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" style="fill:{fill};{style}">{esc(str(s))}</text>'
    )


def line(x1, y1, x2, y2, *, stroke=LINE, w=1, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'style="stroke:{stroke}" stroke-width="{w}"{d}/>'
    )


def rect(x, y, w, h, *, fill, rx=2, opacity=1.0):
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" '
        f'rx="{rx}" style="fill:{fill}" opacity="{opacity}"/>'
    )


def polyline(points, *, stroke, w=2.4, dash=None):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" style="stroke:{stroke}" stroke-width="{w}" '
        f'stroke-linejoin="round" stroke-linecap="round"{d}/>'
    )


def svg(w, h, body, label):
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="{esc(label)}" '
        f'style="display:block;overflow:visible">{body}</svg>'
    )


# ------------------------------------------------------------------ the charts
def chart_reach(curves, headline_total, budget_k, players):
    """Three ranking rules, cumulative rake reached against list depth."""
    W, H = 1140, 326
    L, R, T, B = 78, 300, 20, 52
    pw, ph = W - L - R, H - T - B

    def sx(frac):
        return L + frac * pw

    def sy(dollars):
        return T + ph - (dollars / headline_total) * ph

    parts = [
        # y grid
        *[
            line(L, sy(headline_total * f), L + pw, sy(headline_total * f), stroke=LINE)
            for f in (0, 0.25, 0.5, 0.75, 1.0)
        ],
        *[
            txt(L - 10, sy(headline_total * f) + 4, fmt(headline_total * f, "usdk"),
                size=11.5, anchor="end", fill=DIM, mono=True)
            for f in (0, 0.25, 0.5, 0.75, 1.0)
        ],
        # x axis
        *[
            txt(sx(f), T + ph + 20, f"{f * 100:.0f}%", size=11.5, anchor="middle", fill=DIM)
            for f in (0, 0.25, 0.5, 0.75, 1.0)
        ],
        *[
            txt(sx(f), T + ph + 36, fmt(players * f, "int"), size=10.5, anchor="middle",
                fill=DIM, mono=True)
            for f in (0, 0.25, 0.5, 0.75, 1.0)
        ],
        txt(L + pw / 2, T + ph + 52, "players contacted, in ranked order",
            size=11.5, anchor="middle", fill=SOFT),
        txt(-(T + ph / 2), 16, "next week's rake reached", size=11.5, anchor="middle",
            fill=SOFT, style="transform:rotate(-90deg);transform-box:view-box"),
    ]

    style = {
        "expected": (CYAN, None, 2.8),
        "money": (GOLD_, "2 4", 2.4),
        "risk": (RED, "9 5", 2.4),
    }
    for key in ("money", "risk", "expected"):
        colour, dash, width = style[key]
        pts = [(sx(f), sy(v)) for f, v in curves[key]["curve"]]
        parts.append(polyline(pts, stroke=colour, w=width, dash=dash))

    # the budget line and its three landing points
    bx = sx(BUDGET)
    parts.append(line(bx, T - 4, bx, T + ph, stroke=SOFT, w=1, dash="3 4"))
    parts.append(txt(bx + 7, T + 8, f"contact budget · {BUDGET:.0%}", size=11, fill=SOFT, weight=600))

    labels = [
        ("expected", CYAN, CYAN_LT, "Expected loss", "risk x rake"),
        ("money", GOLD_, GOLD_LT, "Money alone", "weekly rake"),
        ("risk", RED, RED_LT, "Risk alone", "the churn model's score"),
    ]
    for key, colour, dark, name, note in labels:
        parts.append(
            f'<circle cx="{bx:.1f}" cy="{sy(curves[key]["at_budget"]):.1f}" r="4.5" '
            f'style="fill:{dark}"/>'
        )

    # the label block doubles as the legend: each row carries the series' own dash
    # pattern, so the three lines are identifiable without relying on colour alone.
    rows = sorted(labels, key=lambda r: -curves[r[0]]["at_budget"])
    lx = L + pw + 22
    for i, (key, colour, dark, name, note) in enumerate(rows):
        value = curves[key]["at_budget"]
        ly = T + 16 + i * 80
        _, dash, width = style[key]
        parts.append(polyline([(lx, ly - 5), (lx + 30, ly - 5)], stroke=colour, w=width, dash=dash))
        parts.append(txt(lx + 38, ly, name, size=14, fill=dark, weight=700))
        parts.append(txt(lx, ly + 16, note, size=11.5, fill=DIM))
        parts.append(txt(lx, ly + 40, fmt(value, "usd0"), size=22, fill=INK,
                         weight=700, mono=True))
        parts.append(txt(lx, ly + 57, f"{value / headline_total:.1%} of the money at risk",
                         size=11.5, fill=SOFT))

    return svg(W, H, "".join(parts),
               "Rake reached against contact-list depth, for three ways of sorting the list")


def chart_baselines(rows):
    """Horizontal ROC-AUC bars: what the model has to beat, and by how little."""
    W = 980
    row_h, top = 36, 26
    H = top + row_h * len(rows) + 22
    L, R = 268, 96
    pw = W - L - R
    lo = 0.45

    def bw(v):
        return (v - lo) / (1 - lo) * pw

    parts = [
        *[
            line(L + bw(t), top - 12, L + bw(t), top + row_h * len(rows) - 8, stroke=LINE)
            for t in (0.5, 0.6, 0.7, 0.8, 0.9)
        ],
        *[
            txt(L + bw(t), top - 18, f"{t:.1f}", size=10.5, anchor="middle", fill=LINE2, mono=True)
            for t in (0.5, 0.6, 0.7, 0.8, 0.9)
        ],
    ]
    for i, (name, note, value, colour, weight) in enumerate(rows):
        y = top + i * row_h
        parts.append(rect(L, y, bw(value), 21, fill=colour, opacity=0.92))
        parts.append(txt(L - 12, y + 12, name, size=13.5, anchor="end", fill=INK, weight=weight))
        parts.append(txt(L - 12, y + 27, note, size=10.8, anchor="end", fill=DIM))
        parts.append(txt(L + bw(value) + 10, y + 15, fmt(value, "num3"), size=14,
                         fill=colour, weight=700, mono=True))
    parts.append(txt(L, top + row_h * len(rows) + 12, "ROC-AUC on 22,596 held-out players",
                     size=11, fill=SOFT))
    return svg(W, H, "".join(parts), "ROC-AUC of two baselines against three MLlib models")


def chart_calibration(before, after):
    """Reliability curve: predicted probability against what actually happened."""
    W, H = 520, 372
    L, R, T, B = 56, 22, 22, 52
    pw, ph = W - L - R, H - T - B

    def sx(p):
        return L + p * pw

    def sy(p):
        return T + ph - p * ph

    parts = [
        *[line(L, sy(t), L + pw, sy(t), stroke=LINE) for t in (0, 0.25, 0.5, 0.75, 1)],
        *[txt(L - 9, sy(t) + 4, f"{t:.0%}", size=10.5, anchor="end", fill=DIM, mono=True)
          for t in (0, 0.25, 0.5, 0.75, 1)],
        *[txt(sx(t), T + ph + 18, f"{t:.0%}", size=10.5, anchor="middle", fill=DIM, mono=True)
          for t in (0, 0.25, 0.5, 0.75, 1)],
        line(sx(0), sy(0), sx(1), sy(1), stroke=LINE2, w=1.2, dash="4 4"),
        txt(sx(0.62), sy(0.60), "perfect", size=10.5, fill=LINE2, weight=600),
        txt(L + pw / 2, T + ph + 38, "predicted chance of going quiet",
            size=11, anchor="middle", fill=SOFT),
        txt(-(T + ph / 2), 14, "share who actually did", size=11, anchor="middle", fill=SOFT,
            style="transform:rotate(-90deg);transform-box:view-box"),
    ]
    for bins, colour, dash, name in (
        (before, RED, "7 4", "with class weights"),
        (after, GREEN, None, "weights removed"),
    ):
        pts = [(sx(b["mean_predicted"]), sy(b["observed"])) for b in bins]
        parts.append(polyline(pts, stroke=colour, w=2.4, dash=dash))
        for x, y in pts:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" style="fill:{colour}"/>')

    worst = max(before, key=lambda b: abs(b["deviation"]))
    wx, wy0, wy1 = sx(worst["mean_predicted"]), sy(worst["mean_predicted"]), sy(worst["observed"])
    parts.append(line(wx, wy0, wx, wy1, stroke=RED_LT, w=1.4))
    parts.append(txt(wx + 8, (wy0 + wy1) / 2 + 4,
                     f"{abs(worst['deviation']):.3f} off", size=11.5, fill=RED_LT, weight=700))

    parts.append(rect(L + 6, T + 6, 11, 11, fill=RED, rx=2))
    parts.append(txt(L + 23, T + 16, "before — the shipped weighted model", size=11, fill=RED_LT, weight=600))
    parts.append(rect(L + 6, T + 26, 11, 11, fill=GREEN, rx=2))
    parts.append(txt(L + 23, T + 36, "after — same model, weights removed", size=11, fill=GREEN_LT, weight=600))
    return svg(W, H, "".join(parts),
               "Reliability curve before and after removing class weights")


# ------------------------------------------------------------------ the tables
def table_segments(rake_json, seg_json):
    """Segment table with inline bars — who leaves, and whose money leaves with them."""
    by_seg = {r["segment"]: r for r in rake_json["by_segment"]}
    names = rake_json["segment_names"]  # cluster id -> name
    eco = {int(r["prediction"]): r for r in seg_json["segments"]["ecosystem_five_venues"]}
    order = ["Recreational", "Gambler", "Regular", "Grinder"]

    max_risk = max(r["share_of_risk"] for r in by_seg.values())
    head = (
        "<thead><tr>"
        "<th>Segment</th><th class='r'>Share of players</th><th class='r'>Hands</th>"
        "<th class='r'>Tables</th><th class='r'>bb/100</th>"
        "<th class='r'>Lapse rate</th><th class='r'>Rake at risk</th></tr></thead>"
    )
    body = []
    for name in order:
        seg = by_seg[name]
        cid = next(int(k) for k, v in names.items() if v == name)
        eco_row = eco[cid]
        share = seg["players"] / sum(r["players"] for r in by_seg.values())
        tone = "loose" if name in ("Recreational", "Gambler") else "tight"
        body.append(
            f"<tr class='{tone}'>"
            f"<td class='nm'>{name}</td>"
            f"<td class='r m'>{share:.1%}<span class='ss'>{seg['players']:,}</span></td>"
            f"<td class='r m'>{eco_row['p_hands']:,.0f}</td>"
            f"<td class='r m'>{eco_row['p_max_tables']:.1f}</td>"
            f"<td class='r m neg'>{eco_row['p_bb_per_100']:+.1f}</td>"
            f"<td class='r'><span class='bar'><i style='width:{seg['lapse_rate'] * 100:.1f}%'></i>"
            f"</span><b class='m'>{seg['lapse_rate']:.1%}</b></td>"
            f"<td class='r'><span class='bar g'>"
            f"<i style='width:{seg['share_of_risk'] / max_risk * 100:.1f}%'></i></span>"
            f"<b class='m'>{fmt(seg['expected_at_risk'], 'usd0')}</b>"
            f"<span class='ss'>{seg['share_of_risk']:.1%} · {fmt(seg['usd_per_player'], 'usd2')}/head</span></td>"
            f"</tr>"
        )
    return f"<table class='seg'>{head}<tbody>{''.join(body)}</tbody></table>"


def table_ranking(curves, headline_total, budget_k):
    rows = [
        ("expected", "Expected loss", "risk x weekly rake", CYAN_LT, True),
        ("money", "Money alone", "rank by weekly rake", GOLD_LT, False),
        ("risk", "Risk alone", "rank by the churn score", RED_LT, False),
    ]
    head = (
        "<thead><tr><th>Sort the list by</th><th class='r'>Rake reached</th>"
        "<th class='r'>Share of the money</th><th class='r'>True leavers found</th>"
        "<th class='r'>Precision</th><th class='r'>Names in common</th></tr></thead>"
    )
    body = []
    for key, name, note, colour, best in rows:
        c = curves[key]
        body.append(
            f"<tr class='{'win' if best else ''}'>"
            f"<td class='nm' style='color:{colour}'>{name}<span class='ss'>{note}</span></td>"
            f"<td class='r m big'>{fmt(c['at_budget'], 'usd0')}</td>"
            f"<td class='r m'>{c['at_budget'] / headline_total:.1%}</td>"
            f"<td class='r m'>{c['lapsers']:,}</td>"
            f"<td class='r m'>{c['precision']:.0%}</td>"
            f"<td class='r m'>{c['overlap']:.1%}</td>"
            f"</tr>"
        )
    return (
        f"<table class='rank'>{head}<tbody>{''.join(body)}</tbody></table>"
        f"<p class='cap'>Same {budget_k:,} calls (10% of 77,268 players), three ways of choosing who gets them.</p>"
    )


# -------------------------------------------------------------------- the data
def ranking_curves(df: pl.DataFrame, budget_k: int, points: int = 121) -> dict:
    total = df["expected_rake_at_risk"].sum()
    keys = {
        "expected": "expected_rake_at_risk",
        "risk": "risk_cal",
        "money": "weekly_rake_usd",
    }
    # player_id is unique only WITHIN a venue — the same code on two sites is two
    # different people (and must stay that way). Identity is site + player_id.
    df = df.with_columns((pl.col("site") + ":" + pl.col("player_id")).alias("key"))
    top_ids: dict[str, set] = {}
    out: dict[str, dict] = {}
    n = df.height
    for name, col in keys.items():
        # the key breaks ties deterministically; risk_cal is stored rounded to 4 dp,
        # so its ties are many and the tie-break is a real (documented) choice.
        ranked = df.sort([col, "key"], descending=[True, False])
        cum = ranked["expected_rake_at_risk"].cum_sum()
        curve = [(0.0, 0.0)]
        for i in range(1, points):
            idx = min(int(round(i / (points - 1) * n)) - 1, n - 1)
            curve.append(((idx + 1) / n, float(cum[idx])))
        head = ranked.head(budget_k)
        out[name] = {
            "curve": curve,
            "at_budget": float(cum[budget_k - 1]),
            "lapsers": int(head["lapsed"].sum()),
            "precision": float(head["lapsed"].sum() / budget_k),
            "realised": float(head.filter(pl.col("lapsed") == 1)["weekly_rake_usd"].sum()),
        }
        top_ids[name] = set(head["key"].to_list())
        assert len(top_ids[name]) == budget_k, "site+player_id is not unique"
    for name in out:
        out[name]["overlap"] = len(top_ids[name] & top_ids["expected"]) / budget_k
    out["_total"] = total
    return out


def load_gold() -> pl.DataFrame:
    if not GOLD.exists():
        sys.exit(f"missing {GOLD} — run src/rake_at_risk.py first")
    return pl.read_parquet(str(GOLD / "*.parquet"))


# ------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only; do not write")
    args = ap.parse_args()

    docs = ROOT / "docs"
    rake = json.loads((docs / "rake-at-risk-results.json").read_text())
    forka = json.loads((docs / "fork-a-results.json").read_text())
    segs = json.loads((docs / "segments-results.json").read_text())

    df = load_gold()
    players = df.height
    budget_k = int(round(players * BUDGET))
    curves = ranking_curves(df, budget_k)
    total = curves["_total"]

    # ---- regression checks against the run log ------------------------------
    checks = []

    def check(label, got, want, tol):
        ok = abs(got - want) <= tol
        checks.append((label, got, want, ok))
        return ok

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
    # risk-alone reach: risk_cal ships rounded to 4 dp, so ties break differently here
    # than they did in memory during the run. Documented in rake-at-risk-results.md §11.
    check("risk-alone reach (ties, +-2%)", curves["risk"]["at_budget"],
          rake["ranking_full"]["risk alone"]["expected_at_risk_reached"],
          0.02 * rake["ranking_full"]["risk alone"]["expected_at_risk_reached"])

    print("REGRESSION CHECKS vs docs/rake-at-risk-results.json")
    for label, got, want, ok in checks:
        mark = "ok  " if ok else "FAIL"
        print(f"  [{mark}] {label:34s} got {got:>14,.2f}   want {want:>14,.2f}")
    if not all(ok for *_, ok in checks):
        sys.exit("regression checks failed — the deck was NOT written")

    # ---- everything the deck can quote --------------------------------------
    calc = {
        "players": players,
        "budget_k": budget_k,
        "total_at_risk": total,
        "expected_reach": curves["expected"]["at_budget"],
        "expected_share": curves["expected"]["at_budget"] / total,
        "expected_lapsers": curves["expected"]["lapsers"],
        "risk_reach": curves["risk"]["at_budget"],
        "risk_share": curves["risk"]["at_budget"] / total,
        "risk_lapsers": curves["risk"]["lapsers"],
        "risk_precision": curves["risk"]["precision"],
        "money_reach": curves["money"]["at_budget"],
        "money_share": curves["money"]["at_budget"] / total,
        "overlap": curves["risk"]["overlap"],
        "loose_share_of_risk": (
            rake["by_segment"][0]["share_of_risk"] + rake["by_segment"][1]["share_of_risk"]
        ),
        "loose_at_risk_usd": (
            rake["by_segment"][0]["expected_at_risk"] + rake["by_segment"][1]["expected_at_risk"]
        ),
        "loose_share_of_players": (
            rake["by_segment"][0]["players"] + rake["by_segment"][1]["players"]
        ) / players,
        # stated the way rake-at-risk-results.md §9 states it: how much of the true
        # headline the uncalibrated scores would have MISSED, not the uplift on top.
        "calibration_understatement": 1 - (
            rake["sensitivity_probability"]["shipped weighted scores"]
            / hl["expected_rake_at_risk_usd"]
        ),
        "uncalibrated_headline": rake["sensitivity_probability"]["shipped weighted scores"],
        "naive_inflation": rake["sensitivity_estimator"]["inflation_x"] - 1,
        "gbt_extra_lapsers": (
            forka["core_all_venues"]["models"]["gbt_classifier"]["budget_curve"][1][
                "true_lapsers_caught"]
            - forka["core_all_venues"]["baselines"]["recency_rank"]["budget_curve"][1][
                "true_lapsers_caught"]
        ),
        "gbt_budget_contacts": forka["core_all_venues"]["models"]["gbt_classifier"][
            "budget_curve"][1]["contacted"],
        "gbt_budget_caught": forka["core_all_venues"]["models"]["gbt_classifier"][
            "budget_curve"][1]["true_lapsers_caught"],
        "recency_budget_caught": forka["core_all_venues"]["baselines"]["recency_rank"][
            "budget_curve"][1]["true_lapsers_caught"],
    }

    # paths JSON keys make awkward to write inside an HTML attribute (dots, ">=")
    gbt = forka["core_all_venues"]["models"]["gbt_classifier"]
    hard = forka["core_all_venues"]["baselines"]["recency_rank"]["hard_rules"]
    cal = rake["calibration"]
    eco = {int(r["prediction"]): r for r in segs["segments"]["ecosystem_five_venues"]}
    names = {v: int(k) for k, v in rake["segment_names"].items()}
    lapsed_total = sum(r["lapsed_actual"] for r in rake["by_segment"])
    calc.update({
        "prevalence": forka["population"]["prevalence_lapsed"],
        "gbt_f1_default": gbt["at_0.5"]["f1"],
        "gbt_f1_tuned": gbt["best_f1"]["f1"],
        "gbt_thr_tuned": gbt["best_f1"]["threshold"],
        "quiet1_f1": hard[">= 1 days quiet"]["f1"],
        "recal_f1": cal["best_f1"]["f1"],
        "recal_thr": cal["best_f1"]["threshold"],
        "roc_before": cal["weighted"]["roc_auc"],
        "roc_after": cal["unweighted"]["roc_auc"],
        "ece_before": cal["weighted"]["reliability"]["ece"],
        "ece_after": cal["unweighted"]["reliability"]["ece"],
        "worst_decile_before": max(
            abs(b["deviation"]) for b in cal["weighted"]["reliability"]["bins"]),
        "worst_decile_after": max(
            abs(b["deviation"]) for b in cal["unweighted"]["reliability"]["bins"]),
        "grinder_bb100": eco[names["Grinder"]]["p_bb_per_100"],
        "grinder_tables": eco[names["Grinder"]]["p_max_tables"],
        "gambler_bb100": eco[names["Gambler"]]["p_bb_per_100"],
        "rec_bb100": eco[names["Recreational"]]["p_bb_per_100"],
        "loose_share_of_lapsers": (
            rake["by_segment"][0]["lapsed_actual"] + rake["by_segment"][1]["lapsed_actual"]
        ) / lapsed_total,
        "kmeans_sil": segs["algorithm_comparison"]["ecosystem_five_venues"]["kmeans_silhouette"],
        "bisecting_sil": segs["algorithm_comparison"]["ecosystem_five_venues"]["bisecting_silhouette"],
        "cluster_agreement": segs["algorithm_comparison"]["ecosystem_five_venues"]["agreement"],
        "sil_k2": segs["sweeps"]["style_all_venues"]["2"]["silhouette"],
    })
    roots = {"rake": rake, "forka": forka, "seg": segs, "calc": calc}

    if not DECK.exists():
        sys.exit(f"missing {DECK}")
    html = DECK.read_text()
    original = html

    # ---- charts -------------------------------------------------------------
    core = forka["core_all_venues"]
    blocks = {
        "reach": chart_reach(curves, total, budget_k, players),
        "baselines": chart_baselines([
            ("Flag every player", "no ranking at all", core["baselines"]["majority_class"]["roc_auc"], DIM, 500),
            ("Sort by days since last seen", "four minutes in a spreadsheet",
             core["baselines"]["recency_rank"]["roc_auc"], GOLD_, 700),
            ("MLlib · logistic regression", "readable coefficients",
             core["models"]["logistic_regression"]["roc_auc"], CYAN, 500),
            ("MLlib · random forest", "100 trees",
             core["models"]["random_forest"]["roc_auc"], CYAN, 500),
            ("MLlib · gradient-boosted trees", "the shipped model",
             core["models"]["gbt_classifier"]["roc_auc"], CYAN_LT, 700),
        ]),
        "calibration": chart_calibration(
            rake["calibration"]["weighted"]["reliability"]["bins"],
            rake["calibration"]["unweighted"]["reliability"]["bins"],
        ),
        "segtable": table_segments(rake, segs),
        "ranktable": table_ranking(curves, total, budget_k),
    }
    for name, content in blocks.items():
        pattern = re.compile(
            rf"(<!--CHART:{name}:START-->).*?(<!--CHART:{name}:END-->)", re.S
        )
        if not pattern.search(html):
            sys.exit(f"deck has no marker pair for {name!r}")
        html = pattern.sub(lambda m: m.group(1) + content + m.group(2), html)
        print(f"  built  {name:12s} {len(content):>7,} chars")

    # ---- every tagged figure ------------------------------------------------
    figure = re.compile(
        r'(?P<open><(?P<tag>[a-z]+)[^>]*class="fig"[^>]*data-fig="(?P<path>[^"]+)"'
        r'[^>]*data-fmt="(?P<fmt>[^"]+)"[^>]*>)(?P<text>[^<]*)(?P<close></(?P=tag)>)'
    )
    changed, seen = [], 0

    def substitute(m):
        nonlocal seen
        seen += 1
        value = resolve(roots, m.group("path"))
        rendered = fmt(value, m.group("fmt"))
        if rendered != m.group("text"):
            changed.append((m.group("path"), m.group("text"), rendered))
        return m.group("open") + rendered + m.group("close")

    html = figure.sub(substitute, html)
    print(f"\n{seen} tagged figures resolved, {len(changed)} rewritten")
    for path, was, now in changed:
        print(f"  {path:52s} {was!r} -> {now!r}")

    if args.check:
        if html != original:
            sys.exit("\nDECK IS STALE — re-run without --check")
        print("\ndeck is current")
        return 0

    DECK.write_text(html)
    print(f"\nwrote {DECK.relative_to(ROOT)}  ({len(html):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
