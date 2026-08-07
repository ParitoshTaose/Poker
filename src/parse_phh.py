"""Parse PHH (.phhs) poker hand-history files into flat tables."""
import tomllib
from collections import Counter
from pathlib import Path

STREETS = ["preflop", "flop", "turn", "river"]


# TOML types are VALUE-dependent: `25` loads as int, `25.0` as float. So the same
# field is int in one hand and float in the next, which makes Polars' schema
# inference unreliable and blows up the Parquet write on a mixed batch.
# Every value below is coerced to one fixed type at the source.
def _f(x, nd=4):
    """-> float or None"""
    return None if x is None else round(float(x), nd)


def _i(x):
    """-> int or None"""
    return None if x is None else int(x)


def _s(x):
    """-> str or None"""
    return None if x is None else str(x)


def _folder_meta(path):
    """Pull site/stake out of the folder names.

    Handles BOTH layouts:
      Bronze   .../venue=PS/stake=0.25/file.phhs
      original .../PS-2009-07-01_..._25NLH_OBFU/0.25/file.phhs
    """
    parent, grand = path.parent.name, path.parent.parent.name
    if grand.startswith("venue="):
        site, venue_dir = grand.split("=", 1)[1], None
    else:
        site, venue_dir = grand.split("-")[0], grand
    stake = parent.split("=", 1)[1] if parent.startswith("stake=") else parent
    try:
        stake = float(stake)
    except ValueError:
        stake = None
    return site, venue_dir, stake


def parse_file(path):
    """Return (hands, hand_players, actions, drops) — first three are lists of dicts."""
    path = Path(path)
    site, venue_dir, stake = _folder_meta(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    hands, hand_players, actions = [], [], []
    drops = Counter()

    for local_id, h in raw.items():
        # A few hands carry no id at all. Drop that ONE hand, never the whole file.
        hid = h.get("hand")
        if hid is None:
            drops["no_hand_id"] += 1
            continue
        # Venues disagree on type: most use ints, Ongame uses strings like 'R5-2483622-63'.
        # Force one type or the Parquet write fails on a mixed batch.
        hid = str(hid)
        uid = f"{site}:{hid}"          # globally unique — ids are only unique WITHIN a venue

        players = h["players"]
        n = len(players)
        blinds = [float(x) for x in h.get("blinds_or_straddles", [])]
        bb = blinds[1] if len(blinds) > 1 else None
        if not bb:
            drops["no_big_blind"] += 1            # cannot normalise money without a big blind
            continue

        committed = [0.0] * n                     # total put in across the whole hand
        street_bet = [0.0] * n                    # amount put in on the current street
        vol = [False] * n                         # did they CHOOSE to put money in?
        raised_pf = [False] * n
        n_bets = [0] * n                          # postflop bets/raises
        n_calls = [0] * n                         # postflop calls
        showed = [False] * n
        folded = [False] * n

        for i, a in enumerate(h.get("antes", [])):
            if i < n:
                committed[i] += float(a)
        for i, b in enumerate(blinds):
            if i < n:
                committed[i] += b
                street_bet[i] = b

        street = 0
        order = 0
        for act in h["actions"]:
            parts = act.split()
            if parts[0] == "d":
                if parts[1] == "db":              # board card dealt -> new betting street
                    street += 1
                    street_bet = [0.0] * n
                continue

            idx = int(parts[0][1:]) - 1
            verb = parts[1]
            amount = 0.0
            kind = verb

            if verb == "cbr":                     # bet or raise TO this street total
                target = float(parts[2])
                amount = target - street_bet[idx]
                committed[idx] += amount
                street_bet[idx] = target
                vol[idx] = True
                kind = "bet" if max(street_bet) == target and sum(
                    1 for x in street_bet if x > 0) <= 1 else "raise"
                if street == 0:
                    raised_pf[idx] = True
                else:
                    n_bets[idx] += 1
            elif verb == "cc":                    # CHECK or CALL - must disambiguate
                target = max(street_bet)
                amount = target - street_bet[idx]
                if amount > 0:                    # money moved -> it was a CALL
                    committed[idx] += amount
                    street_bet[idx] = target
                    vol[idx] = True
                    kind = "call"
                    if street > 0:
                        n_calls[idx] += 1
                else:
                    kind = "check"
            elif verb == "f":
                kind = "fold"
                folded[idx] = True
            elif verb == "sm":
                kind = "show"
                showed[idx] = True

            order += 1
            actions.append({
                "hand_uid": uid, "hand_id": hid, "player_id": _s(players[idx]),
                "street": STREETS[min(street, 3)], "action": kind,
                "amount": _f(amount), "amount_bb": _f(amount / bb),
                "action_order": _i(order),
            })

        # uncalled portion of the final bet is returned to its owner
        s = sorted(committed, reverse=True)
        uncalled = (s[0] - s[1]) if n > 1 else 0.0
        pot = sum(committed) - uncalled
        saw_flop = street >= 1
        w = [float(x) for x in h["winnings"]] if h.get("winnings") else None
        rake = None
        if w and sum(w) > 0:
            r = round(pot - sum(w), 4)
            if -0.005 <= r <= max(pot * 0.15, 0.02):
                rake = max(r, 0.0)

        hands.append({
            "hand_uid": uid, "hand_id": hid, "site": site,
            "venue": _s(h.get("venue")), "venue_dir": _s(venue_dir),
            "stake": _f(stake), "big_blind": _f(bb), "table_id": _s(h.get("table")),
            "year": _i(h.get("year")), "month": _i(h.get("month")), "day": _i(h.get("day")),
            "time": _s(h.get("time")), "n_players": _i(n),
            "seat_count": _i(h.get("seat_count") or n),
            "saw_flop": bool(saw_flop), "showdown": bool(any(showed)),
            "pot": _f(pot), "pot_bb": _f(pot / bb),
            "rake": _f(rake), "rake_bb": _f(rake / bb) if rake is not None else None,
        })

        for i, pid in enumerate(players):
            won = w[i] if w and i < len(w) else None
            hand_players.append({
                "hand_uid": uid, "hand_id": hid, "player_id": _s(pid),
                "position_idx": _i(i + 1),                 # p1=SB, p2=BB, pN=button
                "is_sb": i == 0, "is_bb": i == 1, "is_button": i == n - 1,
                "starting_stack": _f(h["starting_stacks"][i]),
                "invested": _f(committed[i]),
                "invested_bb": _f(committed[i] / bb),
                "winnings": _f(won),
                "net_bb": _f((won - committed[i]) / bb) if won is not None else None,
                "vpip": bool(vol[i]), "pfr": bool(raised_pf[i]), "folded": bool(folded[i]),
                "showed": bool(showed[i]), "postflop_bets": _i(n_bets[i]),
                "postflop_calls": _i(n_calls[i]),
            })

    return hands, hand_players, actions, drops


if __name__ == "__main__":
    import sys, collections, statistics
    H, HP, A, D = parse_file(sys.argv[1])
    print(f"hands={len(H)}  hand_players={len(HP)}  actions={len(A)}")
    print(f"dropped hands: {dict(D) or 'none'}")
    print(f"actions per hand = {len(A)/len(H):.2f}   players per hand = {len(HP)/len(H):.2f}")
    print("action mix:", collections.Counter(a["action"] for a in A).most_common())
    raked = [h for h in H if h["rake"] is not None]
    pf = [h for h in raked if not h["saw_flop"]]
    fl = [h for h in raked if h["saw_flop"]]
    print(f"reconciled rake on {len(raked)} hands  (preflop-only {len(pf)}, flop {len(fl)})")
    if pf: print("  preflop rake values:", sorted(set(h['rake'] for h in pf)))
    if fl: print(f"  flop rake median = {statistics.median(h['rake'] for h in fl):.2f}"
                 f"  median % = {statistics.median(h['rake']/h['pot']*100 for h in fl):.2f}%")
    vp = sum(1 for r in HP if r["vpip"]) / len(HP) * 100
    pr = sum(1 for r in HP if r["pfr"]) / len(HP) * 100
    print(f"pooled VPIP = {vp:.1f}%   pooled PFR = {pr:.1f}%   (all players mixed together)")
    bb_seats = [r for r in HP if r["is_bb"]]
    print(f"big blinds who 'vpip'd = {sum(1 for r in bb_seats if r['vpip'])/len(bb_seats)*100:.1f}%"
          "  <- should NOT be ~100%, else the check/call bug is present")
