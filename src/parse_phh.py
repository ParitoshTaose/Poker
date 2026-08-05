"""Parse PHH (.phhs) poker hand-history files into flat tables."""
import tomllib
from pathlib import Path

STREETS = ["preflop", "flop", "turn", "river"]


def parse_file(path):
    """Return (hands, hand_players, actions) as lists of dicts."""
    path = Path(path)
    venue_dir = path.parent.parent.name          # e.g. PS-2009-07-01_..._25NLH_OBFU
    stake = path.parent.name                     # e.g. 0.25
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    hands, hand_players, actions = [], [], []

    for local_id, h in raw.items():
        players = h["players"]
        n = len(players)
        blinds = h.get("blinds_or_straddles", [])
        bb = blinds[1] if len(blinds) > 1 else None
        if not bb:
            continue                              # cannot normalise money without a big blind

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
                committed[i] += a
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
                "hand_id": h["hand"], "player_id": players[idx],
                "street": STREETS[min(street, 3)], "action": kind,
                "amount": round(amount, 4), "amount_bb": round(amount / bb, 4),
                "action_order": order,
            })

        # uncalled portion of the final bet is returned to its owner
        s = sorted(committed, reverse=True)
        uncalled = (s[0] - s[1]) if n > 1 else 0.0
        pot = sum(committed) - uncalled
        saw_flop = street >= 1
        w = h.get("winnings")
        rake = None
        if w and sum(w) > 0:
            r = round(pot - sum(w), 4)
            if -0.005 <= r <= max(pot * 0.15, 0.02):
                rake = max(r, 0.0)

        hands.append({
            "hand_id": h["hand"], "venue": h.get("venue"), "venue_dir": venue_dir,
            "stake": stake, "big_blind": bb, "table_id": h.get("table"),
            "year": h.get("year"), "month": h.get("month"), "day": h.get("day"),
            "time": str(h.get("time")), "n_players": n,
            "seat_count": h.get("seat_count") or n,
            "saw_flop": saw_flop, "showdown": any(showed),
            "pot": round(pot, 4), "pot_bb": round(pot / bb, 4),
            "rake": rake, "rake_bb": round(rake / bb, 4) if rake is not None else None,
        })

        for i, pid in enumerate(players):
            won = w[i] if w and i < len(w) else None
            hand_players.append({
                "hand_id": h["hand"], "player_id": pid,
                "position_idx": i + 1,                     # p1=SB, p2=BB, pN=button
                "is_sb": i == 0, "is_bb": i == 1, "is_button": i == n - 1,
                "starting_stack": h["starting_stacks"][i],
                "invested": round(committed[i], 4),
                "invested_bb": round(committed[i] / bb, 4),
                "winnings": won,
                "net_bb": round((won - committed[i]) / bb, 4) if won is not None else None,
                "vpip": vol[i], "pfr": raised_pf[i], "folded": folded[i],
                "showed": showed[i], "postflop_bets": n_bets[i],
                "postflop_calls": n_calls[i],
            })

    return hands, hand_players, actions


if __name__ == "__main__":
    import sys, collections, statistics
    H, HP, A = parse_file(sys.argv[1])
    print(f"hands={len(H)}  hand_players={len(HP)}  actions={len(A)}")
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
