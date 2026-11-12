"""Terminal chart of a network's port settlements, from the port_picks probe's
output: per turn bucket, how many settlements were built and what share sat on
a port (any port, and a 2:1 port matching the builder's most-produced resource),
against the chance line (share of land nodes that carry a port).

    runs/port_picks.py <probe output> [<probe output> ...]   # one chart per file, side by side at the end
"""
import collections
import os
import sys

BUCKETS = [(0, 0, "open 1st"), (1, 1, "open 2nd"), (2, 19, "t2-19"), (20, 39, "t20-39"), (40, 59, "t40-59"), (60, 79, "t60-79"), (80, 119, "t80-119"), (120, 400, "t120+")]
WIDTH = 40


def load(path):
    share, rows = None, []
    for l in open(path):
        p = l.split()
        if not p:
            continue
        if p[0] == "#port_share":
            share = float(p[1])
            continue
        turn, phase, port, matches, top = int(p[0]), p[1], p[2], int(p[3]), float(p[4])
        rows.append((turn, phase, port, matches, top))
    return share, rows


def bucket_of(turn, phase, opening_count):
    if phase == "open":
        return 0 if opening_count == 0 else 1
    for i, (lo, hi, _) in enumerate(BUCKETS):
        if i >= 2 and lo <= turn <= hi:
            return i
    return len(BUCKETS) - 1


def chart(path):
    share, rows = load(path)
    n = collections.Counter()
    ports = collections.Counter()
    matched = collections.Counter()
    three = collections.Counter()
    # Opening placements come in pairs per player per game; count the first and second by order within the phase.
    # The probe prints settlements in game order, so within each game the first two "open" rows per seat alternate;
    # we use the running count of opening rows modulo 4 (2 players x 2 placements).
    open_seen = 0
    for turn, phase, port, matches, top in rows:
        if phase == "open":
            b = 0 if open_seen % 4 < 2 else 1
            open_seen += 1
        else:
            b = bucket_of(turn, phase, 0)
        n[b] += 1
        if port != "none":
            ports[b] += 1
        if port == "3:1":
            three[b] += 1
        if matches:
            matched[b] += 1
    total = sum(n.values())
    print(f"{os.path.basename(path)}: {total} settlements over {open_seen // 4} games; chance line (share of land nodes with a port) {100 * share:.0f}%")
    print(f"{'bucket':>9} {'n':>5} {'port%':>6} {'3:1%':>5} {'match%':>6}  " + "chart: # = any port, = = 2:1 matching top resource, | = chance")
    for b, (_, _, name) in enumerate(BUCKETS):
        if n[b] == 0:
            continue
        p = ports[b] / n[b]
        m = matched[b] / n[b]
        t = three[b] / n[b]
        bar = ["." for _ in range(WIDTH)]
        for i in range(int(round(p * WIDTH))):
            bar[i] = "#"
        for i in range(int(round(m * WIDTH))):
            bar[i] = "="
        c = min(WIDTH - 1, int(round(share * WIDTH)))
        bar[c] = "|"
        print(f"{name:>9} {n[b]:>5} {100 * p:>5.0f}% {100 * t:>4.0f}% {100 * m:>5.0f}%  {''.join(bar)}")
    mid = [r for r in rows if r[1] == "mid"]
    if mid:
        pm = sum(r[2] != "none" for r in mid) / len(mid)
        mm = sum(r[3] for r in mid) / len(mid)
        print(f"{'all mid':>9} {len(mid):>5} {100 * pm:>5.0f}% {'':>5} {100 * mm:>5.0f}%")
    print()


for path in sys.argv[1:]:
    chart(path)
