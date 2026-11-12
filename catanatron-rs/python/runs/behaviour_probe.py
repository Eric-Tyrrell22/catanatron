"""Terminal summary of the behaviour_probe example's output: dead roads (roads
that never joined two of the owner's buildings) for winners and losers, and
Monopoly timing: how long cards were held, whether the resource taken was the
best one at the moment, and how the haul compares with the best moment seen
while the card was held.

    runs/behaviour_probe.py <probe output> [<probe output> ...]
"""
import collections
import os
import sys

WIDTH = 30


def bar(x, scale=1.0):
    n = int(round(x / scale * WIDTH))
    return "#" * n + "." * max(0, WIDTH - n)


def summarise(path):
    roads, mono, unplayed = [], [], []
    for l in open(path):
        p = l.split()
        if not p:
            continue
        if p[0] == "roads":
            roads.append(tuple(int(x) for x in p[1:]))
        elif p[0] == "mono":
            mono.append(tuple(int(x) for x in p[1:]))
        elif p[0] == "mono_unplayed":
            unplayed.append(tuple(int(x) for x in p[1:]))
    games = len(roads) // 2
    print(f"== {os.path.basename(path)}: {games} games")
    print(f"{'dead roads':>14} {'n':>5} {'roads':>6} {'dead':>5} {'dead%':>6} {'LR%':>5}")
    for name, rows in (("winners", [r for r in roads if r[0] == 1]), ("losers", [r for r in roads if r[0] == 0]),
                       ("with LR", [r for r in roads if r[3] == 1]), ("without LR", [r for r in roads if r[3] == 0])):
        if not rows:
            continue
        tot = sum(r[1] for r in rows); dead = sum(r[2] for r in rows)
        print(f"{name:>14} {len(rows):>5} {tot / len(rows):>6.2f} {dead / len(rows):>5.2f} {100 * dead / max(1, tot):>5.1f}% {100 * sum(r[3] for r in rows) / len(rows):>4.0f}%")
    dist = collections.Counter(r[2] for r in roads)
    print(f"{'dead per player':>14}  " + "  ".join(f"{k}:{100 * dist[k] / len(roads):.0f}%" for k in sorted(dist)))
    print()
    if not mono and not unplayed:
        print("no Monopoly cards drawn\n")
        return
    n = len(mono)
    print(f"Monopoly: {n} played, {len(unplayed)} still in hand at the end ({100 * len(unplayed) / max(1, n + len(unplayed)):.0f}% of drawn); "
          f"{n / games:.2f} plays per game")
    if mono:
        hold = [m[0] for m in mono]
        buckets = [(0, 1, "same/next turn"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 31, "16-31"), (32, 999, "32+")]
        print(f"{'held (game turns)':>18} {'n':>5} {'share':>6} {'haul':>5} {'best':>5}  share of plays")
        for lo, hi, name in buckets:
            rows = [m for m in mono if lo <= m[0] <= hi]
            if not rows:
                continue
            print(f"{name:>18} {len(rows):>5} {100 * len(rows) / n:>5.0f}% {sum(m[1] for m in rows) / len(rows):>5.1f} {sum(m[2] for m in rows) / len(rows):>5.1f}  {bar(len(rows) / n)}")
        haul = sum(m[1] for m in mono) / n
        best = sum(m[2] for m in mono) / n
        seen = sum(m[3] for m in mono) / n
        print(f"{'mean haul':>18} {haul:.2f} cards; best resource at that moment {best:.2f}; best moment while held {seen:.2f}")
        print(f"{'took best now':>18} {100 * sum(m[1] == m[2] for m in mono) / n:.0f}% of plays; played at the best moment seen {100 * sum(m[1] >= m[3] for m in mono) / n:.0f}%; "
              f"haul of 0-1 cards {100 * sum(m[1] <= 1 for m in mono) / n:.0f}%")
        hd = collections.Counter(min(m[1], 8) for m in mono)
        print(f"{'haul':>18}  " + "  ".join(f"{k}{'+' if k == 8 else ''}:{100 * hd[k] / n:.0f}%" for k in sorted(hd)))
        turn_vp = sorted(mono, key=lambda m: m[4])
        q = len(turn_vp) // 4
        print(f"{'played at turn':>18} median {turn_vp[len(turn_vp) // 2][4]}, quartiles {turn_vp[q][4]}-{turn_vp[3 * q][4]}; holder's VP at play median {sorted(m[5] for m in mono)[n // 2]}")
    if unplayed:
        print(f"{'unplayed':>18} held {sum(u[0] for u in unplayed) / len(unplayed):.0f} turns on average, best haul seen {sum(u[1] for u in unplayed) / len(unplayed):.1f}, "
              f"holder won {100 * sum(u[2] for u in unplayed) / len(unplayed):.0f}%")
    print()


for path in sys.argv[1:]:
    summarise(path)
