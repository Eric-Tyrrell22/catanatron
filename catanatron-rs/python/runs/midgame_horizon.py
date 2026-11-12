"""Summarises examples/midgame_horizon output: for each candidate value label (game outcome z, one-ply
value v1, search root value q, horizon values h<K>_<P>) the Pearson and Spearman correlation with the
truth (win rate over the truth playouts), overall and by game phase, plus each label's cost in forwards.

    runs/midgame_horizon.py <probe output>
"""
import sys
import numpy as np

COLS = ["truth", "z", "v1", "q", "h3_4", "h3_12", "h5_4", "h5_12", "h10_4", "h10_12", "h20_4", "h20_12"]
rows, times = [], ""
for l in open(sys.argv[1]):
    p = l.split()
    if p and p[0] == "pos":
        rows.append([int(p[1]), int(p[2]), int(p[3]), int(p[4])] + [float(x) for x in p[5:]])
    elif p and p[0] == "time":
        times = l.strip()
a = np.array(rows)
turn, truth = a[:, 1], a[:, 4]
print(f"{len(a)} positions from {len(set(a[:, 0]))} games; truth = win rate over 64 playouts (sd of a label-free coin at p=0.5: 0.06)")
print(times)
def spearman(x, y):
    return np.corrcoef(x.argsort().argsort(), y.argsort().argsort())[0, 1]
def report(sel, name):
    t = truth[sel]
    if len(t) < 20:
        return
    print(f"\n{name}: {len(t)} positions, truth mean {t.mean():.2f} sd {t.std():.2f}")
    print(f"  {'label':>7} {'pearson':>8} {'spearman':>9}   forwards/label")
    cost = {"z": "0 (free)", "v1": "1", "q": "400", "h3_4": "~4x3 turns", "h3_12": "~12x3 turns", "h5_4": "~4x5", "h5_12": "~12x5", "h10_4": "~4x10", "h10_12": "~12x10", "h20_4": "~4x20", "h20_12": "~12x20"}
    for j, c in enumerate(COLS[1:], start=5):
        x = a[sel, j]
        print(f"  {c:>7} {np.corrcoef(x, t)[0, 1]:8.3f} {spearman(x, t):9.3f}   {cost[c]}")
report(np.ones(len(a), bool), "ALL")
for lo, hi in ((0, 20), (20, 40), (40, 60), (60, 200)):
    report((turn >= lo) & (turn < hi), f"turns {lo}-{hi}")
# Decided positions dominate late; the interesting cases are the open ones.
open_pos = (truth > 0.2) & (truth < 0.8)
report(open_pos, "OPEN positions only (truth between 0.2 and 0.8)")
