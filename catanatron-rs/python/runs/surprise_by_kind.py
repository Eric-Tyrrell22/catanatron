"""Search surprise by the kind of action played: plays N self-play games under
the training search settings (the on3 control's self-play block, search targets
at placements) and groups KL(target || prior) by the kind of the move the search
chose. Per kind: how many recorded decisions, their share, mean and p90
surprise, the share above 0.2 nats (a real correction), and the kind's share of
all surprise, which is where the prior is still being taught. Opening
placements are reported on their own rows.

    PYTHONPATH=<branch>/target/pylib_probe .venv/bin/python runs/surprise_by_kind.py <label> <main.ctnn[@opening.ctnn]> [games=64] [threads=12]
"""
import json
import sys
import time

import numpy as np

import catanatron_rs

label, key = sys.argv[1], sys.argv[2]
args = [a for a in sys.argv[3:] if not a.startswith("--") and (sys.argv.index(a) == 0 or sys.argv[sys.argv.index(a) - 1] != "--focus-sims")]
games = int(args[0]) if len(args) > 0 else 64
threads = int(args[1]) if len(args) > 1 else 12
main_path, _, opening_path = key.partition("@")
main = open(main_path, "rb").read()
opening = open(opening_path, "rb").read() if opening_path else None
sp = {k: v for k, v in json.load(open("/home/eric/.claude/jobs/d24dadb6/tmp/control_on3.json"))["self_play"].items() if k not in ("pool", "pool_share")}
sp["placement_horizon_turns"] = 0
if "--focus-sims" in sys.argv:  # deeper search on the focus kinds (roads, settlements, robber, trades, tempo decisions)
    sp["focus_sims"] = int(sys.argv[sys.argv.index("--focus-sims") + 1])
t0 = time.time()
d = catanatron_rs.self_play(main, games, 31337, threads, opening_net=opening, **sp)
s = np.frombuffer(d["surprise"], "<f4")
kind = np.frombuffer(d["kind"], np.uint8)
op = np.frombuffer(d["opening"], np.uint8).astype(bool)
names = d["kind_names"]
assert len(kind) == len(s), (len(kind), len(s))
total = s.sum()
WIDTH = 30
print(f"{label}: {games} games, {len(s)} recorded decisions, mean surprise {s.mean():.3f} nats, focused decisions {d.get('focused')} ({time.time() - t0:.0f}s)")
print(f"{'kind':>16} {'n':>6} {'share':>6} {'mean':>6} {'p90':>6} {'>0.2':>6} {'of all':>7}  share of all surprise")
rows = []
for k, name in enumerate(names):
    for phase, sel in (("open", (kind == k) & op), ("", (kind == k) & ~op)):
        if sel.sum() == 0:
            continue
        x = s[sel]
        rows.append((x.sum() / total, f"{name} (open)" if phase else name, len(x), len(x) / len(s), x.mean(), np.percentile(x, 90), (x > 0.2).mean()))
rows.sort(reverse=True)
for share_all, name, n, share, mean, p90, gt in rows:
    print(f"{name:>16} {n:>6} {100 * share:>5.1f}% {mean:>6.3f} {p90:>6.3f} {100 * gt:>5.1f}% {100 * share_all:>6.1f}%  {'#' * int(round(share_all * WIDTH))}")
