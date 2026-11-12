"""Per-decision search-surprise distribution of a network under the training search settings:
plays N self-play games locally (search placements, the on3 control's self-play settings) and
reports, for mid-game and opening rows separately, the mean, median, p90/p99, the shares above
0.05 / 0.2 / 0.69 nats and the mean of the surprising rows. The same statistic the learner now
logs per iteration (surprise_dist), for networks whose shards are gone.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/surprise_probe.py <label> <main.ctnn[@opening.ctnn]> [games] [threads]
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python")
import catanatron_rs

label, key = sys.argv[1], sys.argv[2]
games = int(sys.argv[3]) if len(sys.argv) > 3 else 64
threads = int(sys.argv[4]) if len(sys.argv) > 4 else 12
main_path, _, opening_path = key.partition("@")
main = open(main_path, "rb").read(); opening = open(opening_path, "rb").read() if opening_path else None
sp = {k: v for k, v in json.load(open("/home/eric/.claude/jobs/d24dadb6/tmp/control_on3.json"))["self_play"].items() if k not in ("pool", "pool_share")}
sp["placement_horizon_turns"] = 0  # the search target at placements, as these networks were trained
t0 = time.time()
d = catanatron_rs.self_play(main, games, 31337, threads, opening_net=opening, **sp)
n = d["n"]; s = np.frombuffer(d["surprise"], "<f4"); op = np.frombuffer(d["opening"], np.uint8).astype(bool)
def dist(x):
    q = np.percentile(x, [50, 90, 99])
    return dict(n=len(x), mean=x.mean(), p50=q[0], p90=q[1], p99=q[2], gt05=(x > 0.05).mean(), gt20=(x > 0.2).mean(), gt69=(x > 0.69).mean(), mean_gt05=x[x > 0.05].mean() if (x > 0.05).any() else 0.0)
for name, rows in (("mid-game", s[~op]), ("opening", s[op])):
    r = dist(rows)
    print(f"{label:10s} {name:9s} n {r['n']:6d} mean {r['mean']:.3f} median {r['p50']:.3f} p90 {r['p90']:.3f} p99 {r['p99']:.3f} | >0.05 {100*r['gt05']:4.1f}% >0.2 {100*r['gt20']:4.1f}% >0.69 {100*r['gt69']:4.1f}% | mean of >0.05: {r['mean_gt05']:.3f}", flush=True)
print(f"{label:10s} ({games} games in {time.time()-t0:.0f}s)")
