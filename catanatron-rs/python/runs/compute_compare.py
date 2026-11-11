"""Arms vs Ham at matched ACTOR COMPUTE (actor-hours of self-play), for arms
whose games cost different amounts (e.g. fewer sims per move).

    python runs/compute_compare.py ac:control:15 ls:low-sims:5

Each arm is run:name:seconds_per_shard (actor play time per 256-game shard,
from the actor logs). Actor-hours = cumulative remote shards * that.
"""
import json, re, sys
ARMS = [a.split(":") for a in sys.argv[1:]]


def load(run, shard_s):
    rows, hours = {}, 0.0
    for l in open(f"runs/{run}/log.jsonl"):
        d = json.loads(l)
        hours += d.get("remote_shards", 0) * float(shard_s) / 3600
        rows[d["iter"]] = (hours, d["games"])
    scores = {}
    try:
        for l in open(f"runs/{run}.vs_ham"):
            m = re.search(r"snap_(\d+)\.ctnn: ([\d.]+)% vs Ham", l)
            if m:
                scores[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        pass
    return [(rows[i][0], rows[i][1], i, p) for i, p in sorted(scores.items()) if i in rows]


data = {run: load(run, s) for run, _, s in ARMS}
print(f"{'actor-h':>8s} | " + " | ".join(f"{name:>22s}" for _, name, _ in ARMS))
step = 0.5
top = max((h for d in data.values() for h, _, _, _ in d), default=0)
h = step
while h <= top + 1e-9:
    cells = []
    for run, _, _ in ARMS:
        c = [x for x in data[run] if x[0] <= h]
        cells.append(f"{c[-1][3]:5.1f}% @{c[-1][1] // 1000}k games" if c and c[-1][0] > h - step else "")
    print(f"{h:8.1f} | " + " | ".join(f"{x:>22s}" for x in cells))
    h += step
for run, name, _ in ARMS:
    if data[run]:
        hh, g, i, p = data[run][-1]
        print(f"{name}: {hh:.1f} actor-hours, {g:,} games, it {i}: {p:.1f}%; best {max(x[3] for x in data[run]):.1f}%")
