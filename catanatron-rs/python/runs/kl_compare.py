"""Four arms vs Ham at matched game counts: kc (control), ke (entropy budget),
kk (KL escalation), kka (KL escalation + admitted fast positions)."""
import json, re
import sys
ARMS = [tuple(a.split(":")) for a in sys.argv[1:]] if len(sys.argv) > 1 else [("kc", "control"), ("ke", "entropy"), ("kk", "kl-escalate"), ("kka", "kl+admit")]
def load(run):
    games = {}
    try:
        for l in open(f"runs/{run}/log.jsonl"):
            d = json.loads(l); games[d["iter"]] = d["games"]
    except FileNotFoundError:
        return []
    scores = {}
    try:
        for l in open(f"runs/{run}.vs_ham"):
            m = re.search(r"snap_(\d+)\.ctnn: ([\d.]+)% vs Ham", l)
            if m: scores[int(m.group(1))] = float(m.group(2))
    except FileNotFoundError:
        pass
    return [(games.get(i, 0), i, p) for i, p in sorted(scores.items())]
data = {run: load(run) for run, _ in ARMS}
print(f"{'games':>8s} | " + " | ".join(f"{name:>14s}" for _, name in ARMS))
rows = sorted({round(g, -4) for d in data.values() for g, _, _ in d})
def at(d, g):
    c = [x for x in d if abs(x[0] - g) <= 5000]
    return f"{c[-1][2]:5.1f}% @{c[-1][0]//1000}k" if c else ""
for g in rows:
    print(f"{g:8d} | " + " | ".join(f"{at(data[run], g):>14s}" for run, _ in ARMS))
for run, name in ARMS:
    if data[run]:
        last = data[run][-1]
        print(f"{name}: latest it {last[1]} ({last[0]:,} games) {last[2]:.1f}%; best {max(p for _, _, p in data[run]):.1f}%")
