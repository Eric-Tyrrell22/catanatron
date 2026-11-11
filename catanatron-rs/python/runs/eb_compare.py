"""eb0 (control) vs eb1 (entropy budget): score vs Ham at matched game counts.
Reads each run's log.jsonl for iteration -> games and the scorer logs for
iteration -> win rate, then prints both arms side by side by games played."""
import json, re, sys
def load(run):
    games = {}
    for l in open(f"runs/{run}/log.jsonl"):
        d = json.loads(l); games[d["iter"]] = d["games"]
    scores = {}
    for l in open(f"runs/{run}.vs_ham"):
        m = re.search(r"snap_(\d+)\.ctnn: ([\d.]+)% vs Ham", l)
        if m: scores[int(m.group(1))] = float(m.group(2))
    return [(games.get(i, 0), i, p) for i, p in sorted(scores.items())]
a, b = load("eb0"), load("eb1")
print(f"{'games':>8s} | {'eb0 control':>18s} | {'eb1 entropy':>18s}")
rows = sorted({round(g, -3) for g, _, _ in a + b})
def at(run, g):
    c = [x for x in run if abs(x[0] - g) <= 1500]
    return f"it{c[-1][1]:>4d} {c[-1][2]:5.1f}%" if c else ""
for g in rows:
    print(f"{g:8d} | {at(a, g):>18s} | {at(b, g):>18s}")
