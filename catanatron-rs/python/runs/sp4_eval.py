"""Evaluates a trial sp4 specialist (from on6) on its board: every snapshot against the
on6 base (network only, seats alternating), the best one against the sp3 specialist of
the same board, and the first picks: the new specialist's policy argmax, the old expert's,
and the forced win rate of each (runs/expert_picks2/forced).

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/sp4_eval.py <board seed> [--games 1000] [--threads 8]
"""
import glob, json, os, re, subprocess, sys
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1
args = sys.argv[1:]
opt = lambda k, d: type(d)(args.pop(args.index(k) + 1)) if k in args else d
GAMES, THREADS = opt("--games", 1000), opt("--threads", 8)
seed = int(args[0]); run = opt("--run", f"sp4-{seed}")
WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
key = lambda snap: "NN:" + os.path.abspath(snap) + "@" + os.path.abspath(os.path.join(os.path.dirname(snap), "o" + os.path.basename(snap)))
base = key(sorted(glob.glob("runs/on6/snap_*.ctnn"))[-1])
play = lambda a, b: catanatron_rs.play_bots([a, b], GAMES, 4242, THREADS, map_type="BASE", board=f"seed:{seed}", **OPENHEX_1V1)[0][0] / GAMES
sp3 = None
for line in open("runs/sp3/results.jsonl"):
    r = json.loads(line)
    if r["seed"] == seed and r.get("evals"):
        best = max(r["evals"], key=lambda e: e["win"])["iter"]; sp3 = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn"); sp3_win = r["best_win"]
print(f"board {seed}: on6 base {os.path.basename(base.split('@')[0])}; sp3 specialist {os.path.basename(sp3)} (won {100 * sp3_win:.1f}% vs its o3 base)")
print(f"  on6 base vs sp3 specialist (network only): {100 * play(base, 'NN:' + sp3):.1f}%")
results = []
for snap in sorted(glob.glob(f"runs/{run}/snap_*.ctnn")):
    w = play(key(snap), base); results.append((w, snap))
    print(f"  {os.path.basename(snap)} vs on6 base: {100 * w:.1f}%", flush=True)
if results:
    w, best = max(results)
    print(f"  best {os.path.basename(best)}: vs on6 base {100 * w:.1f}%; vs sp3 specialist {100 * play(key(best), 'NN:' + sp3):.1f}%")
    forced = {}
    for l in open(f"runs/expert_picks2/forced/{seed}.txt"):
        m = re.search(r"forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m: forced[int(m.group(1))] = float(m.group(2))
    out = subprocess.run([f"{WT}/target/release/examples/node_features", str(seed), f"new={best.replace('snap_', 'snap_')}@{os.path.join(os.path.dirname(best), 'o' + os.path.basename(best))}", f"old={sp3}", f"on6={base[3:]}"], capture_output=True, text=True).stdout
    for l in out.splitlines():
        if l.startswith("#policy"):
            name = l.split()[1]; items = {int(a): float(b) for a, b in (it.split(":") for it in l.split()[2:])}
            top = sorted(items, key=items.get, reverse=True)[:3]
            print(f"  first pick {name:>4}: " + ", ".join(f"node {n} p={items[n]:.2f} forced {forced.get(n, float('nan')):.0f}%" for n in top))
