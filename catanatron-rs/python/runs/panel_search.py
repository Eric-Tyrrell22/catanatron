"""A network with play-time search against the board specialists (network only, or
also searching) on a few panel boards, OpenHex rules: does search recover board
knowledge the opening policy lacks? Prints per board the win rate for each
configuration and the panel's network-only number for reference.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/panel_search.py <run> <board,...> [--games 100] [--sims 600] [--threads 10] [--spec-sims 0]
"""
import glob, json, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1
args = sys.argv[1:]
opt = lambda k, d: type(d)(args.pop(args.index(k) + 1)) if k in args else d
GAMES, SIMS, THREADS, SPEC_SIMS = opt("--games", 100), opt("--sims", 600), opt("--threads", 10), opt("--spec-sims", 0)
run, boards = args[0], [int(b) for b in args[1].split(",")]
spec = {}
for line in open("runs/sp3/results.jsonl"):
    r = json.loads(line)
    if r.get("evals"):
        best = max(r["evals"], key=lambda e: e["win"])["iter"]; spec[r["seed"]] = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn")
snap = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))[-1]; osnap = os.path.join(os.path.dirname(snap), "o" + os.path.basename(snap))
net = os.path.abspath(snap) + (f"@{os.path.abspath(osnap)}" if os.path.exists(osnap) else "")
panel_lines = [l for l in open(f"runs/{run}.panel100") if "|" in l]
panel = {int(a): float(b.rstrip("%")) for a, b in (x.split(":") for x in panel_lines[-1].split("|")[1].split())}
mcts = lambda path, sims: f"MCTS:{sims},c1,f0.25,m0.1,s0.3/150:{path}"
print(f"{run} {os.path.basename(snap)} vs specialists on {len(boards)} boards, {GAMES} games each; our search {SIMS} sims, specialist {'network only' if not SPEC_SIMS else f'{SPEC_SIMS} sims'}", flush=True)
tot = [0, 0]
for b in boards:
    t0 = time.time()
    (w, _), _ = catanatron_rs.play_bots([mcts(net, SIMS), ("NN:" + spec[b]) if not SPEC_SIMS else mcts(spec[b], SPEC_SIMS)], GAMES, 4242, THREADS, map_type="BASE", board=f"seed:{b}", **OPENHEX_1V1)
    tot[0] += w; tot[1] += GAMES
    print(f"  board {b}: network-only panel {panel.get(b, float('nan')):.1f}%  ->  with search {100 * w / GAMES:.0f}%  ({time.time() - t0:.0f}s)", flush=True)
print(f"  mean with search {100 * tot[0] / tot[1]:.1f}% over {tot[1]} games; panel mean on these boards {sum(panel[b] for b in boards) / len(boards):.1f}%")
