"""Head-to-head follower for the sp4 specialists: for every board whose sp4 specialist is finished
(runs/sp4/results.jsonl) and not yet in runs/sp4/h2h.jsonl, plays its best snapshot (with its opening
network) against the sp3 specialist of the same board: network only (--nn-games) and with search on
both sides (--sims, --search-games). Appends {"seed", "nn", "search", "nn_games", "search_games"}.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/sp4_h2h.py [--nn-games 1000] [--search-games 100] [--sims 600] [--threads 6] [--loop MINUTES]
"""
import json, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1
args = sys.argv[1:]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
NN_GAMES, S_GAMES, SIMS, THREADS, LOOP = opt("--nn-games", 1000), opt("--search-games", 100), opt("--sims", 600), opt("--threads", 6), opt("--loop", 0)
OUT = "runs/sp4/h2h.jsonl"
def best_snap(results_path, seed):
    for line in open(results_path):
        r = json.loads(line)
        if r["seed"] == seed and r.get("evals"):
            it = max(r["evals"], key=lambda e: e["win"])["iter"]
            snap = os.path.abspath(f"runs/{r['run']}/snap_{it:06d}.ctnn"); osnap = os.path.join(os.path.dirname(snap), "o" + os.path.basename(snap))
            return snap + (f"@{osnap}" if os.path.exists(osnap) else "")
mcts = lambda path: f"MCTS:{SIMS},c1,f0.25,m0.1,s0.3/150:{path}"
while True:
    done = {json.loads(l)["seed"] for l in open(OUT)} if os.path.exists(OUT) else set()
    todo = [json.loads(l)["seed"] for l in open("runs/sp4/results.jsonl")] if os.path.exists("runs/sp4/results.jsonl") else []
    for seed in [s for s in todo if s not in done]:
        new, old = best_snap("runs/sp4/results.jsonl", seed), best_snap("runs/sp3/results.jsonl", seed)
        if not new or not old:
            continue
        t0 = time.time()
        (w, _), _ = catanatron_rs.play_bots(["NN:" + new, "NN:" + old], NN_GAMES, 4242, THREADS, map_type="BASE", board=f"seed:{seed}", **OPENHEX_1V1)
        (ws, _), _ = catanatron_rs.play_bots([mcts(new), mcts(old)], S_GAMES, 4242, THREADS, map_type="BASE", board=f"seed:{seed}", **OPENHEX_1V1)
        rec = {"seed": seed, "nn": w / NN_GAMES, "search": ws / S_GAMES, "nn_games": NN_GAMES, "search_games": S_GAMES, "sims": SIMS}
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"{time.strftime('%H:%M')} board {seed}: sp4 vs sp3 network-only {100 * rec['nn']:.1f}%, both at {SIMS} sims {100 * rec['search']:.0f}% ({time.time() - t0:.0f}s)", flush=True)
    if not LOOP:
        break
    time.sleep(LOOP * 60)
