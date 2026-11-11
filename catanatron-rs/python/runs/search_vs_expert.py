"""How much play-time search does the generalist (o3-187) need to match a
board specialist on that specialist's board?  Two curves over the
generalist's simulations: vs the specialist network-only, and vs the
specialist searching with the same budget.

    python runs/search_vs_expert.py <board seed> <specialist.ctnn> [threads]
Appends rows to runs/search_vs_expert.tsv.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import catanatron_rs
import model as M
from export import to_bytes
from catanatron_env import OPENHEX_1V1

seed, spec = int(sys.argv[1]), os.path.abspath(sys.argv[2])
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 10
ck = torch.load("runs/o3/model.pt", map_location="cpu", weights_only=False)
m = ck["meta"]
net = M.Policy(m["obs_size"], m["action_size"], m["hidden"], m.get("aux", 0), m["layers"])
net.load_state_dict(ck["model"]); net.eval()
gen = os.path.abspath("runs/o3/o3_187.ctnn")
with open(gen, "wb") as f:
    f.write(to_bytes(net, m["obs_size"], m["action_size"]))

def key(path, sims):
    return f"NN:{path}" if sims == 0 else f"MCTS:{sims},c1,f0.25:{path}"

out = open("runs/search_vs_expert.tsv", "a")
for sims in [0, 25, 50, 100, 200, 400, 800, 1600]:
    games = 400 if sims <= 800 else 200
    for opp_sims, label in [(0, "expert_net"), (sims, "expert_same_sims")]:
        if label == "expert_same_sims" and sims == 0:
            continue
        t = time.time()
        (w, l), capped = catanatron_rs.play_bots([key(gen, sims), key(spec, opp_sims)], games, 777_000 + sims, threads,
                                                 map_type="BASE", board=f"seed:{seed}", **OPENHEX_1V1)
        line = f"{seed}\t{sims}\t{label}\t{games}\t{w}\t{w / games:.3f}\t{capped}\t{time.time() - t:.0f}"
        print(time.strftime("%H:%M"), line, flush=True)
        out.write(line + "\n"); out.flush()
print("done", flush=True)
