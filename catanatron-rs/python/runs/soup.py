"""Average the weights of sibling arms (all from o3-187) and score the soup."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch, catanatron_rs
import model as M
from export import to_bytes
from catanatron_env import OPENHEX_1V1
runs = ["ac", "ax", "vq25", "kc"]
cks = [torch.load(f"runs/{r}/model.pt", map_location="cpu", weights_only=False) for r in runs]
keys = [k for k in cks[0]["model"] if not k.startswith("aux.")]
avg = {k: sum(c["model"][k].float() for c in cks) / len(cks) for k in keys}
m = cks[0]["meta"]
net = M.Policy(m["obs_size"], m["action_size"], m["hidden"], 0, m["layers"]); net.load_state_dict(avg); net.eval()
path = os.path.abspath("runs/distill/soup4.ctnn")
open(path, "wb").write(to_bytes(net, m["obs_size"], m["action_size"]))
for name, ref in [("Ham", "openhex_models/ham.ctnn"), ("ac latest", "runs/ac/snap_001040.ctnn")]:
    (w, _), _ = catanatron_rs.play_bots([f"NN:{path}", f"NN:{os.path.abspath(ref)}"], 10000, 70_000, 6, map_type="BASE", **OPENHEX_1V1)
    print(f"soup of {runs} vs {name}: {w / 100:.1f}% (±0.5)", flush=True)
