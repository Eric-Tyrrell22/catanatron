"""Fast data test for the spot branch, no self-play: fine-tune two copies of
o4 on the same saved positions, then play each network-only vs o4.

  control  o4 alone, all weights at lr 5e-5
  branch   o4 + spot branch (trunk lr 5e-5, branch lr 5e-4), the trunk's value
           fading from 1 to 0 over the first 60% of the steps (sb1's design)

Same data, steps, batch, symmetries, opening share and losses (policy,
value, aux) as az.py's training. Writes runs/ft_<name>.ctnn and prints the
held-out losses on the newest 10% of the replay (not trained on).

    python runs/spot_finetune_check.py <replay.pt> [steps]
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
import catanatron_rs
import model as M
import spots
from export import to_bytes

dev = torch.device("cuda")
REPLAY = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
ck = torch.load("runs/o4/model.pt", map_location="cpu")
sd, meta = ck["model"], ck["meta"]
OBS, ACT = meta["obs_size"], meta["action_size"]
AUX = sd["aux.weight"].shape[0]
tables = spots.tables_for(OBS, opponent_hands=True, opponent_devs=True, extra_info=True, spot_features=True)

d = torch.load(REPLAY, map_location="cpu", mmap=True)
total, oldest = d["z"].shape[0], int(d["oldest"])
order = torch.cat([torch.arange(oldest, total), torch.arange(0, oldest)])
hold = order[-total // 10:].sort().values
train = order[:-total // 10]
data = {k: d[k].clone() for k in ("obs", "mask", "pi", "z", "aux", "opening")}
opening = train[data["opening"][train]]
print(f"{total} positions ({len(train)} train, {len(hold)} held out, {len(opening)} opening)", flush=True)
perms = catanatron_rs.board_symmetries(OBS, "BASE")
sym_obs = torch.tensor([p[0] for p in perms], device=dev)
sym_act = torch.tensor([p[1] for p in perms], device=dev)


def batch(idx, sym=True):
    o, m, p = data["obs"][idx].to(dev).float(), data["mask"][idx].to(dev), data["pi"][idx].to(dev).float()
    z, a = data["z"][idx].to(dev).float(), data["aux"][idx].to(dev).float()
    if sym:
        k = torch.randint(0, sym_obs.shape[0], (len(idx),), device=dev)
        o, m, p = o.gather(1, sym_obs[k]), m.gather(1, sym_act[k]), p.gather(1, sym_act[k])
    return o, m, p, z, a


def losses(net, o, m, p, z, a):
    logits, v, aux = net.forward_aux(o, m)
    ce = -(p * F.log_softmax(logits, -1).masked_fill(~m, 0)).sum(-1).mean()
    return ce, F.mse_loss(v, z), F.mse_loss(aux, a)


@torch.no_grad()
def held_out(net):
    tot = [0.0, 0.0]
    for i in range(0, len(hold), 8192):
        ce, mse, _ = losses(net, *batch(hold[i:i + 8192], sym=False))
        n = len(hold[i:i + 8192])
        tot[0] += ce.item() * n; tot[1] += mse.item() * n
    return tot[0] / len(hold), tot[1] / len(hold)


for name in ("control", "branch"):
    torch.manual_seed(0)
    net = M.Policy(OBS, ACT, 512, AUX, 4, spots=tables if name == "branch" else None).to(dev)
    net.load_state_dict(sd, strict=False)
    br = [p for n, p in net.named_parameters() if n.startswith("spots.")]
    rest = [p for n, p in net.named_parameters() if not n.startswith("spots.")]
    opt = torch.optim.AdamW([{"params": rest, "lr": 5e-5}] + ([{"params": br, "lr": 5e-4}] if br else []), weight_decay=1e-4)
    net.eval()
    print(f"{time.strftime('%H:%M')} {name} start: held out ce {held_out(net)[0]:.4f} mse {held_out(net)[1]:.4f}", flush=True)
    net.train()
    t = time.time()
    for step in range(1, STEPS + 1):
        if name == "branch":
            net.flat_value_scale.fill_(max(0.0, 1.0 - step / (0.6 * STEPS)))
        n_open = round(2048 * 0.3)
        idx = torch.cat([train[torch.randint(0, len(train), (2048 - n_open,))], opening[torch.randint(0, len(opening), (n_open,))]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, mse, aux = losses(net, *batch(idx.sort().values))
        loss = ce + mse + 0.5 * aux
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5000 == 0 or step == STEPS:
            net.eval()
            hce, hmse = held_out(net)
            net.train()
            print(f"{time.strftime('%H:%M')} {name} step {step}: held out ce {hce:.4f} mse {hmse:.4f} ({time.time() - t:.0f}s)", flush=True)
    net.eval()
    with open(f"runs/ft_{name}.ctnn", "wb") as f:
        f.write(to_bytes(net.cpu(), OBS, ACT))
    print(f"wrote runs/ft_{name}.ctnn", flush=True)
