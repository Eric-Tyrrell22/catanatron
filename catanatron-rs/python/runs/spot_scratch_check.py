"""Does board structure generalize better? From scratch, supervised on o4's
replay (1M positions), evaluated on positions neither network trained on:
oh8's replay (an older model's self-play on random boards).

  A  plain 4x512 MLP
  B  MLP + spot branch, value = MLP value + structured tower (the planned design)
  C  MLP + spot branch, value from the structured tower only (the PoC's design)

Same steps, batch, optimizer and board symmetries for all three.
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
import catanatron_rs
import model as M
import spots

torch.manual_seed(0)
dev = torch.device("cuda")
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
OBS = 2202
tables = spots.tables_for(OBS, opponent_hands=True, opponent_devs=True, extra_info=True, spot_features=True)


def load(path, n=None):
    d = torch.load(path, map_location="cpu", mmap=True)
    k = d["z"].shape[0] if n is None else min(n, d["z"].shape[0])
    idx = torch.randperm(d["z"].shape[0])[:k].sort().values if n is not None else slice(None)
    return d["obs"][idx].clone(), d["mask"][idx].clone(), d["pi"][idx].clone(), d["z"][idx].float().clone()


t0 = time.time()
tr = load("runs/o4/replay.pt")
te = load("runs/oh8/replay.pt", 100_000)
ACT = tr[1].shape[1]
assert tr[0].shape[1] == OBS and te[0].shape[1] == OBS, "layouts differ"
print(f"loaded {len(tr[3])} train, {len(te[3])} test positions in {time.time() - t0:.0f}s", flush=True)
perms = catanatron_rs.board_symmetries(OBS, "BASE")
sym_obs = torch.tensor([p[0] for p in perms], device=dev)
sym_act = torch.tensor([p[1] for p in perms], device=dev)


def make(kind):
    net = M.Policy(OBS, ACT, hidden=512, layers=4, spots=None if kind == "A" else tables).to(dev)
    if kind != "A":
        # From scratch: no reason to start the joins at zero.
        nn.init.kaiming_uniform_(net.spots.join.weight, a=5 ** 0.5)
        nn.init.kaiming_uniform_(net.spots.tower_out.weight, a=5 ** 0.5)
    if kind == "C":
        nn.init.zeros_(net.v.weight); nn.init.zeros_(net.v.bias)
        net.v.weight.requires_grad_(False); net.v.bias.requires_grad_(False)
    return net


def losses(net, obs, mask, pi, z):
    logits, v = net(obs, mask)
    lp = F.log_softmax(logits, -1)
    ce = -(pi * lp.masked_fill(~mask, 0)).sum(-1).mean()
    return ce, F.mse_loss(v, z)


@torch.no_grad()
def test(net):
    ce = mse = n = 0.0
    for i in range(0, len(te[3]), 8192):
        o, m, p, z = (x[i:i + 8192].to(dev) for x in te)
        c, v = losses(net, o.float(), m.bool(), p.float(), z)
        ce += c.item() * len(z); mse += v.item() * len(z); n += len(z)
    return ce / n, mse / n


for kind in ("A", "B", "C"):
    torch.manual_seed(1)
    net = make(kind)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=3e-4, weight_decay=1e-4)
    t = time.time()
    for step in range(1, STEPS + 1):
        j = torch.randint(0, len(tr[3]), (2048,)).sort().values
        o, m, p, z = tr[0][j].to(dev).float(), tr[1][j].to(dev).bool(), tr[2][j].to(dev).float(), tr[3][j].to(dev)
        k = torch.randint(0, sym_obs.shape[0], (len(z),), device=dev)
        o, m, p = o.gather(1, sym_obs[k]), m.gather(1, sym_act[k]), p.gather(1, sym_act[k])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, mse = losses(net, o, m, p, z)
        loss = ce + mse
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 3000 == 0 or step == STEPS:
            tce, tmse = test(net)
            print(f"{time.strftime('%H:%M')} {kind} step {step}: train ce {ce.item():.3f} mse {mse.item():.3f} | "
                  f"held out (oh8) ce {tce:.3f} mse {tmse:.3f} | {time.time() - t:.0f}s", flush=True)
