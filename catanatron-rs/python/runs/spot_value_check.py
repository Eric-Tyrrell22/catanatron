"""Go/no-go for the spot branch: with o4 frozen, does a residual value from
the spot branch lower held-out value error more than a same-size residual
head on o4's own last layer (no board structure)?

Data: o4's replay (1M positions, oldest first); the newest 100k are held out.
Training uses random board symmetries.
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
ck = torch.load("runs/o4/model.pt", map_location="cpu")
sd, meta = ck["model"], ck["meta"]
OBS, ACT = meta["obs_size"], meta["action_size"]
AUX = sd["aux.weight"].shape[0]
tables = spots.tables_for(OBS, opponent_hands=True, opponent_devs=True, extra_info=True, spot_features=True)

data = torch.load("runs/o4/replay.pt", map_location="cpu", mmap=True)
total, oldest = data["z"].shape[0], int(data["oldest"])
order = torch.cat([torch.arange(oldest, total), torch.arange(0, oldest)])  # oldest first
HOLD = 100_000
hold_idx, train_idx = order[-HOLD:], order[:-HOLD]
print(f"{total} positions: train {len(train_idx)}, held out {HOLD}", flush=True)
obs_all, z_all = data["obs"], data["z"]
hold_obs = obs_all[hold_idx.sort().values].float()
hold_z = z_all[hold_idx.sort().values].float()

perms = catanatron_rs.board_symmetries(OBS, "BASE")
sym_obs = torch.tensor([p[0] for p in perms], device=dev)


class FlatResidual(nn.Module):
    """Control: o4's last hidden layer + the global features -> residual value."""
    def __init__(self, n_glob, width=128):
        super().__init__()
        self.register_buffer("global_feat", torch.as_tensor(tables["global_feat"]))
        self.tower = nn.Linear(512 + n_glob, width)
        self.out = nn.Linear(width, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)


def build(kind):
    net = M.Policy(OBS, ACT, hidden=512, aux=AUX, layers=4, spots=tables if kind == "spots" else None)
    net.load_state_dict(sd, strict=False)
    for p in net.parameters():
        p.requires_grad_(False)
    extra = None
    if kind == "spots":
        for n, p in net.spots.named_parameters():
            if not n.startswith("join"):  # value only: the policy join stays at zero
                p.requires_grad_(True)
    elif kind == "flat":
        extra = FlatResidual(len(tables["global_feat"])).to(dev)
    return net.to(dev), extra


def value(net, extra, obs):
    h, v = net.body(obs)
    if extra is not None:
        glob = obs[:, extra.global_feat]
        v = v + extra.out(F.relu(extra.tower(torch.cat([h, glob], -1)))).float().squeeze(-1)
    return v


@torch.no_grad()
def held_out_mse(net, extra):
    err, n = 0.0, 0
    for i in range(0, HOLD, 8192):
        o = hold_obs[i:i + 8192].to(dev)
        z = hold_z[i:i + 8192].to(dev)
        err += F.mse_loss(value(net, extra, o), z, reduction="sum").item()
        n += len(z)
    return err / n


STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
for kind in ("o4", "flat", "spots"):
    net, extra = build(kind)
    base = held_out_mse(net, extra)
    if kind == "o4":
        print(f"{time.strftime('%H:%M')} o4 alone: held-out value MSE {base:.4f}", flush=True)
        continue
    params = [p for p in net.parameters() if p.requires_grad] + (list(extra.parameters()) if extra else [])
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    for step in range(1, STEPS + 1):
        j = train_idx[torch.randint(0, len(train_idx), (2048,))]
        o = obs_all[j.sort().values].to(dev).float()
        z = z_all[j.sort().values].to(dev).float()
        k = torch.randint(0, sym_obs.shape[0], (len(z),), device=dev)
        o = o.gather(1, sym_obs[k])
        loss = F.mse_loss(value(net, extra, o), z)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == STEPS:
            print(f"{time.strftime('%H:%M')} {kind} step {step}: train {loss.item():.4f}, held out {held_out_mse(net, extra):.4f} (o4 {base:.4f})", flush=True)
