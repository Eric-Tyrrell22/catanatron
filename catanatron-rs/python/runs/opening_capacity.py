"""Is the opening network's size the limit? Offline sweep on a run's saved
opening replay (runs/<run>/oreplay.pt: ~100k placement decisions with their
search targets and outcomes). The newest 20% of rows are held out (the ring's
most recent positions: generalisation to new games, which is what training
has to do), the rest train networks of several sizes from scratch with the
learner's recipe (AdamW, lr 2e-4, batch 512, bf16, policy CE + value MSE with
the per-row value weights), without the learner's symmetry augmentation.
Reports held-out policy cross-entropy, top-1 agreement with the search
target's argmax, value MSE, and the train loss, per size and epoch budget.
Reading: a bigger net clearly better on held-out = capacity binds; all sizes
meeting on held-out while train loss keeps falling with size = the data or
inputs limit, not the size.

    .venv/bin/python runs/opening_capacity.py <run> [--epochs 8,24] [--sizes 2x256,4x512,6x1024,4x2048]
"""
import sys
import time

import torch

sys.path.insert(0, "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python")
from model import Policy  # noqa: E402

run = sys.argv[1]
opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
EPOCHS = [int(x) for x in opt("--epochs", "8,24").split(",")]
SIZES = [tuple(int(v) for v in s.split("x")) for s in opt("--sizes", "2x256,4x512,6x1024,4x2048").split(",")]
dev = torch.device("cuda")
torch.manual_seed(0)

d = torch.load(f"runs/{run}/oreplay.pt", map_location="cpu", mmap=True)
n, oldest = d["z"].shape[0], d["oldest"]
order = torch.cat([torch.arange(oldest, n), torch.arange(0, oldest)])  # oldest first
cut = int(0.8 * n)
tr, te = order[:cut], order[cut:]
X = d["obs"].to(dev)  # f16, 100k x obs
M, PI, Z, VW = d["mask"].to(dev), d["pi"].to(dev), d["z"].to(dev), d["vw"].to(dev).float()
obs_size, action_size = X.shape[1], M.shape[1]
print(f"{run}: {n:,} opening rows, obs {obs_size}, train {len(tr):,} (oldest) / held-out {len(te):,} (newest)", flush=True)
MASKED = -1e9


def losses(model, idx, bs=4096):
    model.eval()
    ce = mse = top1 = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs].to(dev)
            logits, v = model(X[j].float(), M[j])
            logp = torch.log_softmax(logits.float(), -1)
            pi = PI[j].float()
            ce += -(pi * logp).sum(-1).sum().item()
            top1 += (logp.argmax(-1) == pi.argmax(-1)).sum().item()
            mse += ((v.float() - Z[j]) ** 2 * VW[j]).sum().item()
    return ce / len(idx), top1 / len(idx), mse / VW[idx.to(dev)].sum().item()


for layers, hidden in SIZES:
    model = Policy(obs_size, action_size, hidden=hidden, layers=layers).to(dev)
    params = sum(p.numel() for p in model.parameters())
    optim = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    t0, done = time.time(), 0
    for target in EPOCHS:
        while done < target:
            model.train()
            perm = tr[torch.randperm(len(tr))]
            for i in range(0, len(perm), 512):
                j = perm[i:i + 512].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, v = model(X[j].float(), M[j])
                logp = torch.log_softmax(logits.float(), -1)
                l_pi = -(PI[j].float() * logp).sum(-1).mean()
                l_v = ((v.float() - Z[j]) ** 2 * VW[j]).sum() / VW[j].sum().clamp(min=1.0)
                optim.zero_grad()
                (l_pi + 0.5 * l_v).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            done += 1
        tce, ttop, tmse = losses(model, tr[-len(te):])  # the newest part of the training set, same size as held-out
        hce, htop, hmse = losses(model, te)
        print(f"{layers}x{hidden} ({params / 1e6:.1f}M) after {done:2d} epochs: held-out CE {hce:.4f} top1 {100 * htop:.1f}% vMSE {hmse:.4f} | "
              f"train CE {tce:.4f} top1 {100 * ttop:.1f}% vMSE {tmse:.4f} | {time.time() - t0:.0f}s", flush=True)
