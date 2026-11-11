"""Does a much larger network fit the existing search targets better?
Trains several sizes from scratch, supervised, on o4's replay (oldest 950k
positions) and scores them on the same 50k held-out positions arm_diff uses
(o4's newest), with the train/held-out gap.

    python runs/fit_upward.py "4x512" "6x768" "8x1024" [--steps N]
"""
import os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import torch.nn.functional as F
import catanatron_rs
import model as M

sizes = [a for a in sys.argv[1:] if not a.startswith("--")]
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 15000
dev = torch.device("cuda")
OBS = 2202
d = torch.load("runs/o4/replay.pt", map_location="cpu", mmap=True)
total, oldest = d["z"].shape[0], int(d["oldest"])
order = torch.cat([torch.arange(oldest, total), torch.arange(0, oldest)])
hold = order[-50_000:].sort().values
train = order[:-50_000]
ACT = d["mask"].shape[1]
perms = catanatron_rs.board_symmetries(OBS, "BASE")
sym_obs = torch.tensor([p[0] for p in perms], device=dev); sym_act = torch.tensor([p[1] for p in perms], device=dev)


def batch(idx, sym):
    o, m, p, z = d["obs"][idx].to(dev).float(), d["mask"][idx].to(dev), d["pi"][idx].to(dev).float(), d["z"][idx].to(dev).float()
    if sym:
        k = torch.randint(0, sym_obs.shape[0], (len(idx),), device=dev)
        o, m, p = o.gather(1, sym_obs[k]), m.gather(1, sym_act[k]), p.gather(1, sym_act[k])
    return o, m, p, z


def losses(net, o, m, p, z):
    logits, v = net(o, m)
    ce = -(p * F.log_softmax(logits, -1).masked_fill(~m, 0)).sum(-1).mean()
    return ce, F.mse_loss(v, z), (logits.argmax(-1) == p.argmax(-1)).float().mean()


@torch.no_grad()
def evaluate(net, idx):
    acc = [0.0, 0.0, 0.0]
    for i in range(0, len(idx), 8192):
        j = idx[i:i + 8192]
        ce, mse, top = losses(net, *batch(j, sym=False))
        for k, x in enumerate((ce, mse, top)):
            acc[k] += x.item() * len(j)
    return [a / len(idx) for a in acc]


train_probe = train[torch.randperm(len(train))[:50_000]].sort().values  # fixed train subset for the gap
print(f"train {len(train):,} / held-out {len(hold):,} positions; {STEPS} steps x 2048, lr 3e-4 cosine, 12 symmetries", flush=True)
for size in sizes:
    layers, hidden = (int(x) for x in size.split("x"))
    torch.manual_seed(1)
    net = M.Policy(OBS, ACT, hidden=hidden, layers=layers).to(dev)
    params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    t = time.time()
    for step in range(1, STEPS + 1):
        idx = train[torch.randint(0, len(train), (2048,))].sort().values
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ce, mse, _ = losses(net, *batch(idx, sym=True))
        loss = ce + mse
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 5000 == 0 or step == STEPS:
            net.eval(); h = evaluate(net, hold); tr = evaluate(net, train_probe); net.train()
            print(f"{time.strftime('%H:%M')} {size} ({params/1e6:.1f}M params) step {step}: held-out CE {h[0]:.4f} MSE {h[1]:.4f} top-1 {h[2]:.1%} | "
                  f"train CE {tr[0]:.4f} MSE {tr[1]:.4f} | gap CE {h[0]-tr[0]:+.4f} MSE {h[1]-tr[1]:+.4f} | {time.time()-t:.0f}s", flush=True)
    torch.save({"model": net.state_dict(), "meta": {"obs_size": OBS, "action_size": ACT, "hidden": hidden, "layers": layers, "aux": 0}}, f"runs/distill/up_{size}.pt")
print("done", flush=True)
