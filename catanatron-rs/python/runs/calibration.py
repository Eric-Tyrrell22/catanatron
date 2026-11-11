"""Value calibration by game phase: Ham's predicted value vs realized outcome
on o4's newest replay positions (luck-corrected results), binned by
prediction, split by opening placements / early / mid / late game (turns
left from the auxiliary target t/(t+50))."""
import os, sys
sys.path.insert(0, os.getcwd())
import torch
import model as M

N = 150_000
d = torch.load("runs/o4/replay.pt", map_location="cpu", mmap=True)
total, oldest = d["z"].shape[0], int(d["oldest"])
order = torch.cat([torch.arange(oldest, total), torch.arange(0, oldest)])[-N:]
idx = order.sort().values
ck = torch.load("runs/distill/ham.pt", map_location="cpu", weights_only=False)
m = ck["meta"]
net = M.Policy(m["obs_size"], m["action_size"], m["hidden"], m.get("aux", 0), m["layers"]).cuda(); net.load_state_dict(ck["model"]); net.eval()
pred = []
with torch.no_grad():
    for i in range(0, N, 8192):
        j = idx[i:i + 8192]
        _, v = net(d["obs"][j].cuda().float(), d["mask"][j].cuda())
        pred.append(v.float().cpu())
pred = torch.cat(pred)
z = d["z"][idx].float()
opening = d["opening"][idx].bool()
tl = d["aux"][idx, 3]  # t / (t + 50)
turns_left = 50 * tl / (1 - tl).clamp(min=1e-3)
phases = {"opening placements": opening, "early (>60 turns left)": ~opening & (turns_left > 60),
          "mid (20-60 left)": ~opening & (turns_left <= 60) & (turns_left > 20), "late (<20 left)": ~opening & (turns_left <= 20)}
edges = torch.tensor([-1.3, -0.8, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.8, 1.3])
print(f"{N:,} positions; overall MSE {((pred - z) ** 2).mean():.4f}, mean pred {pred.mean():+.3f}, mean outcome {z.mean():+.3f}")
for name, mask in phases.items():
    p, y = pred[mask], z[mask]
    print(f"\n{name}: n={mask.sum().item():,}  MSE {((p - y) ** 2).mean():.4f}  corr {torch.corrcoef(torch.stack([p, y]))[0, 1]:.3f}  "
          f"bias (mean pred - mean outcome) {p.mean() - y.mean():+.4f}")
    print("   pred bin        n   mean pred  mean outcome   gap")
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (p >= lo) & (p < hi)
        if b.sum() < 200:
            continue
        print(f"  [{lo:+.1f},{hi:+.1f})  {b.sum().item():7d}   {p[b].mean():+.3f}      {y[b].mean():+.3f}     {p[b].mean() - y[b].mean():+.3f}")
