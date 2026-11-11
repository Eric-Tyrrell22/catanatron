"""Looks inside the arms' networks, no games: how each has moved from the
start, how far it is from the control, whether units died, and how it fits
one shared held-out set (the control's newest replay positions).

    python runs/arm_diff.py <start.pt> <control run> <run> [<run> ...]
"""
import json, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, ".")
import model as M

start_path, control, *others = sys.argv[1:]
arms = [control] + others
dev = torch.device("cuda")
base = torch.load(start_path, map_location="cpu")["model"]
cks = {a: torch.load(f"runs/{a}/model.pt", map_location="cpu") for a in arms}
sds = {a: cks[a]["model"] for a in arms}
meta = cks[control]["meta"]
OBS, ACT, AUX = meta["obs_size"], meta["action_size"], meta.get("aux", 0)
layers = ["trunk.0.weight", "trunk.2.weight", "trunk.4.weight", "trunk.6.weight", "pi.weight", "v.weight"]
short = ["L1", "L2", "L3", "L4", "pi", "v"]

print("games:", ", ".join(f"{a} {cks[a]['meta']['games']:,}" for a in arms))
print("\n1. weight change from the start, relative (|w - w0| / |w0|):")
print(f"{'arm':6s}" + "".join(f"{s:>8s}" for s in short))
for a in arms:
    print(f"{a:6s}" + "".join(f"{((sds[a][k] - base[k]).norm() / base[k].norm()).item():8.4f}" for k in layers))
print("\n   distance from the control, relative to the control's own change from the start (1.0 = as far from control as control is from start):")
for a in others:
    print(f"{a:6s}" + "".join(f"{((sds[a][k] - sds[control][k]).norm() / (sds[control][k] - base[k]).norm().clamp_min(1e-9)).item():8.2f}" for k in layers))
print("\n   direction: cosine between this arm's change and the control's change (1 = same direction, 0 = unrelated):")
for a in others:
    print(f"{a:6s}" + "".join(f"{F.cosine_similarity((sds[a][k] - base[k]).flatten(), (sds[control][k] - base[k]).flatten(), dim=0).item():8.2f}" for k in layers))

# Shared held-out set: the newest 50k positions of o4's replay, which none
# of the arms (nor the start) trained on.
d = torch.load("runs/o4/replay.pt", map_location="cpu", mmap=True)
total, oldest = d["z"].shape[0], int(d["oldest"])
order = torch.cat([torch.arange(oldest, total), torch.arange(0, oldest)])
idx = order[-50_000:].sort().values
obs, mask, pi, z = d["obs"][idx].float(), d["mask"][idx], d["pi"][idx].float(), d["z"][idx].float()
nets = {}
for a in arms:
    n = M.Policy(OBS, ACT, 512, AUX, 4).to(dev); n.load_state_dict(sds[a]); n.eval(); nets[a] = n
n0 = M.Policy(OBS, ACT, 512, AUX, 4).to(dev); n0.load_state_dict(base, strict=False); n0.eval(); nets["start"] = n0

@torch.no_grad()
def run(net):
    L, Vs, top, dead = [], [], [], None
    for i in range(0, len(z), 8192):
        o, m = obs[i:i + 8192].to(dev), mask[i:i + 8192].to(dev)
        logits, v = net(o, m)
        L.append(logits.cpu()); Vs.append(v.cpu()); top.append(logits.argmax(-1).cpu())
        # dead units: trunk activations never positive on this set
        h, acts = o, []
        for mod in net.trunk:
            h = mod(h)
            if isinstance(mod, torch.nn.ReLU):
                acts.append((h > 0).any(0).cpu())
        dead = acts if dead is None else [x | y for x, y in zip(dead, acts)]
    return torch.cat(L), torch.cat(Vs), torch.cat(top), [int((~x).sum()) for x in dead]

res = {a: run(nets[a]) for a in nets}
print(f"\n2. dead units per trunk layer (of 512) on the held-out set:")
for a in res:
    print(f"{a:6s} {res[a][3]}")
print(f"\n3. fit on 50k held-out positions (o4's newest; same data for every arm):")
print(f"{'arm':6s} {'policy CE':>10s} {'value MSE':>10s} {'top-1 = target':>15s}")
tgt = pi.argmax(-1)
for a in res:
    L, V, top, _ = res[a]
    ce = -(pi * F.log_softmax(L, -1).masked_fill(~mask, 0)).sum(-1).mean().item()
    print(f"{a:6s} {ce:10.4f} {F.mse_loss(V, z).item():10.4f} {(top == tgt).float().mean().item():15.1%}")
print(f"\n4. vs the control on those positions:")
print(f"{'arm':6s} {'same top move':>14s} {'mean |dv|':>10s} {'corr(v)':>8s}")
for a in res:
    if a == control: continue
    L, V, top, _ = res[a]; Lc, Vc, topc, _ = res[control]
    print(f"{a:6s} {(top == topc).float().mean().item():14.1%} {(V - Vc).abs().mean().item():10.4f} {torch.corrcoef(torch.stack([V, Vc]))[0, 1].item():8.3f}")

# 5. Effective rank per layer: how many directions of each weight matrix
# carry the signal. Two views: the number of singular values needed for 99%
# of the energy, and the entropy-based effective rank exp(H(s^2 / sum s^2)).
def ranks(w):
    s = torch.linalg.svdvals(w.float())
    e = s ** 2 / (s ** 2).sum()
    k99 = int((e.cumsum(0) < 0.99).sum()) + 1
    erank = float(torch.exp(-(e * (e + 1e-12).log()).sum()))
    return k99, erank
print(f"\n5. effective rank per layer (full rank: L1 512, L2-L4 512, pi {min(ACT, 512)}, v 1):")
print(f"{'arm':6s}" + "".join(f"{s:>14s}" for s in short[:-1]) + "   (k99 / entropy-rank)")
for a in ["start"] + arms:
    sd = base if a == "start" else sds[a]
    print(f"{a:6s}" + "".join(f"{'%d / %.0f' % ranks(sd[k]):>14s}" for k in layers[:-1]))
# Activation-side view: effective rank of the hidden activations on the held-out set
# (how many independent features each layer actually produces on real positions).
print(f"\n   activation effective rank on held-out positions (entropy-rank of the layer's outputs):")
@torch.no_grad()
def act_ranks(net):
    o = obs[:8192].to(dev); h = o; out = []
    for mod in net.trunk:
        h = mod(h)
        if isinstance(mod, torch.nn.ReLU):
            c = h - h.mean(0)
            s = torch.linalg.svdvals(c)
            e = s ** 2 / (s ** 2).sum()
            out.append(float(torch.exp(-(e * (e + 1e-12).log()).sum())))
    return out
print(f"{'arm':6s}" + "".join(f"{s:>10s}" for s in short[:4]))
for a in ["start"] + arms:
    print(f"{a:6s}" + "".join(f"{r:10.0f}" for r in act_ranks(nets[a])))
