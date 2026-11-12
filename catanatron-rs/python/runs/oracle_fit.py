"""Oracle-label pilot: can the opening network learn first picks from cheap
network-only playout labels? Data from the pilot directory (made by
forced_pick top:8 + forced_obs per random board): per board the pre-pick
observation, the legal first picks' policy slots, the post-pick observations,
and the oracle win rate of each of the 8 pips-best picks. Fine-tunes a run's
opening network on them: policy target = softmax(win rate / tau) over the 8
labelled picks at the pre-pick position, value target = 2 x win rate - 1 at
the post-pick position. Evaluates, before and after, on the 100 expert boards
(never trained on): first-pick agreement with the expert, the value head's
Spearman against the experts' forced win rates (runs/expert_picks2/forced),
and the one-ply value pick's regret; and on held-out pilot boards: top-1
against the oracle's best pick and Spearman against the oracle.

    .venv/bin/python runs/oracle_fit.py <run> <pilot dir> [--epochs 6] [--lr 1e-4] [--tau 0.05] [--holdout 0.1]
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python")
from model import Policy, spot_tables_of  # noqa: E402

run, pilot = sys.argv[1], sys.argv[2]
opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
EPOCHS, LR, TAU, HOLDOUT = opt("--epochs", 6), opt("--lr", 1e-4), opt("--tau", 0.05), opt("--holdout", 0.1)
dev = torch.device("cuda")
torch.manual_seed(0)
rng = np.random.default_rng(0)


def read_obs(path):
    """(pre obs, {node: slot}, {node: post obs}) from a forced_obs dump (plain or gzipped)."""
    import gzip
    pre, slots, post = None, {}, {}
    opener = gzip.open if path.endswith(".gz") else open
    for l in opener(path, "rt"):
        p = l.split()
        if p[0] == "#pre":
            pre = np.array(p[1:], dtype=np.float32)
        elif p[0] == "#slot":
            slots[int(p[1])] = int(p[2])
        elif p[0] == "node":
            post[int(p[1])] = np.array(p[2:], dtype=np.float32)
    return pre, slots, post


def read_picks(path):
    out = {}
    for l in open(path):
        m = re.search(r"forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m:
            out[int(m.group(1))] = float(m.group(2)) / 100.0
    return out


# ---- pilot boards
boards = []
for fn in sorted(glob.glob(f"{pilot}/picks/*.txt")):
    b = int(os.path.basename(fn)[:-4])
    wr = read_picks(fn)
    obs_path = next((q for q in (f"{pilot}/obs/{b}.txt", f"{pilot}/obs/{b}.txt.gz") if os.path.exists(q)), None)
    if len(wr) < 5 or obs_path is None:
        continue
    pre, slots, post = read_obs(obs_path)
    if pre is None or any(n not in post for n in wr):
        continue
    boards.append((b, pre, slots, {n: post[n] for n in wr}, wr))
print(f"pilot boards with labels and observations: {len(boards)}", flush=True)
rng.shuffle(boards)
n_hold = int(HOLDOUT * len(boards))
hold, train = boards[:n_hold], boards[n_hold:]

# ---- expert boards (evaluation only)
forced = collections.defaultdict(dict)
for fn in glob.glob("runs/expert_picks2/forced/*.txt"):
    for l in open(fn):
        m = re.search(r"board (\d+) forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m:
            forced[int(m.group(1))][int(m.group(2))] = float(m.group(3)) / 100.0
experts = []
for fn in sorted(glob.glob("runs/expert_picks/*.txt")):
    b = int(os.path.basename(fn)[:-4])
    pick = None
    for l in open(fn):
        if l.startswith("#policy expert"):
            items = l.split()[2:]
            pick = int(max(items, key=lambda it: float(it.split(":")[1])).split(":")[0])
    if pick is not None and os.path.exists(f"{pilot}/expert_obs/{b}.txt"):
        pre, slots, post = read_obs(f"{pilot}/expert_obs/{b}.txt")
        experts.append((b, pre, slots, post, pick, forced.get(b, {})))
print(f"expert boards for evaluation: {len(experts)}", flush=True)

ck = torch.load(f"runs/{run}/omodel.pt", map_location="cpu")
meta, state = ck["meta"], {k: v for k, v in ck["model"].items() if not k.startswith("aux.")}
net = Policy(meta["obs_size"], meta["action_size"], hidden=meta["hidden"], layers=meta["layers"], spots=spot_tables_of(state), extra_in=meta.get("extra_in", 0)).to(dev)
net.load_state_dict(state)
A = meta["action_size"]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1]) if len(x) > 2 else float("nan")


def tensors(rows):
    """Pre-pick batch (obs, mask, policy target) and post-pick batch (obs, value target) for labelled boards."""
    pre, mask, tgt, post, z = [], [], [], [], []
    for _, p, slots, po, wr in rows:
        m = np.zeros(A, bool)
        for s in slots.values():
            m[s] = True
        t = np.zeros(A, np.float32)
        nodes = list(wr)
        w = np.array([wr[n] for n in nodes]) / TAU
        w = np.exp(w - w.max()); w /= w.sum()
        for n, pw in zip(nodes, w):
            t[slots[n]] = pw
        pre.append(p); mask.append(m); tgt.append(t)
        for n in nodes:
            post.append(po[n]); z.append(2 * wr[n] - 1)
    f = lambda a, dt: torch.tensor(np.stack(a), dtype=dt, device=dev)
    return f(pre, torch.float32), f(mask, torch.bool), f(tgt, torch.float32), f(post, torch.float32), torch.tensor(z, dtype=torch.float32, device=dev)


def evaluate(model, label):
    model.eval()
    with torch.no_grad():
        # Expert boards: policy argmax among legal first picks vs the expert; value Spearman vs forced rates; value-pick regret.
        agree, corr, regret = 0, [], []
        for b, pre, slots, post, pick, fw in experts:
            m = torch.zeros(A, dtype=torch.bool, device=dev)
            for s in slots.values():
                m[s] = True
            logits, _ = model(torch.tensor(pre, device=dev)[None], m[None])
            best_slot = int(logits[0].argmax())
            agree += slots.get(pick) == best_slot
            if len(fw) >= 5:
                nodes = [n for n in fw if n in post]
                vals = model(torch.tensor(np.stack([post[n] for n in nodes]), device=dev), m[None].expand(len(nodes), A))[1].cpu().numpy()
                corr.append(spearman(vals, [fw[n] for n in nodes]))
                regret.append(100 * (max(fw.values()) - fw[nodes[int(vals.argmax())]]))
        # Held-out pilot boards (and a same-sized slice of the training boards): policy top-1 vs the oracle's best pick;
        # value Spearman vs the oracle.
        ttop = 0
        for b, pre, slots, post, wr in train[:len(hold)]:
            m = torch.zeros(A, dtype=torch.bool, device=dev)
            for s in slots.values():
                m[s] = True
            logits, _ = model(torch.tensor(pre, device=dev)[None], m[None])
            slot_node = {s: n for n, s in slots.items()}
            ttop += slot_node[int(logits[0].argmax())] == max(wr, key=wr.get)
        top, oc = 0, []
        for b, pre, slots, post, wr in hold:
            m = torch.zeros(A, dtype=torch.bool, device=dev)
            for s in slots.values():
                m[s] = True
            logits, _ = model(torch.tensor(pre, device=dev)[None], m[None])
            slot_node = {s: n for n, s in slots.items()}
            top += slot_node[int(logits[0].argmax())] == max(wr, key=wr.get)
            nodes = list(wr)
            vals = model(torch.tensor(np.stack([post[n] for n in nodes]), device=dev), m[None].expand(len(nodes), A))[1].cpu().numpy()
            oc.append(spearman(vals, [wr[n] for n in nodes]))
    print(f"{label:>14}: experts agree {agree}/{len(experts)}  value Spearman {np.mean(corr):.3f}  value-pick regret {np.mean(regret):.1f} pts | "
          f"pilot top-1 train {100 * ttop / max(1, len(hold)):.0f}% / held-out {100 * top / max(1, len(hold)):.0f}%  held-out Spearman {np.mean(oc):.3f}", flush=True)


evaluate(net, "untouched")
pre, mask, tgt, post, z = tensors(train)
optim = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
for ep in range(1, EPOCHS + 1):
    net.train()
    perm = torch.randperm(len(pre), device=dev)
    for i in range(0, len(perm), 64):
        j = perm[i:i + 64]
        logits, _ = net(pre[j], mask[j])
        l_pi = -(tgt[j] * torch.log_softmax(logits, -1)).sum(-1).mean()
        k = torch.randint(0, len(post), (256,), device=dev)
        _, v = net(post[k], mask[j[:1]].expand(256, A))
        l_v = ((v - z[k]) ** 2).mean()
        optim.zero_grad(); (l_pi + 0.5 * l_v).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); optim.step()
    if ep % max(1, EPOCHS // 10) == 0:
        evaluate(net, f"epoch {ep}")
