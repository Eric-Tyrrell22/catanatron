"""Spot-net supervised test: can a per-spot network with shared weights learn
first picks from the oracle labels where the flat MLP memorised them?

Data: spot structure of every board from examples/spot_dump (tiles, port,
neighbours at road distance 1 and 2; the Base map's node graph is fixed, so
node ids are positional and shared by all boards) and the oracle labels (win
rate of the first seat after a forced first pick, 8 pips-best nodes per board,
48 network-only games each). Expert boards (1030001-1030100) are evaluation
only: agreement of the argmax with the board specialist's first pick.

Models, all on the same per-node hand features plus board context:
  spot    one MLP scores every node from its own features, then message passing
          over the node graph (distance 1 and 2) for R rounds; shared weights.
  perspot the same without message passing (R = 0).
  flat    one MLP over the 54 node feature blocks concatenated (positional
          weights, like the opening network today).
Target: softmax(win rate / tau) over the labelled nodes; loss = cross-entropy
over all 54 nodes. Reported every few epochs: held-out top-1 against the
oracle's best pick (label-noise ceiling ~68%), held-out Spearman of the scores
vs the win rates over the 8 labelled nodes, train top-1, expert agreement
(/100) and expert pick within the model's top 3.

    .venv/bin/python runs/spot_fit.py <spots.txt> <picks dir> [<picks dir> ...] [--model spot|perspot|flat] [--rounds 2]
        [--raw] [--relation relation.txt] [--hidden 128] [--epochs 40] [--lr 1e-3] [--tau 0.05] [--boards N] [--seed 0] [--holdout 0.1]
"""
import collections
import glob
import os
import re
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

args = sys.argv[1:]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
MODEL = opt("--model", "spot"); ROUNDS = opt("--rounds", 2); HIDDEN = opt("--hidden", 128)
EPOCHS = opt("--epochs", 40); LR = opt("--lr", 1e-3); TAU = opt("--tau", 0.05); NBOARDS = opt("--boards", 0)
SEED = opt("--seed", 0); HOLDOUT = opt("--holdout", 0.1); RAW = "--raw" in args; RELATION = opt("--relation", "")
positional = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--") or args[i - 1] == "--raw")]
spots_path, pick_dirs = positional[0], positional[1:]
dev = torch.device("cuda")
torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
PROB = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
N = 54

# ---- boards
boards = {}
with open(spots_path) as f:
    cur = None
    for l in f:
        p = l.split()
        if p[0] == "#board":
            cur = {"nodes": {}}; boards[int(p[1])] = cur
            continue
        n = int(p[0])
        tiles = [tuple(int(x) for x in t.split("/")) for t in p[1].split(";")]
        port = None if p[2] == "-" else int(p[2])
        d1 = [int(x) for x in p[3].split(",") if x]
        d2 = [int(x) for x in p[4].split(",") if x]
        cur["nodes"][n] = (tiles, port, d1, d2)
relation = {}
if RELATION:
    # Per node, the opening-relation block's 9 board constants (examples/relation_dump.rs): plan, greedy opponent plan,
    # greedy loss, plan cover, greedy cover, pips x scarcity, hit, pip variance, 5-6-8-9 cover.
    with open(RELATION) as f:
        for l in f:
            p = l.split()
            if p[0] == "#board":
                cur = np.zeros((N, 9), np.float32); relation[int(p[1])] = cur
            else:
                cur[int(p[0])] = [float(x) for x in p[1:]]
print(f"boards with spot structure: {len(boards)}" + (f", with relation constants: {len(relation)}" if RELATION else ""), flush=True)

# Node graph (fixed across boards): normalised adjacency at distance 1 and 2.
any_board = next(iter(boards.values()))
A1 = np.zeros((N, N), np.float32); A2 = np.zeros((N, N), np.float32)
for n, (_, _, d1, d2) in any_board["nodes"].items():
    for m in d1: A1[n, m] = 1
    for m in d2: A2[n, m] = 1
A1 /= A1.sum(1, keepdims=True).clip(1); A2 /= A2.sum(1, keepdims=True).clip(1)
A1, A2 = torch.tensor(A1, device=dev), torch.tensor(A2, device=dev)


def features(board, seed=None):
    """[54, F] per-node features with board context appended; node order = id."""
    nodes = board["nodes"]
    rel = relation[seed] if RELATION else np.zeros((N, 0), np.float32)
    pips = np.zeros((N, 5), np.float32); nums = [[] for _ in range(N)]
    for n, (tiles, _, _, _) in nodes.items():
        for res, num in tiles:
            if res < 5 and num:
                pips[n, res] += PROB[num]; nums[n].append(num)
    board_pips = pips.sum(0)
    tot = pips.sum(1)
    order = np.argsort(-tot); rank = np.empty(N, np.float32); rank[order] = np.arange(N) / N
    rows = []
    for n in range(N):
        tiles, port, d1, d2 = nodes[n]
        p = pips[n]
        red = sum(1 for k in nums[n] if k in (6, 8)); dup = len(nums[n]) - len(set(nums[n]))
        exp = sum(PROB[k] for k in nums[n]) / 36
        port_oh = np.zeros(7, np.float32)
        if port is None: port_oh[0] = 1
        elif port == 3: port_oh[1] = 1
        else: port_oh[2 + port] = 1
        port_match = float(port is not None and port != 3 and p[port] > 0)
        share = p / board_pips.clip(1)
        two = sorted((tot[k] for k in d2), reverse=True)
        best2 = two[0] if two else 0; top2 = sum(two[:2]); open2 = sum(1 for t in two if t >= 8)
        kbest = max(d2, key=lambda k: tot[k]) if d2 else None
        compl2 = float(sum(pips[kbest][r] for r in range(5) if p[r] == 0)) if kbest is not None else 0.0
        raw = np.zeros((3, 18), np.float32)  # per tile, best pips first: resource one-hot (6) + number one-hot (none, 2..12)
        if RAW:
            for i, (res, num) in enumerate(sorted(tiles, key=lambda t: -PROB.get(t[1], 0))[:3]):
                raw[i, res] = 1; raw[i, 6 + (num - 1 if num else 0)] = 1
        rows.append(np.concatenate([
            rel[n],
            raw.ravel() if RAW else np.zeros(0, np.float32),
            p / 15, [tot[n] / 15, len(tiles) / 3, float(len(tiles) < 3), (p > 0).sum() / 5, red / 2, float(dup > 0), exp, rank[n]],
            port_oh, [port_match, min(p[0], p[1]) / 5, min(p[3], p[4]) / 5, min(p[0], p[1], p[2], p[3]) / 5, share.sum(), share.max()],
            [best2 / 15, top2 / 30, open2 / 6, compl2 / 15],
            board_pips / board_pips.sum(), [board_pips.sum() / 100],
        ]).astype(np.float32))
    return np.stack(rows)


# ---- labels
labels = collections.defaultdict(dict)
for d in pick_dirs:
    for fn in glob.glob(os.path.join(d, "*.txt")):
        for l in open(fn):
            m = re.search(r"board (\d+) forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
            if m:
                labels[int(m.group(1))][int(m.group(2))] = float(m.group(3)) / 100
experts = {}
for fn in glob.glob("runs/expert_picks/*.txt"):
    for l in open(fn):
        if l.startswith("#policy expert"):
            items = l.split()[2:]
            experts[int(os.path.basename(fn)[:-4])] = int(max(items, key=lambda it: float(it.split(":")[1])).split(":")[0])
labelled = sorted(b for b in labels if b in boards and len(labels[b]) >= 5 and b not in experts)
rng.shuffle(labelled)
if NBOARDS:
    labelled = labelled[:NBOARDS]
n_hold = max(1, int(HOLDOUT * len(labelled)))
hold, train = labelled[:n_hold], labelled[n_hold:]
expert_boards = sorted(b for b in experts if b in boards)
print(f"labelled boards {len(labelled)} (train {len(train)}, held-out {len(hold)}); expert boards {len(expert_boards)}", flush=True)


def tensors(ids):
    X = torch.tensor(np.stack([features(boards[b], b) for b in ids]), device=dev)
    T = torch.zeros(len(ids), N, device=dev); W = torch.full((len(ids), N), float("nan"), device=dev)
    for i, b in enumerate(ids):
        wr = labels.get(b, {})
        if wr:
            nodes = list(wr); w = torch.tensor([wr[n] for n in nodes]) / TAU
            T[i, nodes] = torch.softmax(w, 0).to(dev)
            W[i, nodes] = torch.tensor([wr[n] for n in nodes], device=dev)
    return X, T, W


Xtr, Ttr, Wtr = tensors(train); Xho, Tho, Who = tensors(hold); Xex, _, _ = tensors(expert_boards)
FEAT = Xtr.shape[-1]


class SpotNet(nn.Module):
    def __init__(self, hidden, rounds):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.rounds = nn.ModuleList(nn.Sequential(nn.Linear(3 * hidden, hidden), nn.ReLU()) for _ in range(rounds))
        self.score = nn.Linear(hidden, 1)

    def forward(self, x):  # x [B, 54, F]
        h = self.enc(x)
        for r in self.rounds:
            h = h + r(torch.cat([h, A1 @ h, A2 @ h], -1))
        return self.score(h).squeeze(-1)


class FlatNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N * FEAT, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, N))

    def forward(self, x):
        return self.net(x.flatten(1))


model = (FlatNet(512 if HIDDEN == 128 else HIDDEN) if MODEL == "flat" else SpotNet(HIDDEN, 0 if MODEL == "perspot" else ROUNDS)).to(dev)
print(f"model {MODEL}: {sum(p.numel() for p in model.parameters()):,} parameters, {FEAT} features per node", flush=True)
optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)


def spearman_rows(scores, W):
    out = []
    for s, w in zip(scores, W):
        m = ~torch.isnan(w)
        if m.sum() >= 5:
            a = s[m].argsort().argsort().float(); b = w[m].argsort().argsort().float()
            out.append(torch.corrcoef(torch.stack([a, b]))[0, 1].item())
    return float(np.mean(out)) if out else float("nan")


@torch.no_grad()
def evaluate(tag):
    model.eval()
    def top1(X, T):
        s = model(X); return (s.argmax(1) == T.argmax(1)).float().mean().item(), s
    tr_acc, _ = top1(Xtr[:len(hold)], Ttr[:len(hold)])
    ho_acc, s_ho = top1(Xho, Tho)
    sp = spearman_rows(s_ho, Who)
    s_ex = model(Xex)
    picks = torch.tensor([experts[b] for b in expert_boards], device=dev)
    agree = (s_ex.argmax(1) == picks).sum().item()
    top3 = (s_ex.topk(3, 1).indices == picks[:, None]).any(1).sum().item()
    print(f"{tag:>9}: train top-1 {100 * tr_acc:3.0f}%  held-out top-1 {100 * ho_acc:3.0f}%  held-out Spearman {sp:.3f}  | experts agree {agree}/{len(expert_boards)}  in top-3 {top3}", flush=True)
    model.train()


evaluate("untouched")
B = 128
for ep in range(1, EPOCHS + 1):
    perm = torch.randperm(len(train), device=dev)
    for i in range(0, len(perm), B):
        j = perm[i:i + B]
        logits = model(Xtr[j])
        loss = -(Ttr[j] * F.log_softmax(logits, -1)).sum(-1).mean()
        optim.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
    if ep % max(1, EPOCHS // 8) == 0 or ep == EPOCHS:
        evaluate(f"epoch {ep}"); print(f"{'':>9}  train loss {loss.item():.3f}", flush=True)
