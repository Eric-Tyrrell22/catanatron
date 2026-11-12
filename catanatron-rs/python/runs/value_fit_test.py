"""Can the opening network's value head learn forced-play first-pick win rates
from its own inputs? Fits the network to the measured forced win rates of the
100 expert boards (runs/expert_picks2/forced: ~8 first picks per board, 400
games each) with board-level cross-validation, and reports the held-out
Spearman between its value and the forced win rate per board, against the
untouched network. Two fits: the value head alone (are the trunk's features
enough, linearly?) and the whole network. A held-out gain says clean labels
would teach the value ranking (a forced-play label stream is worth building);
no gain says the inputs are missing what the ranking needs.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/value_fit_test.py <run> [--folds 5] [--epochs 300] [--lr 1e-4]
"""
import collections
import glob
import os
import re
import subprocess
import sys

import numpy as np
import torch

WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
sys.path.insert(0, WT + "/python")
from model import Policy, spot_tables_of  # noqa: E402

run = sys.argv[1]
opt = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
FOLDS, EPOCHS, LR = opt("--folds", 5), opt("--epochs", 300), opt("--lr", 1e-4)
torch.manual_seed(0)

forced = collections.defaultdict(dict)
for fn in glob.glob("runs/expert_picks2/forced/*.txt"):
    for l in open(fn):
        m = re.search(r"board (\d+) forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m:
            forced[int(m.group(1))][int(m.group(2))] = float(m.group(3)) / 100.0
boards = sorted(b for b, d in forced.items() if len(d) >= 5)

snap = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))[-1]
osnap = os.path.join(os.path.dirname(snap), "o" + os.path.basename(snap))
key = os.path.abspath(snap) + (f"@{os.path.abspath(osnap)}" if os.path.exists(osnap) else "")
ck = torch.load(f"runs/{run}/omodel.pt" if os.path.exists(osnap) else f"runs/{run}/model.pt", map_location="cpu")
meta, state = ck["meta"], {k: v for k, v in ck["model"].items() if not k.startswith("aux.")}
net = Policy(meta["obs_size"], meta["action_size"], hidden=meta["hidden"], layers=meta["layers"], aux=0,
             spots=spot_tables_of(state), extra_in=meta.get("extra_in", 0))
net.load_state_dict(state)
base_state = {k: v.clone() for k, v in net.state_dict().items()}

# Observations after each forced first pick, from the probe.
X, y, who = [], [], []
for b in boards:
    out = subprocess.run([f"{WT}/target/release/examples/forced_obs", str(b), key], capture_output=True, text=True, check=True).stdout
    rows = {}
    for l in out.splitlines():
        if l.startswith("node "):
            p = l.split()
            rows[int(p[1])] = np.array(p[2:], dtype=np.float32)
    for n, p in forced[b].items():
        if n in rows:
            X.append(rows[n]); y.append(2.0 * p - 1.0); who.append(b)
X, y, who = torch.tensor(np.stack(X)), torch.tensor(y, dtype=torch.float32), np.array(who)
assert X.shape[1] == meta["obs_size"], (X.shape, meta["obs_size"])
print(f"{len(boards)} boards, {len(y)} forced first picks, obs {X.shape[1]}, network {meta['layers']}x{meta['hidden']}", flush=True)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def per_board(values, idx):
    out = []
    for b in sorted(set(who[idx])):
        sel = idx[who[idx] == b]
        out.append(spearman(values[sel], y[sel].numpy()))
    return float(np.mean(out))


def values(model):
    """The value of every row (indexed like X, so per_board can select by global index)."""
    model.eval()
    with torch.no_grad():
        return model.body(X)[1].numpy()


def fit(head_only, train_idx):
    model = Policy(meta["obs_size"], meta["action_size"], hidden=meta["hidden"], layers=meta["layers"], aux=0,
                   spots=spot_tables_of(state), extra_in=meta.get("extra_in", 0))
    model.load_state_dict(base_state)
    params = list(model.v.parameters()) if head_only else list(model.parameters())
    optim = torch.optim.AdamW(params, lr=LR if not head_only else 1e-3, weight_decay=1e-4)
    for _ in range(EPOCHS):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx)).numpy()]
        for i in range(0, len(perm), 128):
            sel = perm[i:i + 128]
            loss = torch.nn.functional.mse_loss(model.body(X[sel])[1], y[sel])
            optim.zero_grad(); loss.backward(); optim.step()
    return model


all_idx = np.arange(len(y))
print(f"untouched network: held-out Spearman {per_board(values(net), all_idx):.3f} (all boards)", flush=True)
rng = np.random.default_rng(0)
order = rng.permutation(boards)
res = {"head": [], "full": [], "head_train": [], "full_train": []}
for f in range(FOLDS):
    test_b = set(order[f::FOLDS])
    test_idx = all_idx[[b in test_b for b in who]]
    train_idx = all_idx[[b not in test_b for b in who]]
    for name, head_only in (("head", True), ("full", False)):
        m = fit(head_only, train_idx)
        v = values(m)
        res[name].append(per_board(v, test_idx))
        res[name + "_train"].append(per_board(v, train_idx))
    print(f"fold {f + 1}/{FOLDS}: held-out Spearman head-only {res['head'][-1]:.3f} (train {res['head_train'][-1]:.3f}), "
          f"full fine-tune {res['full'][-1]:.3f} (train {res['full_train'][-1]:.3f})", flush=True)
print(f"mean held-out Spearman: head-only {np.mean(res['head']):.3f}, full fine-tune {np.mean(res['full']):.3f} "
      f"(train fits {np.mean(res['head_train']):.3f} / {np.mean(res['full_train']):.3f}); experts rank at 0.80", flush=True)
