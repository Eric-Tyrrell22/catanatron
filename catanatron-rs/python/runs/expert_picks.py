"""What do the board specialists know about first picks that the generalist doesn't?

Builds a dataset from examples/node_features (one row per land node per board: hand
features; plus each network's first-placement policy), fits a per-node ranking model
(softmax over a board's nodes of a linear score, L2) to the specialists' first picks
with leave-one-board-out, and reports held-out accuracy, feature weights, group
ablations, and how the generalists' picks differ from the experts' on the features.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/expert_picks.py [--rebuild]
Data cache: runs/expert_picks/<board>.txt
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.getcwd()
WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
PROBE = f"{WT}/target/release/examples/node_features"
OUT = "runs/expert_picks"
os.makedirs(OUT, exist_ok=True)

# Specialists: best snapshot per board.
experts = {}
for line in open("runs/sp3/results.jsonl"):
    r = json.loads(line)
    if r.get("evals"):
        best = max(r["evals"], key=lambda e: e["win"])["iter"]
        p = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn")
        if os.path.exists(p):
            experts[r["seed"]] = p
nets = {"o3": os.path.abspath("runs/o3/o3_187.ctnn"),
        "on1": os.path.abspath("runs/on1_panel100_snap.ctnn") + "@" + os.path.abspath("runs/on1_panel100_osnap.ctnn")}

if "--rebuild" in sys.argv or not glob.glob(f"{OUT}/*.txt"):
    for board, path in experts.items():
        args = [PROBE, str(board), f"expert={path}"] + [f"{k}={v}" for k, v in nets.items()]
        with open(f"{OUT}/{board}.txt", "w") as f:
            subprocess.run(args, stdout=f, check=True)

boards = []
for fn in sorted(glob.glob(f"{OUT}/*.txt")):
    lines = open(fn).read().splitlines()
    header = None; rows = {}; policies = {}
    for l in lines:
        if l.startswith("#board"):
            continue
        if l.startswith("#policy"):
            _, label, *items = l.split()
            policies[label] = {int(a): float(b) for a, b in (it.split(":") for it in items)}
        elif l.startswith("node\t"):
            header = l.split("\t")
        else:
            v = l.split("\t"); rows[int(v[0])] = [float(x) for x in v[1:]]
    boards.append({"seed": int(os.path.basename(fn)[:-4]), "header": header[1:], "rows": rows, "policies": policies})

feat_names = boards[0]["header"]
F = len(feat_names)
# Standardise features over all nodes of all boards.
allx = np.array([x for b in boards for x in b["rows"].values()])
mu, sd = allx.mean(0), allx.std(0) + 1e-6


def board_matrix(b, names=None):
    nodes = sorted(b["rows"])
    X = (np.array([b["rows"][n] for n in nodes]) - mu) / sd
    if names is not None:
        keep = [feat_names.index(k) for k in names]
        X = X[:, keep]
    return nodes, X


def fit(train, names, l2=1.0, steps=400, lr=0.1):
    """Conditional logit: maximise the expert's probability of its top pick (soft target = its policy), L2."""
    d = len(names)
    w = np.zeros(d)
    data = []
    for b in train:
        nodes, X = board_matrix(b, names)
        t = np.array([b["policies"]["expert"].get(n, 0.0) for n in nodes]); t = t / max(t.sum(), 1e-9)
        data.append((X, t))
    for _ in range(steps):
        g = np.zeros(d)
        for X, t in data:
            z = X @ w; z -= z.max(); p = np.exp(z); p /= p.sum()
            g += X.T @ (p - t)
        g = g / len(data) + l2 * w / len(data)
        w -= lr * g
    return w


def evaluate(w, test, names):
    nodes, X = board_matrix(test, names)
    z = X @ w
    order = [nodes[i] for i in np.argsort(-z)]
    exp_pick = max(test["policies"]["expert"], key=test["policies"]["expert"].get)
    return order.index(exp_pick) + 1


groups = {
    "production": ["pips", "p_wood", "p_brick", "p_sheep", "p_wheat", "p_ore", "exp", "n_res"],
    "pairs": ["road_pair", "city_pair", "settle_set"],
    "scarcity": ["scarce", "max_share"],
    "ports": ["port_any", "port_match", "port31"],
    "numbers": ["red", "dup_num", "n_tiles", "coast"],
    "expansion": ["best2", "top2sum", "open2", "compl2"],
}
allnames = [n for g in groups.values() for n in g]


def loo(names):
    ranks = []
    for i, b in enumerate(boards):
        w = fit(boards[:i] + boards[i + 1:], names)
        ranks.append(evaluate(w, b, names))
    r = np.array(ranks)
    return (r == 1).mean(), (r <= 3).mean(), r.mean()


print(f"{len(boards)} boards, {F} features. Held-out (leave-one-board-out) prediction of the expert's first pick:")
base = loo(allnames)
print(f"  all features     : top-1 {base[0]:.0%}  top-3 {base[1]:.0%}  mean rank {base[2]:.2f}")
for g in groups:
    r = loo([n for n in allnames if n not in groups[g]])
    print(f"  without {g:11s}: top-1 {r[0]:.0%}  top-3 {r[1]:.0%}  mean rank {r[2]:.2f}   (drop {base[0]-r[0]:+.0%} top-1)")
for g in groups:
    r = loo(groups[g])
    print(f"  only {g:14s}: top-1 {r[0]:.0%}  top-3 {r[1]:.0%}  mean rank {r[2]:.2f}")
# Generalists' agreement with the expert.
for label in nets:
    agree = sum(max(b["policies"][label], key=b["policies"][label].get) == max(b["policies"]["expert"], key=b["policies"]["expert"].get) for b in boards)
    print(f"  {label} top pick = expert's on {agree}/{len(boards)} boards")
w = fit(boards, allnames)
print("\nstandardised weights (full fit):")
for name, wi in sorted(zip(allnames, w), key=lambda x: -abs(x[1])):
    print(f"  {name:11s} {wi:+.2f}")
# Where the generalist disagrees: mean standardised feature difference (expert's pick - o3's pick).
diffs = []
for b in boards:
    e = max(b["policies"]["expert"], key=b["policies"]["expert"].get); o = max(b["policies"]["o3"], key=b["policies"]["o3"].get)
    if e != o:
        diffs.append((np.array(b["rows"][e]) - np.array(b["rows"][o])) / sd)
diffs = np.array(diffs)
print(f"\non the {len(diffs)} boards where o3-187 disagrees with the expert, the expert's spot minus o3's (in standard deviations):")
for name, m in sorted(zip(feat_names, diffs.mean(0)), key=lambda x: -abs(x[1]))[:10]:
    print(f"  {name:11s} {m:+.2f} sd   (expert higher on {int((diffs[:, feat_names.index(name)] > 0).sum())}/{len(diffs)})")
