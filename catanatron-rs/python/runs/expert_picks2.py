"""What do the 100 board specialists know about opening placements that the generalists lack?
Extended feature search over the data of examples/node_features2 (board geometry + every
network's settlement policy for placements 1-4, placements 2-4 conditioned on a network's line).

For every legal spot of a placement state this script computes the 24 hand features of
runs/expert_picks.py (checked against that script's cache) plus new feature groups (number
diversity, per-resource shares, ranks, port distance, contention, expansion geometry, a
two-settlement plan with and without a greedy opponent, interactions, and for replies the
relation to the opponent's spot and road). It fits a conditional-logit ranker (softmax over
the state's legal spots of a linear score, L2) to the expert's policy (soft target) with
leave-one-board-out and reports held-out top-1 / top-3 / mean rank with bootstrap intervals
for: the baseline, baseline + each group, forward selection over groups, and an MLP ranker
(is the gap features or function class?). Then it characterises the residual boards and the
generalists' disagreements, for the first pick and for the second seat's reply.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/expert_picks2.py [--rebuild] [--quick]
Data cache: runs/expert_picks2/<board>.txt; results: runs/expert_picks2/results.json
"""
import glob
import json
import os
import subprocess
import sys
from collections import deque

import numpy as np
import torch

torch.set_num_threads(4)
WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
PROBE = f"{WT}/target/release/examples/node_features2"
OUT = "runs/expert_picks2"
OLD = "runs/expert_picks"
os.makedirs(OUT, exist_ok=True)
QUICK = "--quick" in sys.argv
RES = ["wood", "brick", "sheep", "wheat", "ore"]
RIDX = {"WOOD": 0, "BRICK": 1, "SHEEP": 2, "WHEAT": 3, "ORE": 4}
PIP = {0: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
INF = 99


def rebuild():
    experts = {}
    for line in open("runs/sp3/results.jsonl"):
        r = json.loads(line)
        if r.get("evals"):
            best = max(r["evals"], key=lambda e: e["win"])["iter"]
            p = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn")
            if os.path.exists(p):
                experts[r["seed"]] = p
    nets = {"o3": os.path.abspath("runs/o3/o3_187.ctnn"),
            "on1": os.path.abspath("runs/on1_panel100_snap.ctnn") + "@" + os.path.abspath("runs/on1_panel100_osnap.ctnn"),
            "on1late": os.path.abspath("runs/on1/snap_002510.ctnn") + "@" + os.path.abspath("runs/on1/osnap_002510.ctnn")}
    for board, path in sorted(experts.items()):
        args = ["nice", "-n", "5", PROBE, str(board), f"expert={path}"] + [f"{k}={v}" for k, v in nets.items()]
        with open(f"{OUT}/{board}.txt", "w") as f:
            subprocess.run(args, stdout=f, check=True)


if "--rebuild" in sys.argv or not glob.glob(f"{OUT}/1*.txt"):
    rebuild()


# ----------------------------------------------------------------------------- boards
class Board:
    def __init__(self, fn):
        self.seed = int(os.path.basename(fn)[:-4])
        self.tiles = {}  # tile index -> (resource idx or None, number)
        self.nodes = {}  # node -> dict(tiles, adj, adjall, port)
        self.policy = {}  # label -> {node: prob}, first seat
        self.cond = {}  # cond -> placement -> label -> {node: prob}
        self.line = {}  # cond -> [(settlement, (a, b)), ...]
        for l in open(fn).read().splitlines():
            w = l.split()
            if w[0] == "#board":
                self.board_pips = np.array([float(x) for x in w[3].split(",")])
            elif w[0] == "#tile":
                self.tiles[int(w[1])] = (RIDX.get(w[2]), int(w[3]))
            elif w[0] == "#node":
                kv = dict(x.split("=", 1) for x in w[2:])
                self.nodes[int(w[1])] = {
                    "tiles": [int(x) for x in kv["tiles"].split(";") if x],
                    "adj": [int(x) for x in kv["adj"].split(";") if x],
                    "adjall": [int(x) for x in kv["adjall"].split(";") if x],
                    "port": kv["port"]}
            elif w[0] == "#policy":
                self.policy[w[1]] = {int(a): float(b) for a, b in (x.split(":") for x in w[2:])}
            elif w[0].startswith("#policy"):
                pl = int(w[0][7:])
                self.cond.setdefault(w[1], {}).setdefault(pl, {})[w[2]] = {int(a): float(b) for a, b in (x.split(":") for x in w[3:])}
            elif w[0] == "#line":
                items = w[2:]
                self.line[w[1]] = [(int(items[i]), tuple(int(x) for x in items[i + 1].split("-"))) for i in range(0, len(items), 2)]
        self.land = sorted(self.nodes)
        # Pips per node per resource, numbers, tile pips.
        self.P = {}
        self.nums = {}
        self.tile_pips = {}
        for n in self.land:
            p = np.zeros(5)
            nums, tp = [], []
            for ti in self.nodes[n]["tiles"]:
                r, num = self.tiles[ti]
                if r is not None:
                    p[r] += PIP[num]
                    nums.append(num)
                    tp.append(PIP[num])
                else:
                    tp.append(0)
            self.P[n] = p
            self.nums[n] = nums
            self.tile_pips[n] = tp
        self.tot = {n: self.P[n].sum() for n in self.land}
        # Road distances over land edges.
        self.dist = {n: self._bfs(n) for n in self.land}
        self.port_nodes = {n: self.nodes[n]["port"] for n in self.land if self.nodes[n]["port"] != "-"}
        self.coast = {n: len(self.nodes[n]["tiles"]) < 3 for n in self.land}
        self.desert_adj = {n: any(self.tiles[ti][0] is None for ti in self.nodes[n]["tiles"]) for n in self.land}

    def _bfs(self, src):
        d = {src: 0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.nodes[u]["adj"]:
                if v not in d:
                    d[v] = d[u] + 1
                    q.append(v)
        return d

    def d(self, a, b):
        return self.dist[a].get(b, INF)


boards = [Board(fn) for fn in sorted(glob.glob(f"{OUT}/1*.txt"))]
print(f"{len(boards)} boards")


# ----------------------------------------------------------------------------- features
def rank_of(values):
    """Dense rank, 1 = largest."""
    order = sorted(set(values), reverse=True)
    pos = {v: i + 1 for i, v in enumerate(order)}
    return [pos[v] for v in values]


def pair_stats(b, pn, pk):
    """Production of a two-settlement plan {n, k}: concave combined score, cover, settle and city floors."""
    s = pn + pk
    return np.sqrt(s).sum(), int((s > 0).sum()), s[:4].min(), min(s[3], s[4])


def spot_features(b, n, legal, own=(), opp=(), opp_road=None):
    """Features of settling `n` in a state: `legal` are the open spots (n among them), `own`/`opp`
    the observer's and the opponent's settlements, `opp_road` the opponent's last road (a, far end)."""
    nd = b.nodes[n]
    p = b.P[n]
    tot = p.sum()
    nums = b.nums[n]
    bp = b.board_pips
    f = {}
    # --- the 24 baseline features (as examples/node_features.rs; two-away over all edges to match its cache)
    f["pips"] = tot
    for r in range(5):
        f[f"p_{RES[r]}"] = p[r]
    f["exp"] = tot / 36.0
    f["n_res"] = int((p > 0).sum())
    f["road_pair"] = min(p[0], p[1])
    f["city_pair"] = min(p[4], p[3])
    f["settle_set"] = min(p[0], p[1], p[2], p[3])
    shares = p / np.maximum(bp, 1.0)
    f["scarce"] = shares.sum()
    f["max_share"] = shares.max()
    port = nd["port"]
    f["port_any"] = int(port != "-")
    f["port_match"] = int(port in RIDX and p[RIDX[port]] > 0)
    f["port31"] = int(port == "3:1")
    f["red"] = sum(1 for k in nums if k in (6, 8))
    f["dup_num"] = len(nums) - len(set(nums))
    f["n_tiles"] = len(nd["tiles"])
    one_all = set(nd["adjall"])
    two_all = set()
    for m in one_all:
        for k in b.nodes.get(m, {"adjall": []})["adjall"]:
            if k != n and k not in one_all and k in b.nodes and k in legal:
                two_all.add(k)
    two_pips = sorted(((b.tot[k], k) for k in two_all), reverse=True)
    f["best2"] = two_pips[0][0] if two_pips else 0.0
    f["top2sum"] = sum(x[0] for x in two_pips[:2])
    f["open2"] = sum(1 for x in two_pips if x[0] >= 8)
    f["compl2"] = sum(b.P[two_pips[0][1]][r] for r in range(5) if p[r] == 0) if two_pips else 0.0
    f["coast"] = int(b.coast[n])
    # --- numbers: diversity and concentration
    distinct = set(nums)
    f["hit"] = sum(PIP[k] for k in distinct)
    f["cov5689"] = len(distinct & {5, 6, 8, 9})
    f["n_nums"] = len(distinct)
    tp = b.tile_pips[n]
    f["pip_max_tile"] = max(tp) if tp else 0
    f["pip_conc"] = max(tp) / tot if tot > 0 else 0.0
    f["pip_var"] = float(np.var(tp)) if tp else 0.0
    f["red_share"] = sum(PIP[k] for k in nums if k in (6, 8)) / tot if tot > 0 else 0.0
    # --- shares per resource
    for r in range(5):
        f[f"sh_{RES[r]}"] = shares[r]
    f["sh_wb"] = shares[0] + shares[1]
    f["sh_ow"] = shares[3] + shares[4]
    f["scarce_sq"] = float((shares ** 2).sum())
    # --- ranks within the state's legal spots (non-linear relative features)
    legal_l = sorted(legal)
    rk = dict(zip(legal_l, rank_of([b.tot[k] for k in legal_l])))
    sc = {k: (b.P[k] / np.maximum(bp, 1.0)).sum() for k in legal_l}
    rks = dict(zip(legal_l, rank_of([sc[k] for k in legal_l])))
    f["rank_pips"] = rk[n]
    f["top1"] = int(rk[n] == 1)
    f["top3"] = int(rk[n] <= 3)
    f["rank_scarce"] = rks[n]
    f["pips_sq"] = tot * tot / 10.0
    f["log_pips"] = float(np.log1p(tot))
    # --- ports: distance to a matching 2:1 port, to any port
    dm, da, mval = INF, INF, 0.0
    for q, pr in b.port_nodes.items():
        d = b.d(n, q)
        da = min(da, d)
        if pr in RIDX and p[RIDX[pr]] > 0:
            dm = min(dm, d)
            mval = max(mval, p[RIDX[pr]] * (1.0 if d == 0 else 0.5 if d <= 2 else 0.0))
    f["port_dmatch"] = max(0, 4 - dm)
    f["port_dany"] = max(0, 4 - da)
    f["port_match_val"] = mval
    f["port_wb"] = int(port in ("WOOD", "BRICK") and p[RIDX[port]] > 0)
    # --- contention: what this pick kills, who else can tap these tiles
    adj_l = [k for k in nd["adj"] if k in legal]
    f["adj_max"] = max((b.tot[k] for k in adj_l), default=0.0)
    f["adj_sum"] = sum(b.tot[k] for k in adj_l)
    f["adj_top3"] = int(any(rk[k] <= 3 for k in adj_l))
    f["adj_top1"] = int(any(rk[k] == 1 for k in adj_l))
    my_tiles = set(nd["tiles"])
    rivals = [k for k in legal if k != n and k not in nd["adj"] and my_tiles & set(b.nodes[k]["tiles"])]
    f["rival_best"] = max((b.tot[k] for k in rivals), default=0.0)
    f["rivals10"] = sum(1 for k in rivals if b.tot[k] >= 10)
    within2 = [k for k in legal if b.d(n, k) <= 2]
    f["local_rank"] = rank_of([b.tot[k] for k in within2])[within2.index(n)]
    f["cluster_best"] = int(all(b.tot[n] >= b.tot[k] for k in within2))
    # --- geometry: expansion room
    d2 = [k for k in legal if b.d(n, k) == 2]
    d3 = [k for k in legal if b.d(n, k) == 3]
    f["n2"] = len(d2)
    f["sum2"] = sum(b.tot[k] for k in d2)
    f["open3"] = sum(1 for k in d3 if b.tot[k] >= 8)
    f["best3"] = max((b.tot[k] for k in d3), default=0.0)
    reach = p.copy()
    for k in d2:
        reach += b.P[k]
    f["res2"] = int((reach > 0).sum())
    f["coast_d"] = min(3, min((b.d(n, k) for k in b.land if b.coast[k]), default=3))
    f["n_edges"] = len(nd["adj"])
    f["desert_adj"] = int(b.desert_adj[n])
    # --- plan: best second settlement anywhere (no opponent in between)
    partners = [k for k in legal if k != n and k not in nd["adj"]]
    best = None
    for k in partners:
        s = pair_stats(b, p, b.P[k])
        if best is None or s[0] > best[0][0]:
            best = (s, k)
    if best:
        (s0, cov, st, ct), k = best
        f["plan_score"], f["plan_cover"], f["plan_total"] = s0, cov, tot + b.tot[k]
    else:
        f["plan_score"], f["plan_cover"], f["plan_total"] = np.sqrt(p).sum(), int((p > 0).sum()), tot
    f["plan_settle"] = max((pair_stats(b, p, b.P[k])[2] for k in partners), default=p[:4].min())
    f["plan_city"] = max((pair_stats(b, p, b.P[k])[3] for k in partners), default=min(p[3], p[4]))
    f["compl_any"] = max((sum(b.P[k][r] for r in range(5) if p[r] == 0) for k in partners), default=0.0)
    f["plan_cover5"] = max((pair_stats(b, p, b.P[k])[1] for k in partners), default=int((p > 0).sum()))
    # --- greedy: opponent takes the two best remaining spots by pips, then I take my best partner
    taken = {n} | set(nd["adj"])
    opp_picks = []
    for _ in range(2):
        cands = [k for k in legal if k not in taken]
        if not cands:
            break
        o = max(cands, key=lambda k: (b.tot[k], -k))
        opp_picks.append(o)
        taken |= {o} | set(b.nodes[o]["adj"])
    rem = [k for k in legal if k not in taken]
    gbest = None
    for k in rem:
        s = pair_stats(b, p, b.P[k])
        if gbest is None or s[0] > gbest[0][0]:
            gbest = (s, k)
    if gbest:
        (s0, cov, st, ct), k = gbest
        f["g_score"], f["g_cover"], f["g_total"] = s0, cov, tot + b.tot[k]
    else:
        f["g_score"], f["g_cover"], f["g_total"] = np.sqrt(p).sum(), int((p > 0).sum()), tot
    f["g_settle"] = max((pair_stats(b, p, b.P[k])[2] for k in rem), default=p[:4].min())
    f["g_city"] = max((pair_stats(b, p, b.P[k])[3] for k in rem), default=min(p[3], p[4]))
    f["g_loss"] = f["plan_score"] - f["g_score"]
    # --- interactions
    f["pips_x_scarce"] = tot * f["scarce"]
    f["pips_x_nres"] = tot * f["n_res"]
    f["scarce_x_nres"] = f["scarce"] * f["n_res"]
    # --- own settlement(s): complementarity with what I already hold
    if own:
        po = sum((b.P[s] for s in own), np.zeros(5))
        s0, cov, st, ct = pair_stats(b, p, po)
        f["own_score"], f["own_cover"], f["own_settle"], f["own_city"] = s0, cov, st, ct
        f["own_compl"] = sum(p[r] for r in range(5) if po[r] == 0)
        f["own_d"] = min(6, min(b.d(n, s) for s in own))
    # --- opponent: relation to the opponent's spot(s) and road
    if opp:
        a = opp[-1]
        pa = sum((b.P[s] for s in opp), np.zeros(5))
        d = min(b.d(n, s) for s in opp)
        f["opp_near"] = max(0, 5 - d)
        f["opp_d2"] = int(d == 2)
        shared = my_tiles & set().union(*(set(b.nodes[s]["tiles"]) for s in opp))
        f["opp_shared_tiles"] = len(shared)
        f["opp_shared_pips"] = sum(PIP[b.tiles[t][1]] for t in shared if b.tiles[t][0] is not None)
        far = None
        if opp_road is not None:
            far = opp_road[1] if opp_road[0] == a else opp_road[0]
        f["opp_block_road"] = int(far is not None and (n == far or n in b.nodes[far]["adj"]))
        a_two = [k for k in legal if b.d(a, k) == 2]
        a_best2 = max(a_two, key=lambda k: (b.tot[k], -k)) if a_two else None
        f["opp_block_best2"] = int(a_best2 is not None and (n == a_best2 or n in b.nodes[a_best2]["adj"]))
        f["opp_deny"] = sum(p[r] / max(bp[r], 1.0) for r in range(5) if pa[r] == 0)
        f["opp_overlap"] = float((shares * (pa / np.maximum(bp, 1.0))).sum())
        # How much my pick hurts the opponent's best two-settlement plan.
        def opp_plan(excl):
            cands = [k for k in legal if k not in excl and k not in b.nodes[a]["adj"] and k != a]
            return max((pair_stats(b, pa, b.P[k])[0] for k in cands), default=np.sqrt(pa).sum())
        f["opp_hurt"] = opp_plan(set()) - opp_plan({n} | set(nd["adj"]))
    return f


GROUPS = {
    "base": ["pips", "p_wood", "p_brick", "p_sheep", "p_wheat", "p_ore", "exp", "n_res", "road_pair", "city_pair", "settle_set",
             "scarce", "max_share", "port_any", "port_match", "port31", "red", "dup_num", "n_tiles", "best2", "top2sum", "open2", "compl2", "coast"],
    "numbers": ["hit", "cov5689", "n_nums", "pip_max_tile", "pip_conc", "pip_var", "red_share"],
    "shares": ["sh_wood", "sh_brick", "sh_sheep", "sh_wheat", "sh_ore", "sh_wb", "sh_ow", "scarce_sq"],
    "rank": ["rank_pips", "top1", "top3", "rank_scarce", "pips_sq", "log_pips"],
    "ports2": ["port_dmatch", "port_dany", "port_match_val", "port_wb"],
    "contest": ["adj_max", "adj_sum", "adj_top3", "adj_top1", "rival_best", "rivals10", "local_rank", "cluster_best"],
    "geo": ["n2", "sum2", "open3", "best3", "res2", "coast_d", "n_edges", "desert_adj"],
    "plan": ["plan_score", "plan_cover", "plan_total", "plan_settle", "plan_city", "compl_any", "plan_cover5"],
    "greedy": ["g_score", "g_cover", "g_total", "g_settle", "g_city", "g_loss"],
    "interact": ["pips_x_scarce", "pips_x_nres", "scarce_x_nres"],
    "own": ["own_score", "own_cover", "own_settle", "own_city", "own_compl", "own_d"],
    "opp": ["opp_near", "opp_d2", "opp_shared_tiles", "opp_shared_pips", "opp_block_road", "opp_block_best2", "opp_deny", "opp_overlap", "opp_hurt"],
}


def state_items(b, cond, placement):
    """(legal spots, own, opp, opp_road) for a placement of the conditioning line; 1 = empty board."""
    if placement == 1:
        return set(b.land), (), (), None
    line = b.line[cond]
    # Placements: 1 by seat A, 2 and 3 by seat B, 4 by seat A.
    placed = line[:placement - 1]
    seats = ["A", "B", "B", "A"]
    me = seats[placement - 1]
    own = tuple(s for (s, _), st in zip(placed, seats) if st == me)
    opp = tuple(s for (s, _), st in zip(placed, seats) if st != me)
    opp_roads = [r for (_, r), st in zip(placed, seats) if st != me]
    taken = set()
    for s, _ in placed:
        taken |= {s} | set(b.nodes[s]["adj"])
    legal = set(b.land) - taken
    return legal, own, opp, (opp_roads[-1] if opp_roads else None)


def build_dataset(placement, cond="expert", target="expert", require_diff_from=None):
    """One item per board: feature rows for the legal spots and the target network's policy."""
    items = []
    for b in boards:
        if placement > 1 and cond not in b.cond:
            continue
        if require_diff_from is not None and b.line[cond][0][0] == b.line[require_diff_from][0][0]:
            continue
        legal, own, opp, opp_road = state_items(b, cond, placement)
        pol = b.policy[target] if placement == 1 else b.cond[cond][placement][target]
        nodes = sorted(legal)
        assert set(pol) == set(nodes), (b.seed, placement, set(pol) ^ set(nodes))
        rows = [spot_features(b, n, legal, own, opp, opp_road) for n in nodes]
        others = {}
        for label in (b.policy if placement == 1 else b.cond[cond][placement]):
            others[label] = b.policy[label] if placement == 1 else b.cond[cond][placement][label]
        items.append({"seed": b.seed, "nodes": nodes, "rows": rows, "target": pol, "others": others, "cond": cond})
    return items


def check_baseline(items):
    """The 24 baseline features must equal runs/expert_picks.py's cache."""
    bad = 0
    tot = 0
    for it in items:
        fn = f"{OLD}/{it['seed']}.txt"
        if not os.path.exists(fn):
            continue
        header = None
        for l in open(fn):
            if l.startswith("node\t"):
                header = l.rstrip("\n").split("\t")[1:]
            elif not l.startswith("#"):
                v = l.split("\t")
                n = int(v[0])
                row = it["rows"][it["nodes"].index(n)]
                for name, x in zip(header, v[1:]):
                    tot += 1
                    if abs(row[name] - float(x)) > 0.51:  # cache prints integers rounded
                        bad += 1
    print(f"baseline check vs runs/expert_picks cache: {bad}/{tot} feature values differ by > 0.5")


# ----------------------------------------------------------------------------- fitting
def tensors(items, names):
    B = len(items)
    N = max(len(it["nodes"]) for it in items)
    X = np.zeros((B, N, len(names)), dtype=np.float32)
    T = np.zeros((B, N), dtype=np.float32)
    M = np.zeros((B, N), dtype=bool)
    for i, it in enumerate(items):
        for j, (n, row) in enumerate(zip(it["nodes"], it["rows"])):
            X[i, j] = [row[k] for k in names]
            T[i, j] = it["target"].get(n, 0.0)
            M[i, j] = True
        T[i] /= max(T[i].sum(), 1e-9)
    return X, T, M


def standardise(X, M):
    flat = X[M]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    return (X - mu) / sd * M[..., None], mu, sd


def fit_linear(X, T, M, l2=1.0, w0=None):
    """Conditional logit, soft targets, L2 (l2/2 |w|^2 per data set, like expert_picks.py)."""
    Xt, Tt, Mt = torch.tensor(X), torch.tensor(T), torch.tensor(M)
    w = torch.zeros(X.shape[-1], requires_grad=True) if w0 is None else torch.tensor(w0, requires_grad=True)
    opt = torch.optim.LBFGS([w], lr=1.0, max_iter=200, tolerance_grad=1e-7, tolerance_change=1e-10, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = Xt @ w
        z = z.masked_fill(~Mt, -1e9)
        logp = torch.log_softmax(z, dim=1)
        loss = -(Tt * logp).sum(1).mean() + l2 * (w * w).sum() / (2 * X.shape[0])
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()


def fit_mlp(X, T, M, hidden=32, wd=1e-2, steps=300, seed=0):
    torch.manual_seed(seed)
    Xt, Tt, Mt = torch.tensor(X), torch.tensor(T), torch.tensor(M)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[-1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        z = net(Xt).squeeze(-1).masked_fill(~Mt, -1e9)
        loss = -(Tt * torch.log_softmax(z, 1)).sum(1).mean()
        loss.backward()
        opt.step()
    return lambda Xq: net(torch.tensor(Xq)).squeeze(-1).detach().numpy()


def loo(items, names, model="linear", l2=1.0, **kw):
    """Leave-one-board-out: per board the rank of the expert's top pick, the model's top pick and the
    probability it gives the expert's pick."""
    X, T, M = tensors(items, names)
    X, mu, sd = standardise(X, M)
    out = []
    for i in range(len(items)):
        tr = [j for j in range(len(items)) if j != i]
        if model == "linear":
            w = fit_linear(X[tr], T[tr], M[tr], l2=l2)
            z = X[i] @ w
        else:
            f = fit_mlp(X[tr], T[tr], M[tr], **kw)
            z = f(X[i])
        z = np.where(M[i], z, -1e9)
        p = np.exp(z - z.max())
        p /= p.sum()
        nodes = items[i]["nodes"]
        exp_pick = max(items[i]["target"], key=items[i]["target"].get)
        ei = nodes.index(exp_pick)
        order = np.argsort(-z)
        rank = int(np.where(order == ei)[0][0]) + 1
        out.append({"seed": items[i]["seed"], "rank": rank, "pick": nodes[int(order[0])], "expert": exp_pick, "p_expert": float(p[ei]),
                    "top3": [nodes[int(k)] for k in order[:3]]})
    return out


def summarise(res):
    r = np.array([x["rank"] for x in res])
    lp = np.array([np.log(max(x["p_expert"], 1e-6)) for x in res])
    return {"top1": float((r == 1).mean()), "top3": float((r <= 3).mean()), "mean_rank": float(r.mean()), "nll": float(-lp.mean()), "n": len(r)}


def boot_ci(hits, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    hits = np.asarray(hits, dtype=float)
    bs = [hits[rng.integers(0, len(hits), len(hits))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def boot_diff(hits_a, hits_b, n=2000, seed=0):
    """Paired bootstrap of mean(a) - mean(b) over boards."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(hits_a, float), np.asarray(hits_b, float)
    d = a - b
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(s):
    return f"top-1 {s['top1']:.0%}  top-3 {s['top3']:.0%}  mean rank {s['mean_rank']:.2f}  nll {s['nll']:.2f}"


def names_of(groups):
    return [n for g in groups for n in GROUPS[g]]


def run_study(items, title, extra_groups, results, key):
    """Baseline, +each group, forward selection, all, MLP; records per-board results."""
    print(f"\n=== {title}: {len(items)} states ===")
    base_res = loo(items, GROUPS["base"])
    base = summarise(base_res)
    base_hits = [x["rank"] == 1 for x in base_res]
    lo, hi = boot_ci(base_hits)
    print(f"  base (24)                : {fmt(base)}   top-1 95% CI [{lo:.0%}, {hi:.0%}]")
    table = {"base": {**base, "ci": [lo, hi]}}
    gains = []
    for g in extra_groups:
        res = loo(items, names_of(["base", g]))
        s = summarise(res)
        hits = [x["rank"] == 1 for x in res]
        d, dlo, dhi = boot_diff(hits, base_hits)
        print(f"  base + {g:10s}           : {fmt(s)}   top-1 gain {d:+.0%} [{dlo:+.0%}, {dhi:+.0%}]")
        table[f"+{g}"] = {**s, "gain": d, "gain_ci": [dlo, dhi]}
        gains.append((d, -s["mean_rank"], g))
    # Forward selection over groups (on held-out top-1, then mean rank).
    chosen = ["base"]
    cur = base
    cur_res = base_res
    pool = list(extra_groups)
    while pool:
        best = None
        for g in pool:
            res = loo(items, names_of(chosen + [g]))
            s = summarise(res)
            score = (s["top1"], -s["mean_rank"])
            if best is None or score > best[0]:
                best = (score, g, s, res)
        if best[0] <= (cur["top1"], -cur["mean_rank"]):
            break
        chosen.append(best[1])
        cur, cur_res = best[2], best[3]
        pool.remove(best[1])
        print(f"  forward: + {best[1]:10s}     -> {fmt(cur)}")
    hits = [x["rank"] == 1 for x in cur_res]
    d, dlo, dhi = boot_diff(hits, base_hits)
    print(f"  forward selection {chosen}: {fmt(cur)}   gain vs base {d:+.0%} [{dlo:+.0%}, {dhi:+.0%}]")
    table["forward"] = {**cur, "groups": chosen, "gain": d, "gain_ci": [dlo, dhi]}
    all_res = loo(items, names_of(["base"] + extra_groups))
    s = summarise(all_res)
    hits = [x["rank"] == 1 for x in all_res]
    d, dlo, dhi = boot_diff(hits, base_hits)
    print(f"  base + all groups        : {fmt(s)}   gain {d:+.0%} [{dlo:+.0%}, {dhi:+.0%}]")
    table["all"] = {**s, "gain": d, "gain_ci": [dlo, dhi]}
    for l2 in (0.1, 10.0):
        s2 = summarise(loo(items, names_of(["base"] + extra_groups), l2=l2))
        print(f"  base + all, l2={l2:<5}      : {fmt(s2)}")
        table[f"all_l2_{l2}"] = s2
    # Leave-one-feature-out from the forward set: ranked list of single features by held-out gain.
    sel_names = names_of(chosen)
    lofo = []
    if not QUICK:
        for name in sel_names:
            res = loo(items, [k for k in sel_names if k != name])
            s = summarise(res)
            lofo.append((cur["top1"] - s["top1"], cur["mean_rank"] - s["mean_rank"], name))
        lofo.sort(reverse=True)
        print("  drop-one-feature from the forward set (top-1 loss, mean-rank change):")
        for d1, dr, name in lofo:
            print(f"    {name:16s} {d1:+.0%}  rank {-dr:+.2f}")
    table["lofo"] = lofo
    # MLP: function class or features?
    if not QUICK:
        for label, nm in (("base", GROUPS["base"]), ("forward", sel_names), ("all", names_of(["base"] + extra_groups))):
            res = loo(items, nm, model="mlp")
            s = summarise(res)
            print(f"  MLP(32) on {label:8s}        : {fmt(s)}")
            table[f"mlp_{label}"] = s
    # Full-data weights of the forward set.
    X, T, M = tensors(items, sel_names)
    X, mu, sd = standardise(X, M)
    w = fit_linear(X, T, M)
    table["weights"] = {k: float(v) for k, v in zip(sel_names, w)}
    print("  standardised weights (forward set, full fit):", ", ".join(f"{k} {v:+.2f}" for k, v in sorted(zip(sel_names, w), key=lambda x: -abs(x[1]))[:16]))
    # Agreement of the other networks with the expert.
    agree = {}
    for label in items[0]["others"]:
        if label == "expert":
            continue
        a = sum(max(it["others"][label], key=it["others"][label].get) == max(it["target"], key=it["target"].get) for it in items)
        mass = np.mean([it["others"][label].get(max(it["target"], key=it["target"].get), 0.0) for it in items])
        agree[label] = {"agree": a, "mass": float(mass)}
        print(f"  {label:8s} top pick = expert's on {a}/{len(items)} (mean prob on the expert's pick {mass:.3f})")
    table["agree"] = agree
    results[key] = {"table": table, "chosen": chosen, "per_board": cur_res, "base_per_board": base_res, "all_per_board": all_res}
    return chosen, cur_res, all_res


def describe(b, n):
    tiles = []
    for ti in b.nodes[n]["tiles"]:
        r, num = b.tiles[ti]
        tiles.append("desert" if r is None else f"{RES[r]}/{num}/{PIP[num]}")
    port = b.nodes[n]["port"]
    return f"node {n:2d} [{', '.join(tiles)}] {int(b.tot[n])} pips" + (f" port {port}" if port != "-" else "")


def residual(items, res, names, title, results, key):
    """What do the expert's picks on the missed boards have in common?"""
    X, T, M = tensors(items, names)
    flat = X[M]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    missed = [(it, r) for it, r in zip(items, res) if r["rank"] != 1]
    print(f"\n--- residual of {title}: {len(missed)} missed states ---")
    diffs = []
    for it, r in missed:
        e = it["nodes"].index(r["expert"])
        m = it["nodes"].index(r["pick"])
        diffs.append((X[items.index(it), e] - X[items.index(it), m]) / sd)
    diffs = np.array(diffs)
    rows = sorted(zip(names, diffs.mean(0), (diffs > 0).mean(0)), key=lambda x: -abs(x[1]))
    print("  expert's pick minus the model's pick, standardised, on missed boards (mean sd, share where expert higher):")
    for name, mn, sh in rows[:18]:
        print(f"    {name:16s} {mn:+.2f} sd   expert higher on {sh:.0%}")
    # Expert's pick rank by raw pips on the missed boards, and whether the expert's pick is the model's #2.
    r2 = sum(1 for it, r in missed if r["rank"] == 2)
    pr = [it["rows"][it["nodes"].index(r["expert"])]["rank_pips"] for it, r in missed]
    conf = [it["target"][r["expert"]] for it, r in missed]
    print(f"  expert's pick is the model's #2 on {r2}/{len(missed)}; expert pick's pips rank: median {np.median(pr):.0f}, "
          f"#1 on {sum(1 for x in pr if x == 1)}, >3 on {sum(1 for x in pr if x > 3)}; expert's confidence median {np.median(conf):.2f}")
    results[key] = {"n_missed": len(missed), "diffs": {n: [float(m), float(s)] for n, m, s in rows},
                    "missed": [{"seed": it["seed"], "expert": r["expert"], "model": r["pick"], "rank": r["rank"], "p_expert": it["target"][r["expert"]],
                                "expert_desc": describe(bmap[it["seed"]], r["expert"]), "model_desc": describe(bmap[it["seed"]], r["pick"]),
                                "others": {lb: max(pol, key=pol.get) for lb, pol in it["others"].items() if lb != "expert"}} for it, r in missed]}
    return missed


def generalist_gap(items, names, label, title):
    X, T, M = tensors(items, names)
    flat = X[M]
    sd = flat.std(0) + 1e-6
    diffs = []
    for i, it in enumerate(items):
        e = max(it["target"], key=it["target"].get)
        o = max(it["others"][label], key=it["others"][label].get)
        if e != o:
            diffs.append((X[i, it["nodes"].index(e)] - X[i, it["nodes"].index(o)]) / sd)
    diffs = np.array(diffs)
    print(f"\n--- {title}: on the {len(diffs)} states where {label} disagrees with the expert, expert's spot minus {label}'s (sd; share expert higher) ---")
    rows = sorted(zip(names, diffs.mean(0), (diffs > 0).mean(0)), key=lambda x: -abs(x[1]))
    for name, mn, sh in rows[:14]:
        print(f"    {name:16s} {mn:+.2f} sd   {sh:.0%}")
    return {n: [float(m), float(s)] for n, m, s in rows}


bmap = {b.seed: b for b in boards}
ALL_NAMES = names_of([g for g in GROUPS if g not in ("own", "opp")])


def main():
    results = {}

    # ---- placement 1: the first seat's first pick
    first = build_dataset(1)
    check_baseline(first)
    groups1 = ["numbers", "shares", "rank", "ports2", "contest", "geo", "plan", "greedy", "interact"]
    chosen1, res1, all1 = run_study(first, "placement 1 (first seat, empty board)", groups1, results, "p1")
    best_names1 = names_of(chosen1)
    missed1 = residual(first, res1, ALL_NAMES, "placement 1 forward model", results, "p1_residual")
    results["p1_gap_o3"] = generalist_gap(first, ALL_NAMES, "o3", "placement 1")
    results["p1_gap_on1"] = generalist_gap(first, ALL_NAMES, "on1", "placement 1")
    print("\n  missed boards (expert's pick vs model's pick):")
    for m in results["p1_residual"]["missed"]:
        print(f"    {m['seed']} rank {m['rank']} (expert p {m['p_expert']:.2f}): expert {m['expert_desc']}  |  model {m['model_desc']}  |  o3 {m['others'].get('o3')} on1 {m['others'].get('on1')}")

    # Does the expert's line cover the five resources more often than the generalists'?
    cov = {}
    for label in ("expert", "o3", "on1"):
        c5, settle, city, tot = [], [], [], []
        for b in boards:
            s1, s4 = b.line[label][0][0], b.line[label][3][0]
            s = b.P[s1] + b.P[s4]
            c5.append(int((s > 0).sum()))
            settle.append(s[:4].min())
            city.append(min(s[3], s[4]))
            tot.append(s.sum())
        cov[label] = {"cover_mean": float(np.mean(c5)), "cover5_share": float(np.mean(np.array(c5) == 5)), "settle_floor": float(np.mean(settle)),
                      "city_floor": float(np.mean(city)), "pips": float(np.mean(tot))}
        print(f"  {label:7s} own line (s1+s4): resources covered {np.mean(c5):.2f} (all 5 on {np.mean(np.array(c5) == 5):.0%}), settle floor {np.mean(settle):.2f}, "
              f"city floor {np.mean(city):.2f}, pips {np.mean(tot):.1f}")
    results["line_cover"] = cov

    # ---- placement 2: the second seat's reply to the expert's first pick (and to o3's where it differs)
    reply = build_dataset(2, cond="expert")
    groups2 = groups1 + ["opp"]
    chosen2, res2, all2 = run_study(reply, "placement 2 (second seat's reply to the expert's first pick + road)", groups2, results, "p2")
    missed2 = residual(reply, res2, names_of([g for g in GROUPS if g != "own"]), "placement 2 forward model", results, "p2_residual")
    results["p2_gap_o3"] = generalist_gap(reply, names_of([g for g in GROUPS if g != "own"]), "o3", "placement 2")
    results["p2_gap_on1"] = generalist_gap(reply, names_of([g for g in GROUPS if g != "own"]), "on1", "placement 2")
    print("\n  missed reply states (expert's reply vs model's):")
    for m in results["p2_residual"]["missed"]:
        print(f"    {m['seed']} rank {m['rank']} (expert p {m['p_expert']:.2f}): expert {m['expert_desc']}  |  model {m['model_desc']}  |  o3 {m['others'].get('o3')} on1 {m['others'].get('on1')}")
    reply_o3 = build_dataset(2, cond="o3", require_diff_from="expert")
    if reply_o3 and not QUICK:
        print(f"\n  robustness: reply to o3's (different) first pick, {len(reply_o3)} states, model trained on the expert-conditioned replies:")
        X, T, M = tensors(reply, names_of(chosen2))
        Xs, mu, sd = standardise(X, M)
        w = fit_linear(Xs, T, M)
        Xo, To, Mo = tensors(reply_o3, names_of(chosen2))
        Xo = (Xo - mu) / sd * Mo[..., None]
        hits = []
        for i, it in enumerate(reply_o3):
            z = np.where(Mo[i], Xo[i] @ w, -1e9)
            hits.append(it["nodes"][int(np.argmax(z))] == max(it["target"], key=it["target"].get))
        print(f"    top-1 {np.mean(hits):.0%} on {len(hits)} states")
        results["p2_o3cond_top1"] = float(np.mean(hits))

    # ---- placements 3 and 4 (own-settlement complementarity), brief
    if not QUICK:
        for pl, title in ((3, "placement 3 (second seat's second pick)"), (4, "placement 4 (first seat's second pick)")):
            items = build_dataset(pl, cond="expert")
            run_study(items, title, groups1 + ["opp", "own"], results, f"p{pl}")

    with open(f"{OUT}/results.json", "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nwrote {OUT}/results.json")


if __name__ == "__main__":
    main()
