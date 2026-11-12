"""Feature-mines the future state five turns out. From examples/midgame_horizon output (features / econ / econ_k /
obs / pos lines), for every future-state feature:

  reliab   split-half reliability of the 12-playout mean (playouts 0-5 vs 6-11, Spearman-Brown corrected to 12):
           the ceiling any predictor can reach on the target as measured
  R2 trunk held-out-by-game R^2 of a ridge fit from the network's trunk activations (strength picked by CV)
  R2 obs   the same from the raw observation
  missing  reliab - R2 trunk: predictable structure the trunk does not carry
  dR2 win  how much the feature adds to predicting the game's result (z) beyond the value head: R^2 of z on
           [value head, feature] minus R^2 of z on [value head] alone, held out by game

Sorted by `missing`; `dR2 win` says whether the missing structure matters for winning. Per-node ownership is
reported pooled. A feature the trunk predicts near its ceiling needs no auxiliary head; one with a high ceiling,
a low trunk R^2 and a positive dR2 win is a candidate target.

    PYTHONPATH=<branch>/target/pylib:<branch>/python .venv/bin/python runs/econ_probe.py <probe output> <model.pt>
"""
import sys, numpy as np, torch
sys.path.insert(0, "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python")
from model import Policy, board_head_tables_of, board_head_width_of, spot_head_tables_of, spot_head_width_of
path, ckpt = sys.argv[1], sys.argv[2]
names, econ, nodes_me, nodes_opp, per_k, obs, pos = None, {}, {}, {}, {}, {}, {}
for l in open(path):
    p = l.split()
    if not p: continue
    if p[0] == "features": names = p[1:]
    elif p[0] == "pos": pos[(int(p[1]), int(p[2]))] = (float(p[5]), float(p[6]), float(p[7]))  # truth, z, v1
    elif p[0] == "econ":
        key = (int(p[1]), int(p[2])); parts = l.split("|")
        econ[key] = np.array([float(x) for x in parts[0].split()[3:]])
        nodes_me[key] = np.array([float(x) for x in parts[1].split()]); nodes_opp[key] = np.array([float(x) for x in parts[2].split()])
    elif p[0] == "econ_k": per_k.setdefault((int(p[1]), int(p[2])), []).append(np.array([float(x) for x in p[4:]]))
    elif p[0] == "obs": obs[(int(p[1]), int(p[2]))] = np.array([float(x) for x in p[3:]], np.float32)
keys = sorted(k for k in econ if k in obs and k in pos and len(per_k.get(k, [])) >= 12)
X = np.stack([obs[k] for k in keys]); games = np.array([k[0] for k in keys])
Y = np.stack([econ[k] for k in keys]); K = np.stack([np.stack(per_k[k][:12]) for k in keys])  # [n, 12, f]
Y_me = np.stack([nodes_me[k] for k in keys]); Y_opp = np.stack([nodes_opp[k] for k in keys])
free = ~((Y_me >= 0.999) | (Y_opp >= 0.999))
Z = np.array([pos[k][1] for k in keys]); V1 = np.array([pos[k][2] for k in keys])
print(f"{len(keys)} positions from {len(set(games))} games, {len(names)} features")

ck = torch.load(ckpt, map_location="cpu"); st = ck["model"]; m = ck["meta"]
net = Policy(m["obs_size"], m["action_size"], hidden=m["hidden"], layers=m["layers"], aux=m.get("aux", 0), spot_head=spot_head_tables_of(st), spot_width=spot_head_width_of(st), board_heads=board_head_tables_of(st), board_width=board_head_width_of(st))
net.load_state_dict(st); net.eval()
with torch.no_grad():
    H, V = [], []
    for i in range(0, len(X), 256):
        h, v = net.body(torch.tensor(X[i:i + 256])); H.append(h.numpy()); V.append(v.numpy())
    H = np.concatenate(H); V = np.concatenate(V).reshape(len(X), -1)[:, 0]

ug = np.unique(games); rng = np.random.default_rng(0); rng.shuffle(ug); parts = np.array_split(ug, 5)
def ridge_cv(F, Y, lam=None, mask=None):
    """Held-out-by-game R^2 per target column (pooled over masked entries if `mask`); lam from a grid by CV."""
    F = np.concatenate([F, np.ones((len(F), 1))], 1)
    def run(lam):
        pred = np.zeros_like(Y, dtype=np.float64)
        for test_games in parts:
            te = np.isin(games, test_games); tr = ~te
            A = F[tr].T @ F[tr] + lam * np.eye(F.shape[1]); W = np.linalg.solve(A, F[tr].T @ Y[tr]); pred[te] = F[te] @ W
        if mask is None:
            return 1 - ((Y - pred) ** 2).sum(0) / np.maximum(((Y - Y.mean(0)) ** 2).sum(0), 1e-9)
        return np.array([1 - ((Y[mask] - pred[mask]) ** 2).sum() / ((Y[mask] - Y[mask].mean()) ** 2).sum()])
    grid = [lam] if lam is not None else [1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    return max((run(g) for g in grid), key=lambda r: np.nanmean(r))

Hs = (H - H.mean(0)) / (H.std(0) + 1e-6); Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
r_h, r_x = ridge_cv(Hs, Y), ridge_cv(Xs, Y)
# reliability: split halves of the playouts
a, b = K[:, :6].mean(1), K[:, 6:].mean(1)
rel = np.full(len(names), np.nan)
for j in range(len(names)):
    if a[:, j].std() > 1e-6 and b[:, j].std() > 1e-6:
        r = np.corrcoef(a[:, j], b[:, j])[0, 1]; rel[j] = 2 * r / (1 + r)
# incremental value for winning: z on [v1] vs z on [v1, feature], held out by game, plain least squares
base = ridge_cv(V1[:, None], Z[:, None], lam=1e-3)[0]
gain = np.array([ridge_cv(np.stack([V1, Y[:, j]], 1), Z[:, None], lam=1e-3)[0] - base if Y[:, j].std() > 1e-6 else np.nan for j in range(len(names))])
print(f"R^2 of the game result from the value head alone: {base:.3f}; corr(value head, vp_swing) = {np.corrcoef(V1, Y[:, names.index('vp_swing')])[0, 1]:.3f}\n")
print(f"{'feature':>16} {'mean':>7} {'sd':>6} {'reliab':>7} {'R2 trunk':>9} {'R2 obs':>7} {'missing':>8} {'dR2 win':>8}")
order = sorted(range(len(names)), key=lambda j: -(rel[j] - r_h[j]) if np.isfinite(rel[j]) else 1e9)
for j in order:
    if Y[:, j].std() < 1e-3: continue
    miss = rel[j] - r_h[j] if np.isfinite(rel[j]) else np.nan
    print(f"{names[j]:>16} {Y[:, j].mean():7.3f} {Y[:, j].std():6.3f} {rel[j]:7.3f} {r_h[j]:9.3f} {r_x[j]:7.3f} {miss:8.3f} {gain[j]:+8.4f}")
for label, Yn in (("node built by mover", Y_me), ("node built by opponent", Y_opp)):
    rh, rx = ridge_cv(Hs, Yn, mask=free), ridge_cv(Xs, Yn, mask=free)
    print(f"{label:>24}  R2 trunk {rh[0]:.3f}  R2 obs {rx[0]:.3f}   (pooled over free-node entries, base rate {Yn[free].mean():.4f})")
