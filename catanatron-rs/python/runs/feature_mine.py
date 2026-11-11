"""Which hand features explain what a board specialist knows and the
generalist doesn't?  Regresses (specialist value - generalist value) on the
candidate features from examples/feature_mine.rs, ridge with game-grouped
5-fold cross-validation, and reports out-of-fold R^2 for: baselines only,
baselines + each candidate group, everything; plus per-feature correlations.

    python runs/feature_mine.py runs/fm_*.tsv
"""
import sys
import numpy as np

rows, header = [], None
for path in sys.argv[1:]:
    with open(path) as f:
        h = f.readline().rstrip("\n").split("\t")
        header = header or h
        assert h == header, path
        for l in f:
            rows.append(l.rstrip("\n").split("\t"))
col = {c: i for i, c in enumerate(header)}
num = [c for c in header if c not in ("kind",)]
X_all = np.array([[float(r[col[c]]) for c in num] for r in rows])
kind = np.array([r[col["kind"]] for r in rows])
idx = {c: i for i, c in enumerate(num)}
target = X_all[:, idx["v_spec"]] - X_all[:, idx["v_gen"]]
groups = X_all[:, idx["board"]] * 1000 + X_all[:, idx["game"]]
print(f"{len(rows):,} positions from {len(set(X_all[:, idx['board']]))} boards; target std {target.std():.4f} "
      f"(v_gen std {X_all[:, idx['v_gen']].std():.3f}); mean |v_spec - v_gen| {np.abs(target).mean():.4f}")

BASE = ["me_vps", "opp_vps", "turn", "initial", "my_prod", "opp_prod", "my_open2", "opp_open2", "my_cards", "opp_cards",
        "my_roads_left", "my_setts_left", "v_gen"]
GROUPS = {
    "secure/contested (me)": ["my_secure", "my_secure_pips", "my_secure_top", "my_contested", "my_contested_top"],
    "secure/contested (opp)": ["opp_secure", "opp_secure_pips", "opp_secure_top", "opp_contested", "opp_contested_top"],
    "far spots 3-4 roads (me)": ["my_far3", "my_far3_top", "my_far4", "my_far4_top"],
    "far spots 3-4 roads (opp)": ["opp_far3", "opp_far3_top", "opp_far4", "opp_far4_top"],
    "expansion plan (me)": ["my_best_reach_value", "my_plan"],
    "expansion plan (opp)": ["opp_best_reach_value", "opp_plan"],
    "affordability (me)": ["my_can_road", "my_short_road", "my_can_sett", "my_short_sett", "my_can_city", "my_short_city", "my_can_dev", "my_short_dev"],
    "affordability (opp)": ["opp_can_road", "opp_can_sett", "opp_can_city", "opp_can_dev"],
    "board-wide open spots": ["open_top1", "open_top2", "open_top3", "open_top4", "open_rich", "open_count"],
}


def design(cols):
    X = X_all[:, [idx[c] for c in cols]]
    return np.hstack([X, np.ones((len(X), 1))])


def cv_r2(cols, lam=1.0, folds=5):
    X, y = design(cols), target
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
    X = X.copy(); X[:, :-1] = (X[:, :-1] - mu) / sd
    g = np.unique(groups); rng = np.random.default_rng(0); rng.shuffle(g)
    fold_of = {gg: i % folds for i, gg in enumerate(g)}
    f = np.array([fold_of[gg] for gg in groups])
    pred = np.zeros_like(y)
    for k in range(folds):
        tr, te = f != k, f == k
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1]); A[-1, -1] -= lam
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        pred[te] = X[te] @ w
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


r_base = cv_r2(BASE)
print(f"\nout-of-fold R^2 of v_spec - v_gen:\n  baselines only (what the observation already summarizes): {r_base:.3f}")
for name, cols in GROUPS.items():
    print(f"  + {name:28s}: {cv_r2(BASE + cols):.3f}  (+{cv_r2(BASE + cols) - r_base:+.3f})")
all_cols = BASE + [c for g in GROUPS.values() for c in g]
r_all = cv_r2(all_cols)
print(f"  everything: {r_all:.3f}")
for name, cols in GROUPS.items():
    rest = [c for c in all_cols if c not in cols]
    print(f"  everything minus {name:28s}: {cv_r2(rest):.3f}  (drop {r_all - cv_r2(rest):.3f})")

print("\nper-feature correlation with v_spec - v_gen (|r| > 0.05):")
for c in all_cols:
    x = X_all[:, idx[c]]
    if x.std() == 0:
        continue
    r = np.corrcoef(x, target)[0, 1]
    if abs(r) > 0.05:
        print(f"  {c:22s} r = {r:+.3f}")

print("\nby decision kind: mean |v_spec - v_gen| and count")
for k in sorted(set(kind)):
    m = kind == k
    print(f"  {k:18s} {np.abs(target[m]).mean():.4f}  n={m.sum()}")
print("\nby board: mean (v_spec - v_gen) for the mover, and its std")
for b in sorted(set(X_all[:, idx['board']])):
    m = X_all[:, idx["board"]] == b
    print(f"  {int(b)}  mean {target[m].mean():+.4f}  std {target[m].std():.4f}  n={m.sum()}")
