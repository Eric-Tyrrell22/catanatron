"""What does a board specialist value in an opening spot that the generalist
doesn't?  Per legal spot at each initial settlement placement, the target is
the specialist's preference shift: log(p_spec) - log(p_gen), centred within
the decision (so only relative preferences count), regressed on precomputed
spot properties with ridge, 5-fold cross-validation grouped by game.

    python runs/opening_mine.py runs/om_*.tsv
"""
import sys
import numpy as np

rows, header = [], None
for path in sys.argv[1:]:
    with open(path) as f:
        h = f.readline().rstrip("\n").split("\t")
        header = header or h
        assert h == header, path
        rows += [l.rstrip("\n").split("\t") for l in f]
X = np.array([[float(x) for x in r] for r in rows])
idx = {c: i for i, c in enumerate(header)}
c = lambda n: X[:, idx[n]]
decision = c("board") * 1e9 + c("game") * 1e3 + c("placement") * 10 + c("seat_order")
shift = np.log(np.maximum(c("p_spec"), 1e-6)) - np.log(np.maximum(c("p_gen"), 1e-6))
# Centre within each decision, weight each spot by the generalist's prior mass + a floor
# (spots nobody considers tell us little).
target = shift.copy()
for d in np.unique(decision):
    m = decision == d
    target[m] -= shift[m].mean()
w = c("p_gen") + c("p_spec") + 0.02
print(f"{len(rows):,} spot rows in {len(np.unique(decision)):,} placement decisions on {len(np.unique(c('board')))} boards; "
      f"target std {target.std():.3f}")
# Where do the two priors put their mass?
top_gen = {}; top_spec = {}
agree = 0; n_dec = 0
for d in np.unique(decision):
    m = np.where(decision == d)[0]
    g, s = m[np.argmax(c("p_gen")[m])], m[np.argmax(c("p_spec")[m])]
    agree += g == s; n_dec += 1
print(f"same top spot in {agree / n_dec:.1%} of decisions")

FEATS = ["pips", "pips_wood", "pips_brick", "pips_sheep", "pips_wheat", "pips_ore", "n_resources", "n_tiles", "n_numbers",
         "has_6_8", "n_6_8", "has_2_12", "scarcity_pips", "ore_wheat", "wood_brick", "port_any", "port_match", "port_3to1",
         "expand_best", "expand_count", "expand_second", "own_combined_resources", "own_new_resources", "own_combined_pips", "own_dist",
         "opp_dist", "opp_contest"]
GROUPS = {
    "production (pips, per resource)": ["pips", "pips_wood", "pips_brick", "pips_sheep", "pips_wheat", "pips_ore"],
    "diversity / combos": ["n_resources", "n_tiles", "n_numbers", "ore_wheat", "wood_brick"],
    "number quality": ["has_6_8", "n_6_8", "has_2_12"],
    "scarcity-weighted pips": ["scarcity_pips"],
    "ports": ["port_any", "port_match", "port_3to1"],
    "expansion room": ["expand_best", "expand_count", "expand_second"],
    "fit with own first settlement": ["own_combined_resources", "own_new_resources", "own_combined_pips", "own_dist"],
    "opponent proximity": ["opp_dist", "opp_contest"],
}
groups = c("board") * 1e4 + c("game")


def cv_r2(cols, lam=1.0, folds=5):
    A = X[:, [idx[k] for k in cols]]
    A = (A - A.mean(0)) / (A.std(0) + 1e-9)
    A = np.hstack([A, np.ones((len(A), 1))])
    g = np.unique(groups); rng = np.random.default_rng(0); rng.shuffle(g)
    fold = {gg: i % folds for i, gg in enumerate(g)}
    f = np.array([fold[gg] for gg in groups])
    pred = np.zeros_like(target)
    for k in range(folds):
        tr, te = f != k, f == k
        Wt = A[tr] * w[tr, None]
        M = Wt.T @ A[tr] + lam * np.eye(A.shape[1]); M[-1, -1] -= lam
        coef = np.linalg.solve(M, Wt.T @ target[tr])
        pred[te] = A[te] @ coef
    ss = lambda e: (w * e ** 2).sum()
    return 1 - ss(target - pred) / ss(target - np.average(target, weights=w))


print("\nweighted out-of-fold R^2 of the specialist's preference shift:")
for name, cols in GROUPS.items():
    print(f"  {name:34s}: {cv_r2(cols):.3f}")
r_all = cv_r2(FEATS)
print(f"  all features: {r_all:.3f}")
for name, cols in GROUPS.items():
    rest = [k for k in FEATS if k not in cols]
    print(f"  all minus {name:30s}: {cv_r2(rest):.3f}  (drop {r_all - cv_r2(rest):+.3f})")

print("\nweighted correlation of each feature with the shift (|r| > 0.04):")
mw = lambda x: np.average(x, weights=w)
for k in FEATS:
    x = c(k)
    if x.std() == 0:
        continue
    r = mw((x - mw(x)) * (target - mw(target))) / np.sqrt(mw((x - mw(x)) ** 2) * mw((target - mw(target)) ** 2))
    if abs(r) > 0.04:
        print(f"  {k:24s} r = {r:+.3f}")

print("\nby placement (1st / 2nd settlement): top-spot agreement and mean |shift| on considered spots")
for p in (1, 2):
    m = c("placement") == p
    ds = np.unique(decision[m]); ag = 0
    for d in ds:
        mm = np.where(decision == d)[0]
        ag += np.argmax(c("p_gen")[mm]) == np.argmax(c("p_spec")[mm])
    considered = m & (c("p_gen") > 0.05)
    print(f"  placement {p}: agree {ag / len(ds):.1%}, mean |shift| {np.abs(shift[considered]).mean():.3f} over {considered.sum()} spots the generalist considers")
