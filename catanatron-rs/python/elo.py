"""Round-robin Elo between runs' latest snapshots (plus fixed bots).

    .venv/bin/python elo.py --runs m-base,m-surp,m-esc,m-end,m-all --bots F --map MINI --games 1000
    .venv/bin/python elo.py ... --sims 100 --games 200

Every pair plays `--games` games (seats are shuffled per game by the engine).
Ratings are a Bradley-Terry maximum-likelihood fit, scaled to Elo and
anchored so the first bot (or the first run, without bots) is 0. Games with
no winner count half a win each. Uses the newest snap_*.ctnn of each run, or
the one from `--iter`.
"""

import argparse
import glob
import itertools
import json
import math
import os
import time

import catanatron_rs
from catanatron_env import add_rule_args, rules_from_args


def latest_snapshot(run, iteration):
    if iteration:
        path = os.path.join("runs", run, f"snap_{iteration:06d}.ctnn")
        if not os.path.exists(path):
            raise SystemExit(f"{path} does not exist")
        return path
    snaps = sorted(glob.glob(os.path.join("runs", run, "snap_*.ctnn")))
    if not snaps:
        raise SystemExit(f"runs/{run} has no snapshots yet")
    return snaps[-1]


def best_snapshots(run, n):
    """The `n` snapshots with the best network-only win rate vs F in the
    run's log (each eval writes the snapshot it measured)."""
    with open(os.path.join("runs", run, "log.jsonl")) as f:
        evals = [e for e in map(json.loads, f) if "vs_F" in e]
    evals.sort(key=lambda e: -e["vs_F"]["net"])
    paths = [os.path.join("runs", run, f"snap_{e['iter']:06d}.ctnn") for e in evals[:n]]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"logged snapshots are missing: {missing}")
    if not paths:
        raise SystemExit(f"runs/{run} has no evaluated snapshots yet")
    return sorted(paths)


def bradley_terry(names, score, games, iters=2000):
    """Strengths from pairwise scores (score[i][j] = i's points vs j) by the
    standard MM iteration; returns Elo ratings."""
    n = len(names)
    p = [1.0] * n
    for _ in range(iters):
        new = []
        for i in range(n):
            w = sum(score[i][j] for j in range(n) if j != i)
            d = sum(games[i][j] / (p[i] + p[j]) for j in range(n) if j != i and games[i][j])
            new.append(max(w, 1e-3) / d)
        g = math.exp(sum(math.log(x) for x in new) / n)
        p = [x / g for x in new]
    return [400 * math.log10(x) for x in p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="comma-separated run names")
    ap.add_argument("--bots", default="F", help="comma-separated fixed bots to include (empty for none)")
    ap.add_argument("--iter", type=int, default=0, help="snapshot iteration to use (0 = newest)")
    ap.add_argument("--top", type=int, default=0,
                    help="instead of one snapshot per run, its N best by network-only win rate vs F")
    ap.add_argument("--sims", type=int, default=0, help="search simulations per move; 0 = network only")
    ap.add_argument("--games", type=int, default=1000, help="games per pair")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=888_000)
    add_rule_args(ap)
    args = ap.parse_args()

    entrants = []
    for run in args.runs.split(","):
        snaps = best_snapshots(run, args.top) if args.top else [latest_snapshot(run, args.iter)]
        for snap in map(os.path.abspath, snaps):
            key = f"NN:{snap}" if args.sims == 0 else f"MCTS:{args.sims}:{snap}"
            entrants.append((f"{run}@{os.path.basename(snap)[5:11].lstrip('0')}", key))
    bots = [(b, b) for b in args.bots.split(",") if b]
    entrants = bots + entrants
    names = [e[0] for e in entrants]
    n = len(entrants)
    score = [[0.0] * n for _ in range(n)]
    games = [[0] * n for _ in range(n)]

    t0 = time.time()
    pairs = list(itertools.combinations(range(n), 2))
    print(f"{n} entrants, {len(pairs)} pairs x {args.games} games", flush=True)
    for k, (i, j) in enumerate(pairs):
        (wins, none) = catanatron_rs.play_bots([entrants[i][1], entrants[j][1]], args.games, args.seed,
                                               args.threads, **rules_from_args(args))
        score[i][j] = wins[0] + none / 2
        score[j][i] = wins[1] + none / 2
        games[i][j] = games[j][i] = args.games
        if n <= 8:
            print(f"  {names[i]:>16} vs {names[j]:<16} {score[i][j] / args.games:6.1%}", flush=True)
        elif (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(pairs)} pairs, {time.time() - t0:.0f}s", flush=True)

    elo = bradley_terry(names, score, games)
    elo = [e - elo[0] for e in elo]
    print(f"\nElo ({'network only' if args.sims == 0 else f'{args.sims} sims'}, {args.games} games per pair, "
          f"anchored {names[0]} = 0, {time.time() - t0:.0f}s):")
    for k in sorted(range(n), key=lambda k: -elo[k]):
        print(f"  {names[k]:>16}  {elo[k]:+6.0f}")
    with open(os.path.join("runs", "elo.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), "sims": args.sims, "games": args.games, "rules": rules_from_args(args),
                            "names": names, "score": score, "elo": elo}) + "\n")


if __name__ == "__main__":
    main()
