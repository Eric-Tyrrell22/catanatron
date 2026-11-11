"""Evaluate the latest checkpoint of a run against a bot, with and without search.

    .venv/bin/python eval.py --run long1
    .venv/bin/python eval.py --run long1 --sims 0,100,400 --games 400

Safe to run while training: it reads the checkpoint (written atomically),
exports a snapshot and uses only a few threads. Results are appended to
runs/<run>/eval.jsonl so progress can be compared over time.
"""

import argparse
import json
import math
import os
import time

import torch

import catanatron_rs
from catanatron_env import add_rule_args, rules_from_args
from export import export


def wilson(wins, n, z=1.96):
    """95% confidence interval for a win rate."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--opponent", default="F")
    ap.add_argument("--sims", default="0,100", help="simulations per move; 0 = network only")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777_000, help="fixed so evaluations are comparable")
    add_rule_args(ap)
    args = ap.parse_args()

    run_dir = os.path.join("runs", args.run)
    ckpt = os.path.join(run_dir, args.checkpoint)
    steps = torch.load(ckpt, map_location="cpu")["meta"]["total_steps"]
    # Export to a per-step file: the Rust side caches networks by path.
    snap = os.path.abspath(os.path.join(run_dir, f"snap_{steps}.ctnn"))
    if not os.path.exists(snap):
        export(ckpt, snap)
    print(f"{args.run}: checkpoint at {steps:,} steps, vs {args.opponent}, "
          f"{args.games} games, {args.threads} threads")

    results = []
    for sims in [int(s) for s in args.sims.split(",")]:
        key = f"NN:{snap}" if sims == 0 else f"MCTS:{sims}:{snap}"
        t = time.time()
        (wins, _), none = catanatron_rs.play_bots([key, args.opponent], args.games, args.seed, args.threads,
                                                   **rules_from_args(args))
        lo, hi = wilson(wins, args.games)
        label = "network only" if sims == 0 else f"search {sims} sims"
        print(f"  {label:<18} {wins / args.games:6.1%}  (95% CI {lo:.1%}-{hi:.1%})  "
              f"{time.time() - t:5.0f}s")
        results.append({"sims": sims, "wins": wins, "games": args.games, "no_winner": none,
                        "win_rate": wins / args.games, "ci": [lo, hi]})

    with open(os.path.join(run_dir, "eval.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.time(), "steps": steps, "opponent": args.opponent, "rules": rules_from_args(args),
                            "results": results}) + "\n")


if __name__ == "__main__":
    main()
