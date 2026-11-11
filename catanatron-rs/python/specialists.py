"""Train one board specialist after another (az.py --fixed-board seed:<n>).

    .venv/bin/python specialists.py --boards runs/boards/round1.json --prefix sp1 \\
        --base-pt runs/oh2/model.pt --base-net openhex_models/breeze.ctnn -- [az.py args...]

Each specialist starts from the base network and trains on its one board
(self-play on the remote actors, which follow whichever run is current). Every
--eval-every iterations its snapshot plays the base network on that board
(network only, --eval-games games, seats alternating); the specialist is done
as soon as it wins at least --success of them, or gives up at --max-iters.

A finished specialist keeps its checkpoint and snapshots; its replay buffer
is deleted. Results go to runs/<prefix>/results.jsonl, one line per board; boards already
there are skipped, so a stopped driver resumes where it left off. Boards are
taken alternately from the "screened" and "random" groups of the board file.
"""

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

import catanatron_rs
from catanatron_env import OPENHEX_1V1

stop = {"requested": False}


def order(boards):
    """Alternate the groups, so results for both come in from the start."""
    groups = {}
    for b in boards:
        groups.setdefault(b["group"], []).append(b)
    out, lists = [], list(groups.values())
    for i in range(max(len(l) for l in lists)):
        out.extend(l[i] for l in lists if i < len(l))
    return out


def evaluate(snap, base_net, board, games, threads):
    (w, l), capped = catanatron_rs.play_bots(["NN:" + os.path.abspath(snap), "NN:" + os.path.abspath(base_net)],
                                             games, 4242, threads, map_type="BASE", board=board, **OPENHEX_1V1)
    return w / games, capped / games


def run_one(b, args, az_args, results_path):
    run = f"{args.prefix}-{b['seed']}"
    board = f"seed:{b['seed']}"
    log_path = os.path.join("runs", run + ".out")
    cmd = [sys.executable, "-u", "az.py", "--run", run, "--init", args.base_pt, "--fixed-board", board,
           "--eval-every", str(args.eval_every), "--eval-games", "0", "--eval-search-games", "0",
           "--eval-random-games", "0"] + az_args
    # A board that was interrupted before has an unfinished run directory
    # (az.py won't start over it); it is set aside, not resumed.
    if os.path.exists(os.path.join("runs", run)):
        stale = os.path.join("runs", f"{run}-interrupted-{time.strftime('%Y%m%d%H%M%S')}")
        os.rename(os.path.join("runs", run), stale)
        print(f"{time.strftime('%H:%M')} {run}: moved an unfinished earlier attempt to {stale}", flush=True)
    t0 = time.time()
    with open(log_path, "a") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    evals, seen, outcome = [], set(), None
    try:
        while outcome is None:
            if stop["requested"]:
                outcome = "interrupted"
                break
            for snap in sorted(set(glob.glob(f"runs/{run}/snap_*.ctnn")) - seen):
                time.sleep(2)  # let the writer finish
                it = int(os.path.basename(snap)[5:11])
                p, capped = evaluate(snap, args.base_net, board, args.eval_games, args.threads)
                evals.append({"iter": it, "win": p, "capped": capped})
                seen.add(snap)
                print(f"{time.strftime('%H:%M')} {run} it {it}: {p:.1%} vs base"
                      f"{f' ({capped:.1%} capped)' if capped else ''}", flush=True)
                if p >= args.success:
                    outcome = "success"
                elif it >= args.max_iters:
                    outcome = "gave_up"
                if outcome:
                    break
            if outcome is None and proc.poll() is not None:
                outcome = f"az_exited_{proc.returncode}"
            if outcome is None:
                time.sleep(10)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            proc.wait()
    best = max(evals, key=lambda e: e["win"]) if evals else None
    # Only the networks are kept (distillation labels positions with them):
    # a specialist's replay buffer is gigabytes.
    replay = os.path.join("runs", run, "replay.pt")
    if outcome != "interrupted" and os.path.exists(replay):
        os.remove(replay)
    rec = {"seed": b["seed"], "group": b["group"], "run": run, "outcome": outcome, "evals": evals,
           "best_iter": best["iter"] if best else None, "best_win": best["win"] if best else None,
           "minutes": (time.time() - t0) / 60, "regret_sum": b.get("regret_sum"), "kl_sum": b.get("kl_sum")}
    if outcome != "interrupted":
        with open(results_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print(f"{time.strftime('%H:%M')} {run} {outcome}: best {rec['best_win']} at it {rec['best_iter']}, "
          f"{rec['minutes']:.0f} min", flush=True)
    return outcome


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: specialists.py [options] -- [az.py args]")
    split = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("--boards", required=True, help="board list (board selection output)")
    ap.add_argument("--prefix", required=True, help="runs are runs/<prefix>-<seed>; results in runs/<prefix>/")
    ap.add_argument("--base-pt", required=True, help="checkpoint every specialist starts from")
    ap.add_argument("--base-net", required=True, help="the same network as .ctnn, the opponent in evaluations")
    ap.add_argument("--success", type=float, default=0.575, help="win rate vs the base on the board that ends a specialist")
    ap.add_argument("--max-iters", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-games", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=12, help="threads for the evaluations")
    args = ap.parse_args(sys.argv[1:split])
    az_args = sys.argv[split + 1:]
    for flag in ("--run", "--init", "--resume", "--fixed-board", "--eval-every"):
        if flag in az_args:
            raise SystemExit(f"{flag} is set by the driver")

    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))
    os.makedirs(os.path.join("runs", args.prefix), exist_ok=True)
    results_path = os.path.join("runs", args.prefix, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            done = {json.loads(l)["seed"] for l in f if l.strip()}
    with open(args.boards) as f:
        boards = order(json.load(f))
    todo = [b for b in boards if b["seed"] not in done]
    print(f"{len(done)} boards done, {len(todo)} to go", flush=True)
    for b in todo:
        if run_one(b, args, az_args, results_path) == "interrupted":
            break
    print("driver finished" if not stop["requested"] else "driver stopped", flush=True)


if __name__ == "__main__":
    main()
