"""Scores each new snapshot of a pool-expert run against its base network
(network only): on held-out boards from the same pool (never trained on),
on the training pool, and on random boards (to catch collapse elsewhere).

    python runs/vs_pool.py <run> <base.ctnn> <holdout seeds file> <train seeds file> [games]
"""
import glob, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

run, base, hold_file, train_file = sys.argv[1:5]
games = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
rules = {"map_type": "BASE", **OPENHEX_1V1}
spec = lambda path: "seeds:" + ",".join(l.strip() for l in open(path) if l.strip())
hold, train = spec(hold_file), spec(train_file)
base_key = "NN:" + os.path.abspath(base)


def play(key, board, seed):
    (w, _), _ = catanatron_rs.play_bots([key, base_key], games, seed, 14, board=board, **rules)
    return w / games


err = 100 * (0.25 / games) ** 0.5
seen = set()
while True:
    for path in sorted(set(glob.glob(f"runs/{run}/snap_*.ctnn")) - seen):
        time.sleep(3)  # let the writer finish
        key = "NN:" + os.path.abspath(path)
        h, t, r = play(key, hold, 51_000), play(key, train, 52_000), play(key, None, 53_000)
        print(f"{time.strftime('%H:%M')} {os.path.basename(path)} vs base: held-out hard boards {h:.1%}, "
              f"training pool {t:.1%}, random boards {r:.1%} (each ±{err:.1f})", flush=True)
        seen.add(path)
    if os.path.exists(f"runs/{run}/model.pt") and not any(
            p for p in os.popen("ps -eo args").read().splitlines() if p.split()[1:4] == ["-u", "az.py", "--run"] and run in p.split()):
        # The learner has finished and every snapshot is scored.
        if set(glob.glob(f"runs/{run}/snap_*.ctnn")) <= seen:
            print("done", flush=True)
            break
    time.sleep(20)
