"""The 100-board panel: a network plays each sp3 specialist (its best snapshot) on
its own board, network only, OpenHex rules. Prints and appends to runs/panel100.txt:
"<time> <label>: mean P% (±se) | min/max | <board>:P% ..." (per-board detail in
runs/panel100/<label>.json).

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/panel100_score.py <label> <net.ctnn[@opening.ctnn]> [more label net pairs] [--games N] [--threads T]

A snapshot with an osnap_*.ctnn beside it is played with both networks.
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
import catanatron_rs  # noqa: E402
from catanatron_env import OPENHEX_1V1  # noqa: E402

args = sys.argv[1:]
GAMES = int(args.pop(args.index("--games") + 1)) if "--games" in args else 1000
if "--games" in args:
    args.remove("--games")
THREADS = int(args.pop(args.index("--threads") + 1)) if "--threads" in args else 6
if "--threads" in args:
    args.remove("--threads")


def specialists():
    out = {}
    for line in open("runs/sp3/results.jsonl"):
        r = json.loads(line)
        if not r.get("evals"):
            continue
        best = max(r["evals"], key=lambda e: e["win"])["iter"]
        path = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn")
        if os.path.exists(path):
            out[r["seed"]] = path
    return out


def net_key(path):
    if "@" in path:
        return "NN:" + "@".join(os.path.abspath(p) for p in path.split("@"))
    opath = os.path.join(os.path.dirname(path), "o" + os.path.basename(path))
    return "NN:" + os.path.abspath(path) + (f"@{os.path.abspath(opath)}" if os.path.exists(opath) else "")


def score(path, panel):
    out = {}
    for board, spec in sorted(panel.items()):
        (w, _), _ = catanatron_rs.play_bots([net_key(path), "NN:" + spec], GAMES, 4242, THREADS, map_type="BASE",
                                            board=f"seed:{board}", **OPENHEX_1V1)
        out[board] = 100.0 * w / GAMES
    return out


panel = specialists()
os.makedirs("runs/panel100", exist_ok=True)
for label, path in zip(args[::2], args[1::2]):
    t0 = time.time()
    s = score(path, panel)
    vals = list(s.values())
    mean = sum(vals) / len(vals)
    se = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1) / len(vals)) ** 0.5
    line = (f"{time.strftime('%H:%M')} {label}: mean {mean:.1f}% (±{se:.1f} over {len(vals)} boards, {GAMES} games each) | "
            f"min {min(vals):.1f} max {max(vals):.1f} | {time.time() - t0:.0f}s")
    with open("runs/panel100.txt", "a") as f:
        f.write(line + "\n")
    with open(f"runs/panel100/{label}.json", "w") as f:
        json.dump({"label": label, "net": path, "games": GAMES, "mean": mean, "se": se, "boards": s}, f, indent=1)
    print(line, flush=True)
