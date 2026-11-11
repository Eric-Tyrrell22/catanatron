"""Board panel: every new snapshot of the given runs plays each of the 100
sp3 board specialists (best snapshot per board, runs/sp3/results.jsonl) on
its board (network only, OpenHex rules, 400 games each, 40k per snapshot;
mean ±0.3), the sensitive yardstick for board-specific evaluation. Appends
"<time> <run> snap_N: mean P% | board:P% ..." to runs/<run>.panel100.
(Until Nov 11 13:50 this scored the 4 sp2 boards with 4,000 games each into
runs/<run>.panel; those files stay as they are.)

    python runs/panel_score.py <run> [<run> ...]            # follow runs (loops)
    python runs/panel_score.py --once <label> <net.ctnn> ... # score fixed nets, print only
"""
import glob, os, re, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

import json


def specialists():
    """board seed -> the sp3 specialist's best snapshot (as runs/panel100_score.py)."""
    out = {}
    for line in open("runs/sp3/results.jsonl"):
        r = json.loads(line)
        if not r.get("evals"):
            continue
        best = max(r["evals"], key=lambda e: e["win"])["iter"]
        path = os.path.abspath(f"runs/{r['run']}/snap_{best:06d}.ctnn")
        if os.path.exists(path):
            out[r["seed"]] = path
    return dict(sorted(out.items()))


PANEL = specialists()
GAMES, THREADS = 400, 6


def net_key(path):
    """A snapshot's player key; with an opening network beside it (az.py --opening-net: osnap_N.ctnn), both."""
    opath = os.path.join(os.path.dirname(path), "o" + os.path.basename(path))
    return "NN:" + os.path.abspath(path) + (f"@{os.path.abspath(opath)}" if os.path.exists(opath) else "")


def score(path):
    out = {}
    for board, spec in PANEL.items():
        (w, _), _ = catanatron_rs.play_bots([net_key(path), "NN:" + spec], GAMES, 4242, THREADS,
                                            map_type="BASE", board=f"seed:{board}", **OPENHEX_1V1)
        out[board] = 100.0 * w / GAMES
    return out


def line(label, s):
    mean = sum(s.values()) / len(s)
    return f"{time.strftime('%H:%M')} {label}: mean {mean:.1f}% | " + " ".join(f"{b}:{p:.1f}%" for b, p in s.items())


if sys.argv[1] == "--once":
    args = sys.argv[2:]
    for label, path in zip(args[::2], args[1::2]):
        print(line(label, score(path)), flush=True)
    sys.exit()

runs = sys.argv[1:]


def done(run):
    try:
        return {m.group(1) for l in open(f"runs/{run}.panel100") for m in [re.search(r"(snap_\d+)", l)] if m}
    except FileNotFoundError:
        return set()


seen = {r: done(r) for r in runs}
while True:
    idle = True
    for run in runs:
        snaps = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))
        new = [p for p in snaps if os.path.basename(p)[:-5] not in seen[run]]
        if not new:
            continue
        idle = False
        path = new[-1]  # the newest first: the panel is for reading the trend as it happens
        time.sleep(2)
        l = line(f"{run} {os.path.basename(path)[:-5]}", score(path))
        with open(f"runs/{run}.panel100", "a") as f:
            f.write(l + "\n")
        print(l, flush=True)
        seen[run].add(os.path.basename(path)[:-5])
    if idle:
        time.sleep(30)
