"""Board panel: every new snapshot of the given runs plays each held-out
board specialist on its board (network only, OpenHex rules, 4,000 games,
±0.8), the sensitive yardstick for board-specific evaluation. Appends
"<time> <run> snap_N: mean P% | board:P% ..." to runs/<run>.panel.

    python runs/panel_score.py <run> [<run> ...]            # follow runs (loops)
    python runs/panel_score.py --once <label> <net.ctnn> ... # score fixed nets, print only
"""
import glob, os, re, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

PANEL = {b: os.path.abspath(f"runs/sp2-{b}/snap_000020.ctnn") for b in (1010194, 1010752, 1011188, 1012006)}
GAMES, THREADS = 4000, 6


def score(path):
    out = {}
    for board, spec in PANEL.items():
        (w, _), _ = catanatron_rs.play_bots(["NN:" + os.path.abspath(path), "NN:" + spec], GAMES, 4242, THREADS,
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
        return {m.group(1) for l in open(f"runs/{run}.panel") for m in [re.search(r"(snap_\d+)", l)] if m}
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
        with open(f"runs/{run}.panel", "a") as f:
            f.write(l + "\n")
        print(l, flush=True)
        seen[run].add(os.path.basename(path)[:-5])
    if idle:
        time.sleep(30)
