"""Scores every new snapshot of several runs against one reference network
(network only, 10,000 random-board games, ±0.5), round-robin, one process.

    python runs/score_runs.py <ref.ctnn> <run> [<run> ...]

Appends "<time> <run> snap_NNNNNN.ctnn: P% vs <Ref>" to runs/<run>.vs_<ref>.
"""
import glob, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

ref_path, runs = sys.argv[1], sys.argv[2:]
ref_name = os.path.basename(ref_path).split(".")[0]
ref = "NN:" + os.path.abspath(ref_path)
import re
def already(run):
    """Snapshots already scored for this run (so a restarted scorer skips them)."""
    try:
        return {f"runs/{run}/{m.group(1)}" for l in open(f"runs/{run}.vs_{ref_name}") for m in [re.search(r"(snap_\d+\.ctnn)", l)] if m}
    except FileNotFoundError:
        return set()
seen = {r: already(r) for r in runs}
while True:
    idle = True
    for run in runs:
        new = sorted(set(glob.glob(f"runs/{run}/snap_*.ctnn")) - seen[run])
        if not new:
            continue
        idle = False
        path = new[0]  # oldest unscored first, so every run advances
        time.sleep(3)
        (w, _), _ = catanatron_rs.play_bots(["NN:" + os.path.abspath(path), ref], 10000, 70_000, 14, map_type="BASE", **OPENHEX_1V1)
        line = f"{time.strftime('%H:%M')} {run} {os.path.basename(path)}: {w / 100:.1f}% vs {ref_name.capitalize()} (±0.5)"
        with open(f"runs/{run}.vs_{ref_name}", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
        seen[run].add(path)
    if idle:
        time.sleep(20)
