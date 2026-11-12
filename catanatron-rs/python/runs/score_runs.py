"""Scores every new snapshot of several runs against one reference network
(network only, 10,000 random-board games, ±0.5), round-robin, one process.

    python runs/score_runs.py <ref.ctnn> <run> [<run> ...]

Appends "<time> <run> snap_NNNNNN.ctnn: P% vs <Ref>" to runs/<run>.vs_<ref>.
"""
import glob, json, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

args = sys.argv[1:]
# Cadence in games (default: every snapshot). --every-games N scores a snapshot only once N games have been
# played since the last scored one; --backfill-games M scores, of the snapshots already there, only the one
# nearest each multiple of M (0 = none), so a late-started reference gets coarse history and fine new points.
def opt(k, d):
    if k in args:
        i = args.index(k); v = int(args[i + 1]); del args[i:i + 2]; return v
    return d
EVERY, BACKFILL = opt("--every-games", 0), opt("--backfill-games", -1)
ref_path, runs = args[0], args[1:]
# A reference with its own opening network: "main.ctnn@opening.ctnn" (e.g. openhex_models/gaz.ctnn@openhex_models/gaz_opening.ctnn).
ref_name = os.path.basename(ref_path.split("@")[0]).split(".")[0]
ref = "NN:" + "@".join(os.path.abspath(p) for p in ref_path.split("@"))
import re


def net_key(path):
    """A snapshot's player key; with an opening network beside it (az.py --opening-net: osnap_N.ctnn), both."""
    opath = os.path.join(os.path.dirname(path), "o" + os.path.basename(path))
    return "NN:" + os.path.abspath(path) + (f"@{os.path.abspath(opath)}" if os.path.exists(opath) else "")



def already(run):
    """Snapshots already scored for this run (so a restarted scorer skips them)."""
    try:
        return {f"runs/{run}/{m.group(1)}" for l in open(f"runs/{run}.vs_{ref_name}") for m in [re.search(r"(snap_\d+\.ctnn)", l)] if m}
    except FileNotFoundError:
        return set()
def games_of(run):
    """Snapshot iteration -> games played, from the run's log."""
    out = {}
    try:
        for l in open(f"runs/{run}/log.jsonl"):
            try:
                r = json.loads(l); out[r["iter"]] = r["games"]
            except (ValueError, KeyError):
                pass
    except FileNotFoundError:
        pass
    return out
def snap_iter(path):
    return int(os.path.basename(path)[5:-5])
seen = {r: already(r) for r in runs}
last_games = {r: 0 for r in runs}
backfill_keep = {}
if BACKFILL >= 0:
    # Existing snapshots: keep only the one nearest each multiple of BACKFILL (none when 0); mark the rest seen.
    for run in runs:
        snaps = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))
        g = games_of(run)
        keep = set()
        if BACKFILL > 0 and snaps:
            top = max(g.values()) if g else 0
            for target in range(BACKFILL, top + 1, BACKFILL):
                keep.add(min(snaps, key=lambda p: abs(g.get(snap_iter(p), 0) - target)))
        backfill_keep[run] = keep
        seen[run] |= set(snaps) - keep
        if snaps:
            last_games[run] = g.get(snap_iter(snaps[-1]), 0) - EVERY  # the first new snapshot after start is due
while True:
    idle = True
    for run in runs:
        new = sorted(set(glob.glob(f"runs/{run}/snap_*.ctnn")) - seen[run])
        if EVERY > 0 and new:
            # Forward fill: only a snapshot EVERY games past the last scored one (the backfill picks always count).
            g = games_of(run)
            due = [p for p in new if g.get(snap_iter(p), 0) - last_games[run] >= EVERY or p in backfill_keep.get(run, set())]
            seen[run] |= set(new) - set(due)
            new = due
        if not new:
            continue
        idle = False
        path = new[0]  # oldest unscored first, so every run advances
        time.sleep(3)
        (w, _), _ = catanatron_rs.play_bots([net_key(path), ref], 10000, 70_000, 14, map_type="BASE", **OPENHEX_1V1)
        line = f"{time.strftime('%H:%M')} {run} {os.path.basename(path)}: {w / 100:.1f}% vs {ref_name.capitalize()} (±0.5)"
        with open(f"runs/{run}.vs_{ref_name}", "a") as f:
            f.write(line + "\n")
        print(line, flush=True)
        seen[run].add(path)
        last_games[run] = max(last_games[run], games_of(run).get(snap_iter(path), 0))
    if idle:
        time.sleep(20)
