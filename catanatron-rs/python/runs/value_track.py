"""Placement value tracker for a run's latest snapshot (its opening network when present),
on the 100 expert boards: how well the value head ranks first settlements against the forced
win rates measured by forced_pick (runs/expert_picks2/forced), and how its one-ply value
argmax and its policy argmax agree with the expert's pick. Appends to runs/<run>.valuetrack:
"<time> <run> snap_N: spearman S regret_pol P regret_val V agree_pol A agree_val B value=policy C"

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/value_track.py <run> [--loop MINUTES]
"""
import collections
import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
run = sys.argv[1]
loop = int(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else 0


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


forced = collections.defaultdict(dict)
for fn in glob.glob("runs/expert_picks2/forced/*.txt"):
    for l in open(fn):
        m = re.search(r"board (\d+) forced node\s+(\d+): A \(first seat\) wins ([\d.]+)%", l)
        if m:
            forced[int(m.group(1))][int(m.group(2))] = float(m.group(3))
experts = {}
for fn in glob.glob("runs/expert_picks/*.txt"):
    for l in open(fn):
        if l.startswith("#policy expert"):
            items = l.split()[2:]
            experts[int(os.path.basename(fn)[:-4])] = int(max(items, key=lambda it: float(it.split(":")[1])).split(":")[0])


def argmax_line(out, prefix):
    for l in out.splitlines():
        if l.startswith(prefix):
            items = l.split()[2:]
            return {int(a): float(b) for a, b in (it.split(":") for it in items)}
    return {}


done = set()
try:
    done = {l.split()[2].rstrip(":") for l in open(f"runs/{run}.valuetrack")}
except FileNotFoundError:
    pass
while True:
    snaps = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))
    if snaps:
        snap = snaps[-1]
        name = os.path.basename(snap)[:-5]
        if name not in done:
            osnap = os.path.join(os.path.dirname(snap), "o" + os.path.basename(snap))
            key = os.path.abspath(snap) + (f"@{os.path.abspath(osnap)}" if os.path.exists(osnap) else "")
            corr, rp, rv = [], [], []
            agree_pol = agree_val = same = 0
            for board, exp_pick in sorted(experts.items()):
                vals = argmax_line(subprocess.run([f"{WT}/target/release/examples/value_pick", str(board), f"net={key}"], capture_output=True, text=True).stdout, "#value net")
                pol = argmax_line(subprocess.run([f"{WT}/target/release/examples/node_features", str(board), f"net={key}"], capture_output=True, text=True).stdout, "#policy net")
                if not vals or not pol:
                    continue
                vpick, ppick = max(vals, key=vals.get), max(pol, key=pol.get)
                agree_val += vpick == exp_pick
                agree_pol += ppick == exp_pick
                same += vpick == ppick
                fw = forced.get(board, {})
                common = [n for n in fw if n in vals]
                if len(common) >= 5:
                    corr.append(spearman([vals[n] for n in common], [fw[n] for n in common]))
                    best = max(fw.values())
                    if ppick in fw:
                        rp.append(best - fw[ppick])
                    if vpick in fw:
                        rv.append(best - fw[vpick])
            line = (f"{time.strftime('%H:%M')} {run} {name}: spearman {np.mean(corr):.3f} regret_pol {np.mean(rp):.1f} regret_val {np.mean(rv):.1f} "
                    f"agree_pol {agree_pol}/100 agree_val {agree_val}/100 value=policy {same}/100")
            with open(f"runs/{run}.valuetrack", "a") as f:
                f.write(line + "\n")
            print(line, flush=True)
            done.add(name)
    if not loop:
        break
    time.sleep(loop * 60)
