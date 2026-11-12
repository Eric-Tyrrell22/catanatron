"""First-pick agreement with the 100 board specialists for a run's latest snapshot
(its opening network when present): on how many of the 100 panel boards does the
network's top first settlement equal the expert's? Uses the experts' first-move
policies cached by runs/expert_picks.py (runs/expert_picks/<board>.txt) and the
node_features probe for the network. Appends "<time> <run> snap_N: agree K/100
(prob mass on the expert's pick P)" to runs/<run>.agree.

    PYTHONPATH=<branch>/target/pylib .venv/bin/python runs/agree100.py <run> [--loop MINUTES]
"""
import glob
import os
import subprocess
import sys
import time

WT = "/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs"
PROBE = f"{WT}/target/release/examples/node_features"
run = sys.argv[1]
loop = int(sys.argv[sys.argv.index("--loop") + 1]) if "--loop" in sys.argv else 0

experts = {}
for fn in glob.glob("runs/expert_picks/*.txt"):
    board = int(os.path.basename(fn)[:-4])
    for l in open(fn):
        if l.startswith("#policy expert"):
            items = l.split()[2:]
            experts[board] = max(items, key=lambda it: float(it.split(":")[1])).split(":")[0]


def policy(board, key):
    out = subprocess.run([PROBE, str(board), f"net={key}"], capture_output=True, text=True, check=True).stdout
    for l in out.splitlines():
        if l.startswith("#policy net"):
            return {it.split(":")[0]: float(it.split(":")[1]) for it in l.split()[2:]}
    return {}


done = set()
try:
    done = {l.split()[2].rstrip(":") for l in open(f"runs/{run}.agree")}
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
            agree, mass = 0, 0.0
            for board, pick in sorted(experts.items()):
                pol = policy(board, key)
                if pol:
                    agree += max(pol, key=pol.get) == pick
                    mass += pol.get(pick, 0.0)
            line = f"{time.strftime('%H:%M')} {run} {name}: agree {agree}/{len(experts)} (prob mass on the expert's pick {mass / len(experts):.3f})"
            with open(f"runs/{run}.agree", "a") as f:
                f.write(line + "\n")
            print(line, flush=True)
            done.add(name)
    if not loop:
        break
    time.sleep(loop * 60)
