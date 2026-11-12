"""Waits until an arm has passed a game count and its panel and Ham scorers
have scored a snapshot from past it, then prints the kill-rule inputs: the
control comparison around that count, the latest panel and Ham lines, and
the reach-block weight norm of the latest snapshot (runs with --reach-lr-mult).

    .venv/bin/python runs/arm_checkpoint.py <run> <games> [<control label>]   # e.g. rr2 100000 ac:control
"""
import glob
import json
import os
import re
import struct
import subprocess
import sys
import time

import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
run, target = sys.argv[1], int(sys.argv[2])
control = sys.argv[3] if len(sys.argv) > 3 else "ac:control"


def entries():
    return [json.loads(l) for l in open(f"runs/{run}/log.jsonl")]


def scored(path, first_iter):
    try:
        return [l.rstrip() for l in open(path) if (m := re.search(r"snap_(\d+)", l)) and int(m.group(1)) >= first_iter]
    except FileNotFoundError:
        return []


def learner_alive():
    return subprocess.run(f"ps -eo args | grep -q '[a]z.py --run {run} '", shell=True).returncode == 0


def reach_norm(path, extra):
    """Norm of the first layer's trailing `extra` input columns in a CTNN file (the reach block) and of the rest."""
    b = open(path, "rb").read()
    rows, cols = struct.unpack_from("<II", b, 20)
    w = np.frombuffer(b, "<f4", rows * cols, 28).reshape(rows, cols)
    return float(np.linalg.norm(w[:, cols - extra:])), float(np.linalg.norm(w[:, :cols - extra]))


while True:
    first = next((e["iter"] for e in entries() if e["games"] >= target), None)
    if first is not None:
        panel, ham = scored(f"runs/{run}.panel100" if os.path.exists(f"runs/{run}.panel100") else f"runs/{run}.panel", first), scored(f"runs/{run}.vs_ham", first)
        if panel and ham:
            break
    if not learner_alive():
        print(f"{time.strftime('%H:%M')}: {run}'s learner is gone before {target:,} games", flush=True)
        sys.exit(1)
    time.sleep(120)

print(f"{time.strftime('%H:%M')} == {run} at {target:,} games (first iteration past it: {first})")
rows = subprocess.run([".venv/bin/python", "runs/kl_compare.py", control, f"{run}:{run}"], capture_output=True, text=True).stdout
lo, hi = target - 20_000, target + 10_000
print("\n".join(l for l in rows.splitlines() if (m := re.match(r"\s*(\d+) \|", l)) and lo <= int(m.group(1)) <= hi or l.startswith(("control", run, "  games"))))
print("panel:", *panel[-3:], sep="\n  ")
print("vs Ham:", *ham[-3:], sep="\n  ")
try:
    import torch
    meta = torch.load(f"runs/{run}/model.pt", map_location="cpu")["meta"]
    if meta.get("extra_in"):
        snap = sorted(glob.glob(f"runs/{run}/snap_*.ctnn"))[-1]
        r, m = reach_norm(snap, meta["extra_in"])
        print(f"reach-block weight norm {r:.1f} (main first layer {m:.1f}) at {os.path.basename(snap)}")
except Exception as e:  # the report is still useful without the norm
    print(f"reach norm unavailable: {e}")
