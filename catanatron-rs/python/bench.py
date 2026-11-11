"""Actor-host check and benchmark (runs inside the actor image).

    python bench.py

1. Parity: the network's outputs on stored observations must match the
   reference computed on an x86 build bit for bit (bench/parity.npz).
2. Self-play throughput with oh2's settings on every core: games and
   recorded examples per second, per second per core.
"""

import os
import platform
import sys
import time

import numpy as np
import catanatron_rs

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "bench", "model.ctnn")
RULES = {"map_type": "BASE", "vps_to_win": 15, "discard_limit": 9, "friendly_robber": "public",
         "dice": "openhex", "numbers": "spiral-random-corner"}


def parity():
    ref = np.load(os.path.join(HERE, "bench", "parity.npz"))
    obs, logits, values = ref["obs"], ref["logits"], ref["values"]
    mismatches = 0
    for k in range(len(obs)):
        lg, v = catanatron_rs.net_forward(MODEL, obs[k].astype("<f4").tobytes())
        lg = np.frombuffer(lg, "<f4")
        if not (np.array_equal(lg, logits[k]) and np.float32(v) == values[k]):
            mismatches += 1
    print(f"parity: {len(obs) - mismatches}/{len(obs)} observations bit-identical to the x86 reference", flush=True)
    return mismatches == 0


def throughput(games_per_core=4):
    cores = os.cpu_count()
    net = open(MODEL, "rb").read()
    games = games_per_core * cores
    t = time.time()
    d = catanatron_rs.self_play(net, games, 12345, 0, 400, 50, 0.25, 2400, 15, 0.25, 0.3, 1.0, 0.25, 0.0, 13, 1, 1, 1, 2.0,
                                placement_uniform=0.25, margin_weight=0.1, speed_weight=0.1, speed_turns=400.0, **RULES)
    dt = time.time() - t
    print(f"self-play: {games} games, {d['n']} examples in {dt:.1f}s on {cores} cores = "
          f"{games / dt:.2f} games/s, {d['n'] / dt:.0f} examples/s, {d['n'] / dt / cores:.1f} examples/s/core", flush=True)


if __name__ == "__main__":
    print(f"host: {platform.machine()}, {os.cpu_count()} cores", flush=True)
    ok = parity()
    throughput()
    sys.exit(0 if ok else 1)
