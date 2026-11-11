"""A vs control, B vs control, A vs B. Mirrored boards: each seed is played
twice with seats swapped, so board and seat luck cancel within the pair.

    python runs/head2head.py [pairs] [threads]
"""
import math, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

pairs = int(sys.argv[1]) if len(sys.argv) > 1 else 300
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 14
rules = {"map_type": "BASE", **OPENHEX_1V1}
ham = os.path.abspath("openhex_models/ham.ctnn")
bots = {
    "control": f"MCTS:1000,c1,f0.25,m0.1,s0.3/150:{ham}",
    "A (fpu0.75+d0.5+k0.95+q0.2)": f"MCTS:1000,c1,f0.75,m0.1,s0.3/150,d0.5,k0.95,q0.2:{ham}",
    "B (q0.2 only)": f"MCTS:1000,c1,f0.25,m0.1,s0.3/150,q0.2:{ham}",
}
def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3); return -400 * math.log10(1 / p - 1)
games = 2 * pairs
print(f"{time.strftime('%H:%M')} {pairs} mirrored pairs ({games} games) per matchup, ±{(0.25/games)**0.5:.1%}", flush=True)
for x, y in [("A (fpu0.75+d0.5+k0.95+q0.2)", "control"), ("B (q0.2 only)", "control"), ("A (fpu0.75+d0.5+k0.95+q0.2)", "B (q0.2 only)")]:
    t = time.time()
    (w1, l1), c1 = catanatron_rs.play_bots([bots[x], bots[y]], pairs, 95_000, threads, **rules)
    (l2, w2), c2 = catanatron_rs.play_bots([bots[y], bots[x]], pairs, 95_000, threads, **rules)
    w = w1 + w2; p = w / games
    print(f"{time.strftime('%H:%M')} {x:30s} vs {y:30s} {p:6.1%} {elo(p):+5.0f} Elo  ({w}-{games-w}, first-seat {w1}/{pairs}, second-seat {w2}/{pairs}, {(time.time()-t)/60:.0f} min)", flush=True)
print("done", flush=True)
