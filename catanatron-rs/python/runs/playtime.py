"""Coarse play-time tournament: each variant (one change to Ham's live search)
plays Ham's baseline search, scaled to 1,000 simulations per move, over the
same 400 random-board games. Prints win rate and Elo difference as each
matchup finishes.

    python runs/playtime.py [games] [threads]
"""
import math, os, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1

games = int(sys.argv[1]) if len(sys.argv) > 1 else 400
threads = int(sys.argv[2]) if len(sys.argv) > 2 else 12
rules = {"map_type": "BASE", **OPENHEX_1V1}
ham = os.path.abspath("openhex_models/ham.ctnn")
o4 = os.path.abspath("runs/hp1_pool/o4_final.ctnn")
LIVE = "c1,f0.25,m0.1,s0.3/150"  # Ham's live search settings
base = f"MCTS:1000,{LIVE}:{ham}"
variants = [
    ("4x search everywhere", f"MCTS:4000,{LIVE}:{ham}"),
    ("16x search everywhere", f"MCTS:16000,{LIVE}:{ham}"),
    ("o4 weights (same search)", f"MCTS:1000,{LIVE}:{o4}"),
    ("c_puct 0.7", f"MCTS:1000,c0.7,f0.25,m0.1,s0.3/150:{ham}"),
    ("c_puct 1.5", f"MCTS:1000,c1.5,f0.25,m0.1,s0.3/150:{ham}"),
    ("fpu 0", f"MCTS:1000,c1,f0,m0.1,s0.3/150:{ham}"),
    ("fpu 0.5", f"MCTS:1000,c1,f0.5,m0.1,s0.3/150:{ham}"),
    ("win-only utility (no margin/speed)", f"MCTS:1000,c1,f0.25,m0,s0/150:{ham}"),
    ("16x search on opening placements only", f"MCTS:1000,{LIVE},p16000:{ham}"),
]


def elo(p):
    p = min(max(p, 1e-3), 1 - 1e-3)
    return -400 * math.log10(1 / p - 1)


err = (0.25 / games) ** 0.5
print(f"{time.strftime('%H:%M')} baseline: Ham, 1,000 sims, {LIVE}; {games} games per matchup "
      f"(±{err:.1%} ≈ ±{elo(0.5 + err):.0f} Elo)", flush=True)
for name, key in variants:
    t = time.time()
    (w, l), capped = catanatron_rs.play_bots([key, base], games, 90_000, threads, **rules)
    p = w / games
    print(f"{time.strftime('%H:%M')} {name:40s} {p:6.1%}  {elo(p):+5.0f} Elo  ({w}-{l}, {capped} capped, "
          f"{(time.time() - t) / 60:.0f} min)", flush=True)
print("done", flush=True)
