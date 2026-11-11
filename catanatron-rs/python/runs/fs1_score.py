"""Follows runs/fs1b: each new snapshot plays o3-187 and the sp2 specialist on board 1010194 (network only, 4k games)."""
import glob, os, re, sys, time
sys.path.insert(0, os.getcwd())
import catanatron_rs
from catanatron_env import OPENHEX_1V1
refs = {"Camon": os.path.abspath("openhex_models/camon.ctnn"), "Spook": os.path.abspath("openhex_models/spook.ctnn"),
        "Breeze": os.path.abspath("openhex_models/breeze.ctnn"), "o3-187": os.path.abspath("runs/o3/o3_187.ctnn"),
        "specialist": os.path.abspath("runs/sp2-1010194/snap_000020.ctnn")}
seen = set()
try:
    seen = {m.group(1) for l in open("runs/fs1b.vs_o3") for m in [re.search(r"(snap_\d+)", l)] if m}
except FileNotFoundError:
    pass
while True:
    snaps = [p for p in sorted(glob.glob("runs/fs1b/snap_*.ctnn")) if os.path.basename(p)[:-5] not in seen]
    if not snaps:
        time.sleep(30); continue
    p = snaps[-1]; time.sleep(2)
    parts = []
    for name, ref in refs.items():
        (w, _), _ = catanatron_rs.play_bots(["NN:" + os.path.abspath(p), "NN:" + ref], 2000, 4242, 6, map_type="BASE", board="seed:1010194", **OPENHEX_1V1)
        parts.append(f"vs {name} {w / 20:.1f}%")
    line = f"{time.strftime('%H:%M')} fs1b {os.path.basename(p)[:-5]}: " + "  ".join(parts) + " (±1.1, board 1010194)"
    open("runs/fs1b.vs_o3", "a").write(line + "\n"); print(line, flush=True); seen.add(os.path.basename(p)[:-5])
