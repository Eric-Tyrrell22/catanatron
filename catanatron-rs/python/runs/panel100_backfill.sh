#!/bin/bash
# Backfills a finished run's 100-board panel (runs/<run>.panel100, the follower's format) as a coarse
# reference: the snapshot nearest each multiple of <games> plus the last snapshot, in order.
#   bash runs/panel100_backfill.sh <run> <games>      # e.g. on2 100000, on1 200000
set -euo pipefail
run=$1; every=$2
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
for s in $(python3 - "$run" "$every" <<'PY'
import json, sys, glob, os
run, every = sys.argv[1], int(sys.argv[2])
games = {}
for l in open(f"runs/{run}/log.jsonl"):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    games[r["iter"]] = r["games"]
snaps = {int(os.path.basename(p)[5:-5]): p for p in glob.glob(f"runs/{run}/snap_*.ctnn")}
picks = [min(snaps, key=lambda i: abs(games.get(i, 0) - g)) for g in range(every, max(games.values()) + 1, every)]
picks.append(max(snaps))
print("\n".join(snaps[i] for i in sorted(set(picks))))
PY
); do
  n=$(basename $s .ctnn | sed 's/snap_0*//')
  grep -q "snap_$(printf %06d $n):" runs/$run.panel100 2>/dev/null && continue
  nice -n 15 .venv/bin/python $WT/python/runs/panel_score.py --once "$run snap_$(printf %06d $n)" $s >> runs/$run.panel100
done
echo "$(date +%H:%M) backfill of $run done"
