#!/bin/bash
# Hands the fleet from one opening-network arm to the next: stops the old arm's monitors and learner
# (SIGTERM: it finishes the iteration and saves; waits for "done:"), installs the branch's freshly built
# extension as target/pylib (the old learner's mapped copy is unlinked, not overwritten), points
# runs/fleet.json at the new arm (same game cost unless given), and runs the new arm's launch script.
# The actor image must already be pushed (infra/push.sh) when the engine changed.
#   bash runs/arm_switch.sh <old arm> <new arm> [game cost for the new arm]
set -euo pipefail
old=$1; new=$2; cost=${3:-}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
[ -x runs/${new}_launch.sh ] || { echo "runs/${new}_launch.sh missing"; exit 1; }
ls $WT/target/wheels_probe/catanatron_rs-*.whl >/dev/null 2>&1 || { echo "no wheel in $WT/target/wheels_probe (build it with maturin first)"; exit 1; }
for p in $(ps -eo pid,args | grep -E "python (/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/python/)?runs/(score_runs|panel_score|agree100|value_track|arm_checkpoint)\.py .*\b$old\b" | grep -v grep | grep -v shell-snapshots | awk '{print $1}'); do kill $p 2>/dev/null && echo "stopped $old monitor $p"; done
pid=$(ps -eo pid,args | awk -v run="$old" '$2 ~ /python$/ && $3 ~ /az\.py$/ && $4 == "--run" && $5 == run {print $1}')
if [ -n "$pid" ]; then
  kill -TERM $pid; echo "$(date +%H:%M) SIGTERM to $old learner $pid, waiting for its checkpoint"
  until ! kill -0 $pid 2>/dev/null; do sleep 5; done
  tail -n 1 runs/$old.out | cut -c1-200
else echo "$old learner not running"; fi
rm -rf $WT/target/pylib && mkdir -p $WT/target/pylib && unzip -q -o $WT/target/wheels_probe/catanatron_rs-*.whl -d $WT/target/pylib && echo "pylib installed from $(ls $WT/target/wheels_probe/*.whl | xargs -n1 basename)"
python3 - "$old" "$new" "$cost" <<'PY'
import json, sys
old, new, cost = sys.argv[1], sys.argv[2], sys.argv[3]
p = "runs/fleet.json"; d = json.load(open(p))
share = d["arms"].get(old, next(iter(d["arms"].values())))
d["arms"] = {new: share}
d["arm_game_cost"][new] = float(cost) if cost else d["arm_game_cost"].get(old, 1.0)
json.dump(d, open(p, "w"), indent=1); print("fleet arms", d["arms"], "cost", d["arm_game_cost"][new])
PY
nohup bash runs/${new}_launch.sh > runs/${new}_launch.out 2>&1 &
echo "$(date +%H:%M) $new launcher started (runs/${new}_launch.out)"
