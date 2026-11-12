#!/bin/bash
# Arms the standing monitors for an opening-network arm (launched with runs/on_launch.sh, which
# already starts the Ham scorer and the 4-board panel scorer):
#   - first-pick agreement with the 100 experts every 30 min      -> runs/<run>.agree
#   - placement value tracker (Spearman vs forced win rates) hourly -> runs/<run>.valuetrack
#   - checkpoint reports at the given game counts (default 50k/100k/200k): control comparison,
#     panel and Ham lines, then the 100-board panel of the latest snapshot -> runs/<run>_checkpoints.out
#   bash runs/arm_monitors.sh <run> [control label, default on2:control] [game counts...]
set -euo pipefail
run=$1; control=${2:-on2:control}; shift $(( $# >= 2 ? 2 : $# ))
counts=${@:-50000 100000 200000}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
nohup .venv/bin/python runs/agree100.py $run --loop 30 > runs/${run}_agree.out 2>&1 &
nohup .venv/bin/python runs/value_track.py $run --loop 60 > runs/${run}_valuetrack.out 2>&1 &
for g in $counts; do
  nohup bash -c "
    .venv/bin/python runs/arm_checkpoint.py $run $g $control
    S=\$(ls runs/$run/snap_*.ctnn | sort | tail -n 1); N=\$(basename \$S .ctnn | sed 's/snap_0*//')
    .venv/bin/python runs/panel100_score.py $run@\${N}it \$S --threads 6 | tail -n 1
  " >> runs/${run}_checkpoints.out 2>&1 &
done
echo "monitors armed for $run: agree (30 min), value tracker (60 min), checkpoints at $counts vs $control"
