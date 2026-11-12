#!/bin/bash
# Launch an opening-network arm variant from the opening-net branch (see on1_launch.sh for the baseline).
#   bash runs/on_launch.sh <run> [extra az.py flags...]
# e.g. bash runs/on_launch.sh on2 --opening-lr 5e-5
#      bash runs/on_launch.sh on3 --opening-hidden 1024 --opening-layers 6
# Then add <run> to runs/fleet.json "arms". One arm at a time: stop the previous learner first.
set -euo pipefail
run=$1; shift
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
# Wait for the previous learner of this run to exit: match the learner process itself (python az.py --run <run>),
# not any shell whose command line merely mentions it (a wrapper shell with the same text once blocked this forever).
until ! ps -eo args | awk -v run="$run" '$1 ~ /python$/ && $2 ~ /az\.py$/ && $3 == "--run" && $4 == run' | grep -q .; do sleep 5; done
rm -rf runs/$run runs/$run.vs_ham runs/$run.panel
BASE="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 0 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-branches 0 --opening-branches 1 --placement-forks 4 --fork-settlements-only --placement-margin 2.0 --regret-forks --placement-temp 2.0 --placement-uniform 0.1 --move-forks 0 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 --opening-net --opening-lr 2e-4 --opening-replay 100000 --opening-batch 512 --opening-reuse 8 --opening-min-replay 5000"
# Later flags override earlier ones in argparse, so the variant's flags go last.
nohup .venv/bin/python $WT/python/az.py --run $run --remote-control-key models/control_$run.json $BASE "$@" > runs/$run.out 2>&1 &
echo "learner $run started (pid $!) $(date +%H:%M) with: $*"
sleep 90
nohup .venv/bin/python $WT/python/runs/score_runs.py openhex_models/ham.ctnn $run > runs/${run}_score.out 2>&1 &
nohup .venv/bin/python $WT/python/runs/score_runs.py --every-games 10000 openhex_models/gaz.ctnn@openhex_models/gaz_opening.ctnn $run > runs/${run}_score_gaz.out 2>&1 &
nohup nice -n 5 .venv/bin/python $WT/python/runs/panel_score.py $run > runs/${run}_panel.out 2>&1 &
echo "launched $run $(date +%H:%M)"
