#!/bin/bash
# Resumes on4 (horizon placement targets for settlements, from on3's weights) with its launch flags.
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
until ! ps -eo args | awk -v run=on4 '$1 ~ /python$/ && $2 ~ /az\.py$/ && $3 == "--run" && $4 == run' | grep -q .; do sleep 5; done
if [ -z "${FORCE:-}" ]; then tail -n 3 runs/on4.out | grep -q "^done:" || { echo "$(date +%H:%M) on4's learner exited without finishing normally; not resuming (FORCE=1 overrides)"; exit 1; }; fi
COMMON="--structure-features --relation-features --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 0 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-branches 0 --opening-branches 1 --placement-forks 4 --fork-settlements-only --placement-margin 2.0 --regret-forks --placement-temp 2.0 --placement-uniform 0.1 --move-forks 0 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 --opening-net --opening-lr 2e-4 --opening-replay 100000 --opening-batch 512 --opening-reuse 8 --opening-min-replay 5000 --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --max-steps 400 --max-opening-steps 300 --opening-value-q-weight 0.75"
nohup .venv/bin/python $WT/python/az.py --run on4 --resume --remote-control-key models/control_on4.json $COMMON >> runs/on4.out 2>&1 &
echo "$(date +%H:%M) on4 resumed (pid $!)"
