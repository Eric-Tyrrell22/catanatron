#!/bin/bash
# Trial board specialist from on6 (spot head + all blocks) on one panel board, to refresh the expert labels:
# the learner trains on that board only (actors via models/control_sp4.json; add "sp4" to runs/fleet.json arms
# and force_active first). Snapshots every 10 iterations; evaluate with runs/sp4_eval.py.
#   bash runs/sp4_trial.sh <board seed> [minutes=45]
set -euo pipefail
seed=$1; minutes=${2:-45}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
run=sp4-$seed
rm -rf runs/$run
nohup .venv/bin/python -u $WT/python/az.py --run $run --init runs/on6/model.pt --opening-init runs/on6/omodel.pt --fixed-board seed:$seed \
  --structure-features --relation-features --award-features --opening-net --opening-spot-head --opening-spot-width 64 \
  --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 \
  --c-puct 1.0 --fpu 0.25 --lr 2e-4 --replay 200000 --replay-device cpu --min-replay 20000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 \
  --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --fork-settlements-only --placement-temp 2.0 --placement-uniform 0.25 \
  --move-forks 0 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 \
  --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes $minutes \
  --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-control-key models/control_sp4.json --remote-games-per-core 4 --remote-max-shards 2 --remote-max-lag 8 \
  --opening-lr 4e-4 --opening-replay 20000 --opening-batch 256 --opening-reuse 8 --opening-min-replay 1000 \
  --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --opening-value-q-weight 0.75 --max-steps 400 --max-opening-steps 300 > runs/$run.out 2>&1 &
echo "$(date +%H:%M) $run started (pid $!), $minutes minutes"
