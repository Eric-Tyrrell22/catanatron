#!/bin/bash
# The 100-board specialist batch from on6 (spot head + all blocks): one specialist after another on the panel
# boards, each from on6's main and opening networks, actors via models/control_sp4.json (runs/fleet.json arms
# must list sp4; the keeper hands every box to it while no other learner runs). Resumable: finished boards are
# skipped. Results: runs/sp4/results.jsonl; networks: runs/sp4-<seed>/snap_*.ctnn (+ osnap_*).
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
BASE_SNAP=$(ls runs/on6/snap_*.ctnn | tail -n 1); BASE_OSNAP=runs/on6/o$(basename $BASE_SNAP)
echo "$(date +%H:%M) sp4 batch: base runs/on6/model.pt + omodel.pt, evaluation opponent $BASE_SNAP@$BASE_OSNAP" >> runs/sp4.out
.venv/bin/python -u specialists.py --az $WT/python/az.py --boards runs/boards/panel100.json --prefix sp4 --base-pt runs/on6/model.pt --base-net $BASE_SNAP@$BASE_OSNAP \
  --max-iters 24 --success 2.0 --eval-every 8 --eval-games 1000 --threads 8 -- \
  --opening-init runs/on6/omodel.pt --structure-features --relation-features --award-features --opening-net --opening-spot-head --opening-spot-width 64 \
  --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 \
  --c-puct 1.0 --fpu 0.25 --lr 2e-4 --replay 200000 --replay-device cpu --min-replay 20000 --eval-games 0 --eval-search-games 0 --eval-random-games 0 \
  --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --fork-settlements-only --placement-temp 2.0 --placement-uniform 0.25 \
  --move-forks 0 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 \
  --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 600 \
  --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-control-key models/control_sp4.json --remote-games-per-core 4 --remote-max-shards 4 --remote-max-lag 8 \
  --opening-lr 4e-4 --opening-replay 20000 --opening-batch 256 --opening-reuse 8 --opening-min-replay 1000 \
  --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --opening-value-q-weight 0.75 --max-steps 400 --max-opening-steps 300 >> runs/sp4.out 2>&1
echo sp4-done >> runs/sp4.out
