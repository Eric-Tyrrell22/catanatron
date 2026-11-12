#!/bin/bash
# Continues on1 after its 12-hour budget ends (promote rule: keep training). Waits for the learner to
# finish normally, then resumes it with the same flags for another 12 hours. Scorers keep following runs/on1.
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
until ! ps -eo args | grep -q "[a]z.py --run on1 "; do sleep 60; done
if [ -z "${FORCE:-}" ]; then tail -n 3 runs/on1.out | grep -q "^done:" || { echo "$(date +%H:%M) on1's learner exited without finishing normally; not resuming (FORCE=1 overrides)"; exit 1; }; fi
COMMON="--symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 0 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-branches 0 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --move-forks 2 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 --opening-net --opening-lr 2e-4 --opening-replay 100000 --opening-batch 512 --opening-reuse 8 --opening-min-replay 5000"
nohup .venv/bin/python $WT/python/az.py --run on1 --resume --remote-control-key models/control_on1.json $COMMON >> runs/on1.out 2>&1 &
echo "$(date +%H:%M) on1 resumed (pid $!)"
