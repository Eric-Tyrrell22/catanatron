#!/bin/bash
# The separate-opening-network arm (branch opening-net in the worktree ../catanatron-opening).
# The learner runs the branch's az.py against the branch's extension (target/pylib), from this
# checkout's python dir so runs/on1 lands where the dashboard and scorers look. The actors need
# the image built from the branch (infra/push.sh run in the worktree) and on1 in runs/fleet.json.
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
WT=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs
export PYTHONPATH=$WT/target/pylib
until ! ps -eo args | grep -q "[a]z.py --run on1 "; do sleep 5; done   # a stopping learner still writes its run dir
rm -rf runs/on1 runs/on1.vs_ham runs/on1.panel
# rr2's bundle minus the reach block, main network no longer oversamples placements (--opening-share 0: the opening
# network plays them), opening network: same 4x512 shape as a copy of o3-187, lr 2e-4 (the specialists' rate).
COMMON="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 0 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-branches 0 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --move-forks 2 --sibling-labels --hard-boards 0 --value-positions 8 --opening-share 0.0 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 --opening-net --opening-lr 2e-4 --opening-replay 100000 --opening-batch 512 --opening-reuse 8 --opening-min-replay 5000"
nohup .venv/bin/python $WT/python/az.py --run on1 --remote-control-key models/control_on1.json $COMMON > runs/on1.out 2>&1 &
echo "learner started (pid $!) $(date +%H:%M)"
sleep 90
nohup .venv/bin/python $WT/python/runs/score_runs.py openhex_models/ham.ctnn on1 > runs/on1_score.out 2>&1 &
nohup nice -n 5 .venv/bin/python $WT/python/runs/panel_score.py on1 > runs/on1_panel.out 2>&1 &
echo "launched $(date +%H:%M); scorers follow runs/on1 with the snap@osnap key"
