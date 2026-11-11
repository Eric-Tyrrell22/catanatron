#!/bin/bash
# Single arm "rr2" from o3-187: the rr bundle with the reach block on its own weights at 10x the learning rate
# (--reach-lr-mult 10), so the new inputs actually engage within a 100k-game run.
#   reach features (own weight tensor, lr 5e-4), sibling-averaged value labels, two move forks per game
#   (runner-up alternatives), prioritised boards 256/25%, value loss weighted to ~8 positions per game.
# Actor image: no engine change since rr (the exported network is one matrix), so no push needed.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until ! ps -eo args | grep -q "[a]z.py --run rr2 "; do sleep 5; done   # a stopping learner still writes its run dir
rm -rf runs/rr2 runs/rr2.vs_ham runs/rr2.panel
COMMON="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --reach-features --reach-lr-mult 10 --games 0 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --move-forks 2 --sibling-labels --hard-boards 256 --hard-share 0.25 --value-positions 8 --opening-share 0.3 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8"
nohup .venv/bin/python az.py --run rr2 --remote-control-key models/control_rr2.json $COMMON > runs/rr2.out 2>&1 &
sleep 90
nohup .venv/bin/python runs/score_runs.py openhex_models/ham.ctnn rr2 > runs/rr2_score.out 2>&1 &
nohup nice -n 5 .venv/bin/python runs/panel_score.py rr2 > runs/rr2_panel.out 2>&1 &
echo "launched $(date +%H:%M); add rr2 to the keeper's ARMS (per_arm 2) to get actors"
