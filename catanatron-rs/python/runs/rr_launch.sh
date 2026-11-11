#!/bin/bash
# Single arm "rr" from o3-187 (one arm at a time, read against the recorded control curve):
#   reach/contention/affordability inputs (--reach-features, first-layer columns zero-initialised),
#   sibling-averaged value labels at opening branches and placement/move forks (--sibling-labels),
#   two move forks per game (--move-forks 2),
#   value loss weighted so each game counts ~8 positions (--value-positions 8, AlphaGo-style),
#   prioritised boards: 256 seeds with the largest opening value error, 25% of games (--hard-boards/--hard-share).
# Needs the actor image pushed after these engine changes (infra/push.sh).
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until ! ps -eo args | grep -q "[a]z.py --run rr "; do sleep 5; done   # a stopping learner still writes its run dir
rm -rf runs/rr runs/rr.vs_ham
COMMON="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --reach-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --move-forks 2 --sibling-labels --hard-boards 256 --hard-share 0.25 --value-positions 8 --opening-share 0.3 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8"
nohup .venv/bin/python az.py --run rr --remote-control-key models/control_rr.json $COMMON > runs/rr.out 2>&1 &
sleep 90
sed -i 's/^ARMS=.*/ARMS="rr"/' runs/ax_keeper.sh
nohup bash runs/ax_keeper.sh > runs/rr_keeper.out 2>&1 &
nohup .venv/bin/python runs/score_runs.py openhex_models/ham.ctnn rr > runs/rr_score.out 2>&1 &
echo "launched $(date +%H:%M)"
