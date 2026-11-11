#!/bin/bash
# Single arm from o3-187: move forks (2 mid-game road/robber counterfactual branches per game).
# Read against the recorded control curve (runs/ac, runs/kc, runs/o4): runs/kl_compare.py ac:control mf:move-forks
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
rm -rf runs/mf runs/mf.vs_ham
COMMON="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --opening-share 0.3 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --aux-extra-weight 0.0 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 12 --remote-max-lag 8"
nohup .venv/bin/python az.py --run mf --move-forks 2 --remote-control-key models/control_mf.json $COMMON > runs/mf.out 2>&1 &
sleep 90
sed -i 's/^ARMS=.*/ARMS="mf"/; s/^per_arm() .*/per_arm() { echo 1; }/' runs/ax_keeper.sh
nohup bash runs/ax_keeper.sh > runs/mf_keeper.out 2>&1 &
nohup .venv/bin/python runs/score_runs.py openhex_models/ham.ctnn mf > runs/mf_score.out 2>&1 &
echo "launched $(date +%H:%M)"
