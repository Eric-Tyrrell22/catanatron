#!/bin/bash
# Two arms from o3-187: ax = per-action-type auxiliary targets (--aux-extra-weight 0.5),
# ac = control (same code, extra targets untrained). Same recipe as vq25 otherwise.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
COMMON="--init runs/o3/model.pt --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 5e-5 --replay 600000 --replay-device cpu --min-replay 50000 --eval-every 10 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --opening-share 0.3 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --value-q-weight 0.25 --hidden 512 --layers 4 --minutes 720 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8"
nohup .venv/bin/python az.py --run ax --aux-extra-weight 0.5 --remote-control-key models/control_ax.json $COMMON > runs/ax.out 2>&1 &
nohup .venv/bin/python az.py --run ac --aux-extra-weight 0.0 --remote-control-key models/control_ac.json $COMMON > runs/ac.out 2>&1 &
sleep 90
nohup bash runs/ax_keeper.sh > runs/ax_keeper.out 2>&1 &
nohup .venv/bin/python runs/score_runs.py openhex_models/ham.ctnn ax ac > runs/ax_score.out 2>&1 &
echo "launched $(date +%H:%M)"
