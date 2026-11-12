#!/bin/bash
# Launches on7: board heads on the main network + opening spot head + mid-game horizon value label (5 turns x 12
# playouts, weight 0.5), from the recovered iteration-110 weights of the first on7 attempt (runs/on7-recover, exact
# fp32 from snap_000110 via runs/ctnn_to_pt.py; the first attempt diverged at iteration 119 on an oversize replay add,
# archived as runs/on7-diverged-1419). Fresh optimizer and replay. Actor image: version-4 loader + horizon playouts.
#   bash runs/on7_launch.sh
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
bash runs/on_launch.sh on7 --init runs/on7-recover/model.pt --opening-init runs/on7-recover/omodel.pt \
  --structure-features --relation-features --award-features --opening-spot-head --opening-spot-width 64 --board-heads --board-width 32 \
  --midgame-horizon-turns 5 --midgame-playouts 12 --midgame-horizon-weight 0.5 --midgame-horizon-ramp-games 30000 --lr-warmup-steps 2000 \
  --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --opening-value-q-weight 0.75 \
  --max-steps 400 --max-opening-steps 300
sleep 60
bash runs/arm_monitors.sh on7 on6:control 100000 200000 300000
