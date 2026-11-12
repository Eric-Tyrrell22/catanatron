#!/bin/bash
# Launches on5: on4's recipe (opening network, structure + relation blocks, horizon value labels at
# settlements with the policy from search) plus the award + port-fit block (--award-features), continuing
# from on4's weights (new inputs start at zero, so on5 plays as on4 did at first). Needs the actor image
# pushed from the opening branch first (infra/push.sh) and the branch's target/pylib rebuilt.
#   bash runs/on5_launch.sh
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
bash runs/on_launch.sh on5 --init runs/on4/model.pt --opening-init runs/on4/omodel.pt \
  --structure-features --relation-features --award-features \
  --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --opening-value-q-weight 0.75 \
  --max-steps 400 --max-opening-steps 300
sleep 60
bash runs/arm_monitors.sh on5 on4:control 100000 200000 300000
