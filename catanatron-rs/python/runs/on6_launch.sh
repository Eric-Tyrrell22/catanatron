#!/bin/bash
# Launches on6: on5's recipe plus the shared per-spot settlement head on the opening network
# (--opening-spot-head, model.SpotHead; CTNN version 3), continuing from on5's weights. The head's output
# starts at zero, so on6 plays as on5 did at first. Needs the actor image pushed from the opening branch
# (infra/push.sh, for the version-3 loader) and the branch's target/pylib rebuilt, both after on5 stops.
#   bash runs/on6_launch.sh
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
bash runs/on_launch.sh on6 --init runs/on5/model.pt --opening-init runs/on5/omodel.pt \
  --structure-features --relation-features --award-features --opening-spot-head --opening-spot-width 64 \
  --placement-horizon-turns 5 --placement-playouts 12 --placement-horizon-tau 0.1 --opening-value-q-weight 0.75 \
  --max-steps 400 --max-opening-steps 300
sleep 60
bash runs/arm_monitors.sh on6 on5:control 100000 200000 300000
