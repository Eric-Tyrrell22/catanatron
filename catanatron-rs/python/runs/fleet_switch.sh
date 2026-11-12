#!/bin/bash
# Hands the actors from the stack's Auto Scaling group + runs/ax_keeper.sh to
# runs/fleet.py (price-per-game spot fleet). The group's boxes and their tasks
# go away within a minute or two; fleet.py then launches the cheapest pools.
# (A later `cdk deploy` of the stack restores the group's max size to 4: set it
# back to 0 afterwards, or the two keepers will both provision actors.)
set -euo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
ASG=CatanSelfPlay-AsgASGD1D7B4E2-R4PCAFCK8BiL
for p in $(ps -eo pid,args | awk '/bash runs\/ax_keeper.sh/ && !/awk/ {print $1}'); do kill $p && echo "stopped old keeper $p"; done
aws autoscaling update-auto-scaling-group --auto-scaling-group-name $ASG --min-size 0 --max-size 0 --desired-capacity 0
echo "$(date +%H:%M) group $ASG set to 0/0/0 (its boxes and tasks terminate now)"
nohup .venv/bin/python runs/fleet.py > runs/fleet.out 2>&1 &
echo "$(date +%H:%M) fleet keeper started (pid $!), log runs/fleet.out, state runs/fleet_state.json"
