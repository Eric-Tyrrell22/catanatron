#!/bin/bash
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
END=$(( $(date +%s) + 36000 ))
while [ $(date +%s) -lt $END ]; do
  running=$(aws ecs describe-tasks --cluster catan-selfplay --tasks $(aws ecs list-tasks --cluster catan-selfplay --desired-status RUNNING --query 'taskArns' --output text 2>/dev/null) --query 'tasks[].overrides.containerOverrides[0].command[3]' --output text 2>/dev/null)
  for r in vq25 vf vf2; do
    pgrep -f "az.py --run $r " >/dev/null || continue   # arm's learner gone: don't restart its actor
    if ! echo "$running" | grep -q "control_$r.json"; then
      O=$(python3 -c "import json;print(json.dumps({'containerOverrides':[{'name':'actor','command':['python','actor.py','--control','models/control_$r.json']}]}))")
      aws ecs run-task --cluster catan-selfplay --task-definition catan-actor --count 1 --capacity-provider-strategy capacityProvider=catan-spot,weight=1 --placement-constraints type=distinctInstance --overrides "$O" >/dev/null 2>&1 && echo "$(date +%H:%M) restarted actor for $r"
    fi
  done
  sleep 120
done
echo "$(date +%H:%M) keeper: safety deadline, stopping everything"
for p in $(ps -eo pid,args | awk '$2==".venv/bin/python" && $4=="az.py" {print $1}'); do kill -TERM $p; done; sleep 60
bash ../infra/actors.sh stop >/dev/null 2>&1
for g in $(aws autoscaling describe-auto-scaling-groups --query "AutoScalingGroups[?starts_with(AutoScalingGroupName,'CatanSelfPlay-')].AutoScalingGroupName" --output text); do aws autoscaling set-desired-capacity --auto-scaling-group-name $g --desired-capacity 0; done
