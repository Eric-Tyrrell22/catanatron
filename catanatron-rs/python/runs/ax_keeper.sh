#!/bin/bash
# Keeps PER_ARM spot actors running for each arm (restarts reclaimed ones);
# stops everything at the safety deadline.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
ARMS="sp3 rr2"
per_arm() { echo 2; }
END=$(( $(date +%s) + 43200 ))   # 12 h
while [ $(date +%s) -lt $END ]; do
  tasks=$(aws ecs list-tasks --cluster catan-selfplay --desired-status RUNNING --query 'taskArns' --output text 2>/dev/null)
  running=""
  [ -n "$tasks" ] && running=$(aws ecs describe-tasks --cluster catan-selfplay --tasks $tasks --query 'tasks[].overrides.containerOverrides[0].command[3]' --output text 2>/dev/null)
  learners=$(ps -eo args | grep -o "az.py --run [^ ]*" | awk '{print $3}')   # any python invocation (specialists.py uses python -u)
  for r in $ARMS; do
    echo "$learners" | grep -q "^$r" || continue   # arm's learner gone (prefix match: sp3-<seed>): don't restart its actor
    have=$(echo "$running" | tr '\t ' '\n\n' | grep -c "control_$r.json")
    for _ in $(seq $((have + 1)) $(per_arm $r)); do
      O=$(python3 -c "import json;print(json.dumps({'containerOverrides':[{'name':'actor','command':['python','actor.py','--control','models/control_$r.json']}]}))")
      aws ecs run-task --cluster catan-selfplay --task-definition catan-actor --count 1 --capacity-provider-strategy capacityProvider=catan-spot,weight=1 --placement-constraints type=distinctInstance --overrides "$O" >/dev/null 2>&1 && echo "$(date +%H:%M) started actor for $r"
    done
  done
  sleep 120
done
echo "$(date +%H:%M) keeper: safety deadline, stopping everything"
for p in $(ps -eo pid,args | awk '$2==".venv/bin/python" && $4=="az.py" {print $1}'); do kill -TERM $p; done; sleep 60
bash ../infra/actors.sh stop >/dev/null 2>&1
for g in $(aws autoscaling describe-auto-scaling-groups --query "AutoScalingGroups[?starts_with(AutoScalingGroupName,'CatanSelfPlay-')].AutoScalingGroupName" --output text); do aws autoscaling set-desired-capacity --auto-scaling-group-name $g --desired-capacity 0; done
