#!/bin/bash
# Games-based checkpoint for the kl runs: prints the matched-games table when
# every arm has passed 150k games (and again at 300k), and stops everything
# (learners, actor tasks, instances) at 300k or at the safety deadline.
# The decision to continue past 150k is the user's; this only reports.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
ARMS="kc ke kk kka"
SAFETY=$(( $(date +%s) + 36000 ))   # 10 hours
games() { grep "^it" runs/$1.out | tail -1 | awk '{gsub(",","",$4); print $4}'; }
report150=0
while true; do
  now=$(date +%s)
  min=999999999; for a in $ARMS; do g=$(games $a); g=${g:-0}; [ "$g" -lt "$min" ] && min=$g; done
  if [ "$report150" = 0 ] && [ "$min" -ge 150000 ]; then
    echo "$(date +%H:%M) every arm past 150k games"; .venv/bin/python runs/kl_compare.py; report150=1
  fi
  if [ "$min" -ge 300000 ] || [ "$now" -ge "$SAFETY" ]; then
    echo "$(date +%H:%M) stopping: min games $min, safety $([ "$now" -ge "$SAFETY" ] && echo hit || echo ok)"
    for p in $(ps -eo pid,args | awk '$2==".venv/bin/python" && $4=="az.py" {print $1}'); do kill -TERM $p; done
    sleep 90
    bash ../infra/actors.sh stop >/dev/null 2>&1
    for g in $(aws autoscaling describe-auto-scaling-groups --query "AutoScalingGroups[?starts_with(AutoScalingGroupName,'CatanSelfPlay-')].AutoScalingGroupName" --output text); do
      aws autoscaling set-desired-capacity --auto-scaling-group-name $g --desired-capacity 0; done
    .venv/bin/python runs/kl_compare.py
    echo "$(date +%H:%M) done"; exit 0
  fi
  sleep 120
done
