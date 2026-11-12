#!/bin/bash
# Runs the 100-board specialist batch only while it has an actor: waits for an sp3 actor task,
# starts the batch, pauses it (and drops boards that got < 6 shards) whenever the actors are
# gone for two checks in a row, and when the batch finishes drops sp3 from the fleet keeper's
# force_active list so its box is released.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
MIN_SHARDS=6
sp3_tasks() {
  tasks=$(aws ecs list-tasks --cluster catan-selfplay --desired-status RUNNING --query 'taskArns' --output text 2>/dev/null)
  [ -z "$tasks" ] && { echo 0; return; }
  aws ecs describe-tasks --cluster catan-selfplay --tasks $tasks --query 'tasks[?lastStatus==`RUNNING`].overrides.containerOverrides[0].command[3]' --output text 2>/dev/null | tr '\t' '\n' | grep -c control_sp3
}
stop_batch() {
  for p in $(ps -eo pid,args | awk '/specialists.py --boards/ && !/awk/ {print $1}'); do kill $p; done
  for p in $(ps -eo pid,args | awk '/az.py --run sp3-/ && !/awk/ {print $1}'); do kill -TERM $p; done
  sleep 3
}
drop_underfed() {  # $1 = results line count before this batch segment: drop later boards with too few shards
  python3 - "$1" <<'EOF'
import json, os, sys, re, time
before = int(sys.argv[1]); p = "runs/sp3/results.jsonl"
lines = open(p).read().splitlines()
keep, dropped = lines[:before], []
for l in lines[before:]:
    r = json.loads(l); run = r["run"]; shards = 0
    try:
        for e in open(f"runs/{run}/log.jsonl"):
            shards += json.loads(e).get("remote_shards", 0)
    except FileNotFoundError:
        pass
    if shards >= 6:
        keep.append(l)
    else:
        dropped.append((run, shards))
        if os.path.isdir(f"runs/{run}"):
            os.rename(f"runs/{run}", f"runs/{run}-untrained-{time.strftime('%H%M')}")
open(p, "w").write("\n".join(keep) + ("\n" if keep else ""))
print(f"{time.strftime('%H:%M')} dropped {len(dropped)} underfed boards: {dropped}")
EOF
}
while true; do
  until [ "$(sp3_tasks)" -ge 1 ]; do sleep 30; done
  sleep 60   # the actor's first shard
  before=$(wc -l < runs/sp3/results.jsonl); out_before=$(wc -l < runs/sp3.out)
  nohup bash runs/sp3_batch.sh >/dev/null 2>&1 &
  echo "$(date +%H:%M) batch started (results at $before boards)"
  misses=0
  while ps -eo args | grep -q "[s]pecialists.py --boards"; do
    sleep 30
    if [ "$(sp3_tasks)" -ge 1 ]; then misses=0; else misses=$((misses + 1)); fi
    if [ $misses -ge 2 ]; then
      echo "$(date +%H:%M) no sp3 actor for a minute: pausing the batch"
      stop_batch; drop_underfed "$before"; break
    fi
  done
  if [ "$(tail -n +$((out_before + 1)) runs/sp3.out | grep -c '^sp3-done')" -ge 1 ]; then
    drop_underfed "$before"
    python3 - <<'EOF'
import json
p = "runs/fleet.json"; d = json.load(open(p)); d["force_active"] = [a for a in d.get("force_active", []) if a != "sp3"]
json.dump(d, open(p, "w"), indent=1); print("sp3 removed from force_active")
EOF
    echo "$(date +%H:%M) batch finished"; exit 0
  fi
done
