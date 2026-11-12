#!/bin/bash
# Resumes the 100-board specialist batch once the fleet keeper has an sp3 actor running
# (so the first board after the pause isn't trained without shards), then, when the batch
# finishes, drops sp3 from the keeper's force_active list so its box is released.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
sp3_tasks() {
  tasks=$(aws ecs list-tasks --cluster catan-selfplay --desired-status RUNNING --query 'taskArns' --output text)
  [ -z "$tasks" ] && { echo 0; return; }
  aws ecs describe-tasks --cluster catan-selfplay --tasks $tasks --query 'tasks[?lastStatus==`RUNNING`].overrides.containerOverrides[0].command[3]' --output text | tr '\t' '\n' | grep -c control_sp3
}
until [ "$(sp3_tasks)" -ge 1 ]; do sleep 30; done
before=$(wc -l < runs/sp3.out)
sleep 60   # let the actor's first shard land
nohup bash runs/sp3_batch.sh >/dev/null 2>&1 &
echo "$(date +%H:%M) sp3 batch resumed (pid $!)"
until [ "$(tail -n +$((before + 1)) runs/sp3.out | grep -c '^sp3-done')" -ge 1 ]; do sleep 120; done
python3 - <<'EOF'
import json
p = "runs/fleet.json"; d = json.load(open(p)); d["force_active"] = [a for a in d.get("force_active", []) if a != "sp3"]
json.dump(d, open(p, "w"), indent=1); print("sp3 removed from force_active")
EOF
echo "$(date +%H:%M) sp3 batch done; keeper will release its box"
