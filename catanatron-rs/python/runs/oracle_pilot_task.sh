#!/bin/bash
# Runs the opening-oracle labelling on a borrowed keeper box as an ECS task (no new image: the static
# probes forced_pick / forced_obs and the nets sit under s3://<bucket>/models/pilot/, the only prefix the
# task role may read; results go to shards/pilot/out_<seed>.tgz, the only prefix it may write).
# Stops the box's on3 actor while the task runs; the keeper re-places an actor when the task exits.
#   bash runs/oracle_pilot_task.sh <first seed> <boards> [games per pick, default 48] [instance id]
set -euo pipefail
seed=$1; n=$2; games=${3:-48}; inst=${4:-}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
B=catanselfplay-bucket83908e77-vusaorgcj8uf
CL=catan-selfplay
if [ -z "$inst" ]; then
  # The smallest registered box (fewest vCPUs).
  inst=$(aws ecs describe-container-instances --cluster $CL --container-instances $(aws ecs list-container-instances --cluster $CL --query 'containerInstanceArns' --output text) \
    --query 'containerInstances[].[registeredResources[?name==`CPU`].integerValue|[0],ec2InstanceId]' --output text | sort -n | head -n 1 | cut -f2)
fi
CI=$(aws ecs list-container-instances --cluster $CL --filter "ec2InstanceId == $inst" --query 'containerInstanceArns[0]' --output text)
NAME=$(aws ecs describe-task-definition --task-definition catan-actor --query 'taskDefinition.containerDefinitions[0].name' --output text)
last=$((seed + n - 1))
script="set -e; cd /tmp
python3 -c \"import boto3; s3=boto3.client('s3'); [s3.download_file('$B', 'models/pilot/'+k, k) for k in ('forced_pick','forced_obs','snap.ctnn','osnap.ctnn')]\"
chmod +x forced_pick forced_obs; mkdir -p out/picks out/obs; export K=/tmp/snap.ctnn@/tmp/osnap.ctnn
seq $seed $last | xargs -P 16 -I{} sh -c './forced_pick NN:\$K NN:\$K {} top:8 $games 4 > out/picks/{}.txt && ./forced_obs {} \$K | gzip > out/obs/{}.txt.gz && echo done {}'
tar cf out.tar out
python3 -c \"import boto3; boto3.client('s3').upload_file('out.tar', '$B', 'shards/pilot/out_$seed.tar')\"
echo uploaded"
python3 -c "import json,sys; print(json.dumps({'containerOverrides': [{'name': sys.argv[1], 'command': ['sh', '-c', sys.argv[2]]}]}))" "$NAME" "$script" > /home/eric/.claude/jobs/d24dadb6/tmp/pilot_override_$seed.json
OLD=$(aws ecs list-tasks --cluster $CL --container-instance $CI --query 'taskArns[0]' --output text)
TASK=$(aws ecs start-task --cluster $CL --task-definition catan-actor --container-instances $CI --overrides file:///home/eric/.claude/jobs/d24dadb6/tmp/pilot_override_$seed.json --query 'tasks[0].taskArn' --output text)
echo "$(date +%H:%M) pilot task ${TASK##*/} on $inst: seeds $seed..$last, $games games per pick"
until [ "$(aws ecs describe-tasks --cluster $CL --tasks $TASK --query 'tasks[0].lastStatus' --output text)" != "PENDING" ]; do sleep 5; done
if [ "$OLD" != "None" ]; then aws ecs stop-task --cluster $CL --task $OLD --reason "box lent to the oracle pilot" --query 'task.lastStatus' --output text; fi
echo "task ${TASK##*/}; results at s3://$B/shards/pilot/out_$seed.tar; log stream actor/actor/${TASK##*/}"
