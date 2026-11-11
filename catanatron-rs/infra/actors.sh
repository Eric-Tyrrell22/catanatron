#!/usr/bin/env bash
# Remote self-play actors on the ECS cluster (one per spot instance).
#   infra/actors.sh start [N] [x86|arm]   run N actor tasks (instances scale up to fit, up to the stack's max)
#   infra/actors.sh bench [x86|arm]       run the parity check + self-play benchmark once (python bench.py)
#   infra/actors.sh status                tasks and instances
#   infra/actors.sh stop                  stop every task (instances scale in ~15 min later)
#   infra/actors.sh logs                  tail the actors' logs
set -euo pipefail
cluster=catan-selfplay
output() {
  aws cloudformation describe-stacks --stack-name CatanSelfPlay \
    --query "Stacks[0].Outputs[?OutputKey=='$1'].OutputValue" --output text
}
provider() {
  case "${1:-x86}" in
    x86) echo catan-spot ;;
    arm) echo catan-spot-arm ;;
    *) echo "arch must be x86 or arm, not $1" >&2; exit 1 ;;
  esac
}
run() { # count provider [overrides json]
  aws ecs run-task --cluster "$cluster" --task-definition catan-actor --count "$1" \
    --capacity-provider-strategy capacityProvider="$2",weight=1 \
    --placement-constraints type=distinctInstance ${3:+--overrides "$3"} \
    --query 'tasks[].[taskArn,lastStatus]' --output text
}
case "${1:-}" in
  start)
    run "${2:-1}" "$(provider "${3:-x86}")"
    ;;
  bench)
    run 1 "$(provider "${2:-x86}")" '{"containerOverrides":[{"name":"actor","command":["python","bench.py"]}]}'
    ;;
  status)
    tasks=$(aws ecs list-tasks --cluster "$cluster" --query 'taskArns' --output text)
    if [[ -n "$tasks" ]]; then
      aws ecs describe-tasks --cluster "$cluster" --tasks $tasks \
        --query 'tasks[].[taskArn,lastStatus,capacityProviderName,stoppedReason]' --output text
    else
      echo "no tasks"
    fi
    for g in AsgName ArmAsgName; do
      aws autoscaling describe-auto-scaling-groups --auto-scaling-group-names "$(output $g)" \
        --query 'AutoScalingGroups[0].[DesiredCapacity,Instances[].[InstanceType,AvailabilityZone,LifecycleState]]' --output text
    done
    ;;
  stop)
    for t in $(aws ecs list-tasks --cluster "$cluster" --query 'taskArns' --output text); do
      aws ecs stop-task --cluster "$cluster" --task "$t" --query 'task.lastStatus' --output text
    done
    ;;
  logs)
    group=$(aws logs describe-log-groups --query "logGroups[?contains(logGroupName,'ActorLogs')].logGroupName | [0]" --output text)
    aws logs tail "$group" --follow
    ;;
  *)
    sed -n '2,7p' "$0"; exit 1 ;;
esac
