#!/bin/bash
# One-screen health check for a training arm: learner, its last iterations and the signs of trouble
# (divergence, collapsed targets, shard bursts), the control file and shard tags in S3, the actor
# tasks and their image, the keeper, the monitors, the yardsticks, and the local load.
#   bash runs/status.sh [run]          # default: the arm in runs/fleet.json
set -uo pipefail
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
run=${1:-$(python3 -c "import json; print(next(iter(json.load(open('runs/fleet.json'))['arms'])))")}
B=catanselfplay-bucket83908e77-vusaorgcj8uf
CL=catan-selfplay
echo "== $run  $(date +%H:%M:%S)"

# ---- learner process and its last iterations
pid=$(ps -eo pid,args | awk -v run="$run" '$2 ~ /python$/ && $3 ~ /az\.py$/ && $4 == "--run" && $5 == run {print $1}')
if [ -n "$pid" ]; then echo "learner: pid $pid, up $(ps -o etime= -p $pid | tr -d ' ')"; else echo "learner: NOT RUNNING"; fi
python3 - "$run" <<'EOF'
import json, os, sys
run = sys.argv[1]
try:
    a = json.load(open(f"runs/{run}/args.json"))
    flags = {k: a[k] for k in ("max_steps", "max_opening_steps", "placement_horizon_turns", "placement_playouts", "reuse", "batch", "lr", "remote_max_shards") if k in a}
    print("flags:", flags)
except FileNotFoundError:
    print("flags: no args.json"); sys.exit()
rows = []
try:
    for l in open(f"runs/{run}/log.jsonl"):
        try: rows.append(json.loads(l))
        except ValueError: pass
except FileNotFoundError:
    print("log: none yet"); sys.exit()
if not rows:
    print("log: no iteration yet"); sys.exit()
reuse, batch, cap = a.get("reuse", 4), a.get("batch", 2048), a.get("max_steps", 0)
print(f"{'iter':>5} {'games':>8} {'new_ex':>8} {'steps':>5} {'loss_pi':>9} {'loss_v':>7} {'tgt_ent':>7} {'surpr':>6} {'lag':>4} {'drop':>4} | {'open_pi':>7} {'open_v':>6}")
for r in rows[-5:]:
    st = round(r.get("new_examples", 0) * reuse / batch); st = min(st, cap) if cap else st
    o = r.get("opening") or {}
    print(f"{r['iter']:>5} {r['games']:>8,} {r.get('new_examples', 0):>8,} {st:>5} {r['loss_pi']:>9.3f} {r['loss_v']:>7.3f} {r.get('target_entropy', 0):>7.2f} {r.get('surprise', 0):>6.3f} "
          f"{str(r.get('remote_lag')):>4} {str(r.get('remote_dropped')):>4} | {o.get('loss_pi', 0):>7.3f} {o.get('loss_v', 0):>6.3f}")
last = rows[-1]
warn = []
if last["loss_pi"] > 5: warn.append(f"DIVERGED: policy loss {last['loss_pi']:.0f}")
if last.get("target_entropy", 1) < 0.2: warn.append(f"TARGETS COLLAPSED (entropy {last['target_entropy']:.2f}): actors may be playing bad weights")
if last.get("new_examples", 0) > 600_000: warn.append(f"SHARD BURST: {last['new_examples']:,} rows in one iteration")
if last.get("remote_dropped"): warn.append(f"dropping shards: {last['remote_dropped']}")
print("WARNINGS: " + ("; ".join(warn) if warn else "none"))
EOF

# ---- control file, shard tags, actors
ctrl=$(aws s3 cp s3://$B/models/control_$run.json - 2>/dev/null)
if [ -n "$ctrl" ]; then
  echo "control: $(echo "$ctrl" | python3 -c "import json,sys; c=json.load(sys.stdin); print('iter', c['iter'], 'model', c['model'].split('/')[-1], 'launch', c.get('launch'))")"
else echo "control: MISSING"; fi
tags=$(aws s3 ls s3://$B/shards/$run/ 2>/dev/null | awk '{print $2}' | tr -d / | tr '\n' ' ')
echo "shards waiting in S3, by model iteration: ${tags:-none}"
latest=$(aws ecr describe-images --repository-name catan-selfplay --image-ids imageTag=latest --query 'imageDetails[0].imageDigest' --output text 2>/dev/null | cut -c8-19)
tasks=$(aws ecs list-tasks --cluster $CL --desired-status RUNNING --query 'taskArns' --output text 2>/dev/null)
if [ -n "$tasks" ]; then
  aws ecs describe-tasks --cluster $CL --tasks $tasks --query 'tasks[].[startedAt,containers[0].imageDigest,overrides.containerOverrides[0].command[-1]]' --output text 2>/dev/null \
    | awk -v L="$latest" '{d=substr($2,8,12); printf "actor: started %s image %s%s arm %s\n", substr($1,12,8), d, (d==L ? "" : " (NOT the latest image " L ")"), $3}'
else echo "actors: none running"; fi
echo "keeper: $(tail -n 1 runs/fleet.out | cut -c1-120)"

# ---- monitors and leftovers
for pat in "score_runs.py openhex_models/ham.ctnn $run" "panel_score.py $run" "agree100.py $run " "value_track.py $run " "arm_checkpoint.py $run "; do
  n=$(ps -eo args | grep -F "$pat" | grep -v grep | grep -v shell-snapshots | grep -vc "bash -c")
  printf "monitor %-28s %s\n" "${pat%% *}" "$n$( [ "$pat" = "arm_checkpoint.py $run " ] && echo ' (3 expected)' || { [ "$n" = 1 ] || echo ' (1 expected)'; })"
done
stale=$(ps -eo pid,args | grep -F "on_launch.sh $run" | grep -v grep | grep -vc shell-snapshots)
[ "$stale" != 0 ] && echo "LEFTOVER launcher shells: $stale"
wrap=$(ps -eo args | grep -F "az.py --run $run" | grep -v grep | grep -c shell-snapshots)
[ "$wrap" != 0 ] && echo "LEFTOVER wrapper shells mentioning the learner: $wrap (they block the launcher's wait loop)"

# ---- yardsticks and local load
echo "Ham:    $(tail -n 1 runs/$run.vs_ham 2>/dev/null | cut -c1-70)"
echo "Gaz:    $(tail -n 1 runs/$run.vs_gaz 2>/dev/null | cut -c1-70)"
echo "panel:  $(tail -n 1 runs/$run.panel100 2>/dev/null | cut -c1-40)"
echo "agree:  $(tail -n 1 runs/$run.agree 2>/dev/null | cut -c1-60)"
echo "value:  $(tail -n 1 runs/$run.valuetrack 2>/dev/null | cut -c1-90)"
echo "local:  load $(cut -d' ' -f1-3 /proc/loadavg) on $(nproc) cores; top: $(ps -eo pcpu,comm --sort=-pcpu | sed -n 2,4p | awk '{printf "%s %s%%  ", $2, $1}')"
