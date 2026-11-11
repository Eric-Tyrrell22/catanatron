#!/bin/bash
# start_distill.sh <run> <specialists> <teacher share> [init checkpoint]: waits for that many
# specialist results, stops the driver, and starts distillation run <run>
# with every successful specialist as a teacher (no held-out boards).
set -e
RUN=$1; N=$2; SHARE=$3; INIT=${4:-runs/oh2/model.pt}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until [ "$(wc -l < runs/sp1/results.jsonl)" -ge "$N" ]; do sleep 15; done
P=$(ps -eo pid,args | awk '$2==".venv/bin/python" && $3=="-u" && $4=="specialists.py" {print $1}')
kill -TERM $P
while kill -0 $P 2>/dev/null; do sleep 2; done
echo "$(date +%H:%M) driver stopped at $(wc -l < runs/sp1/results.jsonl) results"
mkdir -p runs/$RUN
.venv/bin/python - "$RUN" <<'PY'
import json, sys
run = sys.argv[1]
ok = [json.loads(l) for l in open("runs/sp1/results.jsonl")]
ok = [r for r in ok if r["outcome"] == "success"]
with open(f"runs/{run}/teachers.jsonl", "w") as f:
    for r in ok:
        f.write(json.dumps(r) + "\n")
open(f"runs/{run}/holdout.jsonl", "w").close()
print(f"{len(ok)} teachers ({sum(r['group'] == 'screened' for r in ok)} screened, {sum(r['group'] == 'random' for r in ok)} random)")
PY
nohup .venv/bin/python az.py --run $RUN --init $INIT --teachers runs/$RUN/teachers.jsonl --teacher-share $SHARE --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 2e-4 --replay 1000000 --replay-device cpu --min-replay 50000 --eval-every 20 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --opening-share 0.1 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --value-q-weight 0.25 --minutes 600 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 > runs/$RUN.out 2>&1 &
echo "$(date +%H:%M) $RUN started"
