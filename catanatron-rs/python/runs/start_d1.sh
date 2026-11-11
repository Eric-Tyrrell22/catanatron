#!/bin/bash
# Waits for 26 specialist results, stops the driver, splits 20 teachers / 6
# held out (3 screened + 3 random), and starts distillation run d1.
set -e
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until [ "$(wc -l < runs/sp1/results.jsonl)" -ge 26 ]; do sleep 15; done
P=$(ps -eo pid,args | awk '$2==".venv/bin/python" && $3=="-u" && $4=="specialists.py" {print $1}')
kill -TERM $P
while kill -0 $P 2>/dev/null; do sleep 2; done
echo "$(date +%H:%M) driver stopped at $(wc -l < runs/sp1/results.jsonl) results"
mkdir -p runs/d1
.venv/bin/python - <<'PY'
import json, random
ok = [json.loads(l) for l in open("runs/sp1/results.jsonl")]
ok = [r for r in ok if r["outcome"] == "success"][:26]
rng = random.Random(9131)
hold = rng.sample([r for r in ok if r["group"] == "screened"], 3) + rng.sample([r for r in ok if r["group"] == "random"], 3)
hs = {r["seed"] for r in hold}
with open("runs/d1/teachers.jsonl", "w") as f:
    for r in ok:
        if r["seed"] not in hs:
            f.write(json.dumps(r) + "\n")
with open("runs/d1/holdout.jsonl", "w") as f:
    for r in hold:
        f.write(json.dumps(r) + "\n")
print(f"{len(ok) - len(hold)} teachers, held out: " + ", ".join(f"{r['seed']} ({r['group']}, {r['best_win']:.1%})" for r in hold))
PY
nohup .venv/bin/python az.py --run d1 --init runs/oh2/model.pt --teachers runs/d1/teachers.jsonl --teacher-share 0.3 --symmetry --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 2e-4 --replay 1000000 --replay-device cpu --min-replay 50000 --eval-every 20 --eval-games 0 --eval-search-games 0 --eval-random-games 0 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --opening-share 0.1 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --value-q-weight 0.25 --minutes 600 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 > runs/d1.out 2>&1 &
echo "$(date +%H:%M) d1 started"
