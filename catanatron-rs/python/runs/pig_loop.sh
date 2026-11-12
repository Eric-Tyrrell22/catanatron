#!/bin/bash
# Keeps a Pig match going: waits for any running pig_match.py to finish, then plays batch after batch of 100 games
# with a fresh seed base each time (new boards), appending to the same results file so the /pig page's tally
# grows without a break. Stop it by killing this script and the running pig_match.py.
#   bash runs/pig_loop.sh <label, e.g. on6_random> <net key> [parallel=12] [first seed base=7101]
set -uo pipefail
label=$1; key=$2; par=${3:-12}; seed=${4:-7101}
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
export PYTHONPATH=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/target/pylib
until ! ps -eo args | grep -q "[p]ig_match.py"; do sleep 30; done
while true; do
  echo "$(date +%H:%M) next batch: 100 games from seed $seed, $par at once" >> runs/pig_$label.out
  .venv/bin/python pig_match.py 100 "$key" --parallel $par --seed $seed --out runs/pig_$label.jsonl >> runs/pig_$label.out 2>&1
  seed=$((seed + 100))
done
