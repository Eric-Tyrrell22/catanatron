#!/bin/bash
# When d2 has written its iteration-200 snapshot: stop d2 and resume the
# specialist driver (round 1 boards, then round1b past 100).
set -e
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until [ -f runs/d2/snap_000200.ctnn ]; do sleep 15; done
sleep 10
P=$(ps -eo pid,args | awk '$2==".venv/bin/python" && $3=="az.py" && $5=="d2" {print $1}')
kill -TERM $P
while kill -0 $P 2>/dev/null; do sleep 2; done
rm -f runs/d2/replay.pt
echo "$(date +%H:%M) d2 stopped: $(tail -1 runs/d2.out | cut -c1-80)"
nohup .venv/bin/python -u specialists.py --boards runs/boards/round1.json --prefix sp1 --base-pt runs/oh2/model.pt --base-net openhex_models/breeze.ctnn -- --openhex-1v1 --opponent-hands --opponent-devs --extra-info --spot-features --games 2 --sims-full 400 --sims-fast 50 --sims-placement 2400 --c-puct 1.0 --fpu 0.25 --lr 2e-4 --replay 500000 --replay-device cpu --min-replay 20000 --threads 2 --device cuda --bf16 --endgame-vp 13 --endgame-branches 1 --opening-branches 1 --placement-forks 1 --placement-temp 2.0 --placement-uniform 0.25 --opening-share 0.1 --luck --margin-weight 0.1 --speed-weight 0.3 --speed-turns 150 --aux-weight 0.5 --value-q-weight 0.25 --minutes 600 --remote-bucket catanselfplay-bucket83908e77-vusaorgcj8uf --remote-games-per-core 4 --remote-max-shards 6 --remote-max-lag 8 >> runs/sp1.out 2>&1 &
echo "$(date +%H:%M) specialist driver resumed"
