#!/bin/bash
# Retire Breeze between games and put Ham on the OpenHex ladder.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
P=$(ps -eo pid,args | awk '$4=="openhex_live.py" && $6=="Breeze" {print $1}')
start=$(wc -l < runs/breeze.out)
echo "$(date +%H:%M) waiting for Breeze (pid $P) to finish its current game"
# A finished game ends with "over; winner"; then Breeze pauses 60 s before queueing again.
until tail -n +$((start + 1)) runs/breeze.out | grep -q "over; winner"; do sleep 5; done
kill -TERM $P; while kill -0 $P 2>/dev/null; do sleep 1; done
echo "$(date +%H:%M) Breeze stopped: $(tail -n +$((start + 1)) runs/breeze.out | grep 'over; winner' | tail -1)"
.venv/bin/python openhex_live.py register Ham
nohup .venv/bin/python -u openhex_live.py play Ham "MCTS:100000,t10000,c1,f0.25,m0.1,s0.3/150:$PWD/openhex_models/ham.ctnn" >> runs/ham.out 2>&1 &
sleep 20; echo "$(date +%H:%M) Ham started:"; tail -3 runs/ham.out
