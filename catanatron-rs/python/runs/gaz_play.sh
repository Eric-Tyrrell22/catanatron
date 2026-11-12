#!/bin/bash
# Keeps Gaz (on3's final networks, opening network + relation block) on the OpenHex ladder with 5 s per move.
# The bot abandons a game and exits if its mirror ever disagrees with the server (OutOfStep); this loop
# restarts it after 3 s so the ladder presence survives that, at the cost of that one game.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
export PYTHONPATH=/home/eric/Desktop/prog/catan/catanatron-opening/catanatron-rs/target/pylib
KEY="MCTS:100000,t5000,c1,f0.25,m0.1,s0.3/150:$PWD/openhex_models/gaz.ctnn@$PWD/openhex_models/gaz_opening.ctnn"
while true; do
  echo "$(date '+%m-%d %H:%M') starting Gaz"
  .venv/bin/python -u openhex_live.py play Gaz "$KEY"
  echo "$(date '+%m-%d %H:%M') Gaz exited ($?); restarting in 3 s"
  sleep 3
done
