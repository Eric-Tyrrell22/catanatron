#!/bin/bash
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until grep -q "^done" runs/screen2.out; do sleep 15; done
.venv/bin/python -u runs/screen3.py 400 12 93000 > runs/screen3.out 2>&1
