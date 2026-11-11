#!/bin/bash
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until grep -q "^done" runs/screen1.out; do sleep 15; done
.venv/bin/python -u runs/screen2.py 400 12 92000 > runs/screen2.out 2>&1
