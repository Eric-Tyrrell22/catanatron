#!/bin/bash
# sb1's replay -> fine-tune control and branch -> each network-only vs o4 (10k games) once the CPU is free.
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
until grep -q "^done" runs/sb1.out; do sleep 30; done
echo "$(date +%H:%M) sb1 done; fine-tuning on its replay"
.venv/bin/python -u runs/spot_finetune_check.py runs/sb1/replay.pt 20000 2>&1 | grep -v Warning
until ! ps -eo args | grep -q "[s]creen1.py\|[v]s_ref10k.py sb1"; do sleep 30; done
for name in control branch; do
  .venv/bin/python -c "
import os, catanatron_rs
from catanatron_env import OPENHEX_1V1
(w, l), _ = catanatron_rs.play_bots(['NN:' + os.path.abspath('runs/ft_$name.ctnn'), 'NN:' + os.path.abspath('runs/hp1_pool/o4_final.ctnn')], 10000, 70_000, 14, map_type='BASE', **OPENHEX_1V1)
print(f'$name vs o4 (network only, 10k games): {w/10000:.1%} ±0.5', flush=True)"
done
echo "$(date +%H:%M) done"
