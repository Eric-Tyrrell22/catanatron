#!/bin/bash
# Frees the disk by deleting the replay buffers (replay.pt, a regenerable cache only a --resume reads)
# of finished runs. Keeps on1 (running arm), rr2 (kept resumable) and any run whose learner is running.
# Snapshots, model.pt, logs and ledgers are untouched.
#   bash runs/free_disk.sh            # delete
#   bash runs/free_disk.sh --dry-run  # list only
cd /home/eric/Desktop/prog/catan/catanatron/catanatron-rs/python
keep="on1 rr2"
freed=0; n=0
for f in runs/*/replay.pt runs/*/replay.pt.tmp runs/*/oreplay.pt.tmp; do
  [ -f "$f" ] || continue
  run=$(basename "$(dirname "$f")")
  case " $keep " in *" $run "*) [ "$(basename "$f")" = "oreplay.pt.tmp" ] || continue;; esac   # a partial .tmp is junk even for kept runs
  ps -eo args | grep -q "[a]z.py --run $run " && { echo "skip $run: learner running"; continue; }
  sz=$(du -m "$f" | cut -f1)
  if [ "${1:-}" = "--dry-run" ]; then echo "would delete $f (${sz} MB)"; else rm -f "$f"; fi
  freed=$((freed + sz)); n=$((n + 1))
done
echo "$([ "${1:-}" = "--dry-run" ] && echo would free || echo freed) $((freed / 1024)) GB in $n files"
df -h . | tail -1
