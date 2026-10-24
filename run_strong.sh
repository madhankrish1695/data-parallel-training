#!/usr/bin/env bash
# Person B: strong scaling (global batch fixed at 512). Usage: MAXG=4 EP=30 ./run_strong.sh
set -e; MAXG=${MAXG:-4}; EP=${EP:-30}
for N in $(seq 1 $MAXG); do
  torchrun --standalone --nproc_per_node=$N train_ddp.py --scaling strong --batch-size 512 --epochs $EP --tag strong_N$N
done
