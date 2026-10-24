#!/usr/bin/env bash
# Person A: weak scaling (per-GPU batch fixed at 128). Usage: MAXG=4 EP=30 ./run_weak.sh
set -e; MAXG=${MAXG:-4}; EP=${EP:-30}
for N in $(seq 1 $MAXG); do
  torchrun --standalone --nproc_per_node=$N train_ddp.py --scaling weak --batch-size 128 --epochs $EP --tag weak_N$N
done
