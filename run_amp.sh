#!/usr/bin/env bash
# Person C: mixed-precision extra experiment (compare against weak_N<MAXG> from Person A, same GPU type).
set -e; MAXG=${MAXG:-4}; EP=${EP:-30}
torchrun --standalone --nproc_per_node=$MAXG train_ddp.py --scaling weak --batch-size 128 --epochs $EP --amp --tag weak_amp_N$MAXG
