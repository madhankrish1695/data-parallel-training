#!/usr/bin/env bash
# run_experiments.sh - Runs weak scaling, strong scaling, and AMP comparison, then generates summary & plots.
set -e

# Detect available GPUs if MAXG is not explicitly provided
if [ -z "$MAXG" ]; then
  if command -v nvidia-smi &> /dev/null; then
    MAXG=$(nvidia-smi -L | wc -l)
  else
    MAXG=2
  fi
fi
EP=${EP:-30}

echo "=== MLSO DDP Experiment Suite ==="
echo "Target GPUs (MAXG): $MAXG"
echo "Epochs per run (EP): $EP"
echo "================================="

echo "--> Step 1/3: Running Weak Scaling (N=1..$MAXG)..."
MAXG=$MAXG EP=$EP ./run_weak.sh

echo "--> Step 2/3: Running Strong Scaling (N=1..$MAXG)..."
MAXG=$MAXG EP=$EP ./run_strong.sh

echo "--> Step 3/3: Running Mixed-Precision AMP Run (N=$MAXG)..."
MAXG=$MAXG EP=$EP ./run_amp.sh

echo "--> Generating Analysis & Plots..."
python3 analyze.py

echo "=== All experiments completed successfully! Results in ./results ==="
