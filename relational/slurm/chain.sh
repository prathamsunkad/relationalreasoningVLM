#!/bin/bash
# Submit N copies of a resumable job, each starting after the previous one ends (any exit state).
#   bash slurm/chain.sh 3 --export=ALL,MODEL_ID=...,TAG=... slurm/stage0.sbatch
N=$1; shift
J=$(sbatch --parsable "$@") || exit 1
echo "submitted $J"
for ((i = 2; i <= N; i++)); do
  J=$(sbatch --parsable --dependency=afterany:$J "$@") || exit 1
  echo "submitted $J (after the previous)"
done
