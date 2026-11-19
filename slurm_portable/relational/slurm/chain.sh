#!/bin/bash
# Submit N copies of a resumable job, each starting after the previous one ends (any exit state).
#   bash slurm/chain.sh 3 --export=ALL,MODEL_ID=...,TAG=... slurm/stage0.sbatch
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

N=$1; shift
J=$(sbatch --parsable "$@") || exit 1
echo "submitted $J"
for ((i = 2; i <= N; i++)); do
  J=$(sbatch --parsable --dependency=afterany:$J "$@") || exit 1
  echo "submitted $J (after the previous)"
done
