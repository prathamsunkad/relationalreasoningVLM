#!/bin/bash
#SBATCH -J relvlm_datacpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o logs/datacpu_%j.log

# Run our own data scripts in the container:  sbatch --export=ALL,CMD="datasets/verify.py 'art/*/*.jsonl'" slurm/data_cpu.sbatch
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export RELVLM_DATA=${RELVLM_DATA:-$RELVLM_DATA}
cd $REPO
for c in "${CMD:?set CMD}"; do
  apptainer exec --env RELVLM_DATA=$RELVLM_DATA "$CONTAINER" bash -c "python -W ignore $c"
done
