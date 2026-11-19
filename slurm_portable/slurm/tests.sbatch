#!/bin/bash
#SBATCH -J pgm_pc_tests
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o logs/tests_%j.log

# CPU test suite (random-weight tiny models, no downloads):  sbatch $REPO_ROOT/slurm_portable/slurm/tests.sbatch
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$REPO_ROOT
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}

cd $REPO
apptainer exec \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python tests/test_pipeline.py
