#!/bin/bash
#SBATCH -J relvlm_tests
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/tests_%j.log

# CPU tests with random-weight tiny models (no downloads):  sbatch slurm/tests.sbatch   [ARGS="cli grads" to filter]
REPO=${REPO:-/home/hice1/psunkad3/scratch/relvlm}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
cd $REPO
apptainer exec --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" python tests/test_tiny.py $ARGS
