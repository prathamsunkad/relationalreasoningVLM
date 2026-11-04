#!/bin/bash
#SBATCH -J pgm_lora_tests
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_runs/logs/tests_%j.log

# CPU test suite (random-weight tiny models, no downloads):  sbatch slurm/tests.sbatch
REPO=${REPO:-/home/hice1/psunkad3/scratch/vlm_lora}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}

cd $REPO
apptainer exec \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python tests/test_pipeline.py
