#!/bin/bash
#SBATCH -J relvlm_data
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 08:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/data_%x_%j.log

# Download + convert public datasets (data only):  sbatch --export=ALL,WHAT="vsr whatsup cvbench blink mmvp coco clevr" slurm/data_downloads.sbatch
REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning/relational}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export RELVLM_DATA=${RELVLM_DATA:-/home/hice1/psunkad3/scratch/relvlm_data}
WHAT=${WHAT:-"vsr whatsup cvbench blink mmvp coco clevr"}
mkdir -p $RELVLM_DATA/_hf_cache
cd $REPO
# HF cache on scratch: home has a 30 GB quota
apptainer exec --env HF_HOME=$RELVLM_DATA/_hf_cache --env RELVLM_DATA=$RELVLM_DATA "$CONTAINER" \
    python -W ignore datasets/get_downloads.py $WHAT
