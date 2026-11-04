#!/bin/bash
#SBATCH -J relvlm_datacpu
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/datacpu_%j.log

# Run our own data scripts in the container:  sbatch --export=ALL,CMD="datasets/verify.py 'art/*/*.jsonl'" slurm/data_cpu.sbatch
REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning/relational}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export RELVLM_DATA=${RELVLM_DATA:-/home/hice1/psunkad3/scratch/relvlm_data}
cd $REPO
for c in "${CMD:?set CMD}"; do
  apptainer exec --env RELVLM_DATA=$RELVLM_DATA "$CONTAINER" bash -c "python -W ignore $c"
done
