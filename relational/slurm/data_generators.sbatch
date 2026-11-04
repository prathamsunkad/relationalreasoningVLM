#!/bin/bash
#SBATCH -J relvlm_gen
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/gen_%j.log

# Generators that run third-party code: SVRT (Fleuret's C++ generator, compiled) and I-RAVEN (official generator,
# ported from Python 2).   sbatch --export=ALL,WHAT="svrt iraven" slurm/data_generators.sbatch
REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning/relational}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export RELVLM_DATA=${RELVLM_DATA:-/home/hice1/psunkad3/scratch/relvlm_data}
WHAT=${WHAT:-"svrt iraven"}
cd $REPO
RUN="apptainer exec --env RELVLM_DATA=$RELVLM_DATA --env TMPDIR=${TMPDIR:-/tmp} $CONTAINER python -W ignore"
set -e
for w in $WHAT; do
  case $w in
    svrt)   $RUN datasets/make_svrt.py ;;
    iraven) $RUN datasets/make_iraven.py --smoke && $RUN datasets/make_iraven.py --num_samples 10000 --workers 7 ;;
  esac
done
