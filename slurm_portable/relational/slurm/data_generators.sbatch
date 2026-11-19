#!/bin/bash
#SBATCH -J relvlm_gen
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/gen_%j.log

# Generators that run third-party code: SVRT (Fleuret's C++ generator, compiled) and I-RAVEN (official generator,
# ported from Python 2).   sbatch --export=ALL,WHAT="svrt iraven" slurm/data_generators.sbatch
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export RELVLM_DATA=${RELVLM_DATA:-$RELVLM_DATA}
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
