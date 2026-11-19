#!/bin/bash
#SBATCH -J pgm_bundle_check
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --exclude=atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0
#SBATCH --mem=48G
#SBATCH -t 00:30:00
#SBATCH -o check_%j.log
# Verifies the installation: scores the sample problems with the final adapters (Qwen epoch 5, InternVL epoch 6) and
# compares with the scores recorded in the original environment. ~5 min. Run from inside the bundle folder:
#   sbatch check.sbatch        then look for "ALL CHECKS PASSED" in check_<jobid>.log
# (the excluded nodes had a broken or overheating GPU in Oct 2026)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
bash run.sh check
