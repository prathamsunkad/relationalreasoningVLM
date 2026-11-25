#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-6
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
set -euo pipefail
: "${VLM_REPO:?Set immutable source directory}"
: "${VLM_PYTHON:?Set verified Python interpreter}"
: "${VLM_A_PLAN:?Set resolved training plan}"
: "${VLM_A_OUTPUT:?Set new campaign output directory}"
: "${VLM_TRAIN_SECONDS:?Set authorized allocation seconds minus cleanup reserve}"
: "${SLURM_JOB_ID:?GPU allocation required}"
: "${SLURM_ARRAY_TASK_ID:?Seven-design array required}"
cd "$VLM_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$VLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
designs=(early late joint dual memory_projected rca_plugin memory_contextual)
[[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-6]$ ]]
design=${designs[$SLURM_ARRAY_TASK_ID]}
deadline=$("$VLM_PYTHON" -c 'import os,time; print(time.time()+int(os.environ["VLM_TRAIN_SECONDS"]))')
exec "$VLM_PYTHON" -u tools/train_staged_abstractor.py --plan "$VLM_A_PLAN" \
  --design "$design" --output "$VLM_A_OUTPUT/$design" --deadline "$deadline"
