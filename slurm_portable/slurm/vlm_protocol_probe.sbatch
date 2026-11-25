#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
set -euo pipefail
: "${VLM_REPO:?Set the frozen source directory}"
: "${VLM_PYTHON:?Set the verified Python interpreter}"
: "${VLM_PROTOCOL_PLAN:?Set the resolved plan}"
: "${VLM_PROTOCOL_OUTPUT:?Set the new output directory}"
: "${SLURM_JOB_ID:?GPU allocation required}"
cd "$VLM_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$VLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
deadline=$("$VLM_PYTHON" -c 'import time; print(time.time()+6960)')
"$VLM_PYTHON" -m unittest discover -s tests -p test_protocol_probe.py -v
exec "$VLM_PYTHON" -u tools/protocol_comparison_pilot.py \
  --plan "$VLM_PROTOCOL_PLAN" --output "$VLM_PROTOCOL_OUTPUT" --deadline "$deadline"
