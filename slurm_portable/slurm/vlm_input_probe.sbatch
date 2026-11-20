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
: "${VLM_REPO:?Set VLM_REPO to the frozen source checkout}"
: "${VLM_PYTHON:?Set VLM_PYTHON to the verified Python environment}"
: "${VLM_INPUT_PLAN:?Set VLM_INPUT_PLAN to a resolved plan JSON}"
: "${VLM_INPUT_OUTPUT:?Set VLM_INPUT_OUTPUT to a new output folder}"
: "${SLURM_JOB_ID:?GPU allocation required}"
cd "$VLM_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$VLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
deadline=$("$VLM_PYTHON" -c 'import time; print(time.time()+7020)')
"$VLM_PYTHON" -m unittest discover -s tests -p test_input_probe.py -v
exec "$VLM_PYTHON" -u tools/input_representation_pilot.py \
  --plan "$VLM_INPUT_PLAN" --output "$VLM_INPUT_OUTPUT" --deadline "$deadline"
