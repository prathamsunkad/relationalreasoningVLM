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
: "${CONNECTOR_REPO:?Set the frozen source snapshot}"
: "${CONNECTOR_PLAN:?Set the resolved pilot plan}"
: "${CONNECTOR_OUTPUT:?Set a fresh output directory}"
: "${VLM_PYTHON:?Set the verified Python interpreter}"
: "${SLURM_JOB_ID:?GPU allocation required}"
cd "$CONNECTOR_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export USE_TF=0 USE_FLAX=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CONNECTOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
# Start the deadline BEFORE GPU-library import and correctness tests.
deadline=$("$VLM_PYTHON" -c 'import time; print(time.time()+6900)')
"$VLM_PYTHON" -m unittest discover -s tests -p test_pretrained_connectors.py -v
exec "$VLM_PYTHON" -u tools/pretrained_connector_pilot.py \
 --plan "$CONNECTOR_PLAN" --output "$CONNECTOR_OUTPUT" --deadline "$deadline"
