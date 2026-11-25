#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
set -euo pipefail
: "${CONNECTOR_REPO:?}"
: "${CONNECTOR_PLAN:?}"
: "${CONNECTOR_OUTPUT:?}"
: "${CONNECTOR_MODE:?}"
: "${CONNECTOR_SECONDS:?}"
: "${VLM_PYTHON:?}"
cd "$CONNECTOR_REPO"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export USE_TF=0 USE_FLAX=0 TOKENIZERS_PARALLELISM=false CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONPATH="$CONNECTOR_REPO${PYTHONPATH:+:$PYTHONPATH}"
deadline=$("$VLM_PYTHON" -c 'import time,os; print(time.time()+int(os.environ["CONNECTOR_SECONDS"])-120)')
args=(--mode "$CONNECTOR_MODE" --plan "$CONNECTOR_PLAN" --output "$CONNECTOR_OUTPUT" --deadline "$deadline")
if [[ "$CONNECTOR_MODE" == train ]]; then args+=(--design "${CONNECTOR_DESIGN:?}"); fi
if [[ "${CONNECTOR_RESUME:-0}" == 1 ]]; then args+=(--resume); fi
exec "$VLM_PYTHON" -u tools/train_pretrained_connectors.py "${args[@]}"
