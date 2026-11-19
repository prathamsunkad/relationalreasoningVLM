#!/usr/bin/env bash
#SBATCH --job-name=vlm-integration-smoke
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:20:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${VLM_REPO:?Set VLM_REPO to the pinned integration checkout}"
: "${VLM_PYTHON:?Set VLM_PYTHON to the verified interpreter}"
: "${VLM_SMOKE_DATA:?Set VLM_SMOKE_DATA to small train.jsonl / validation.jsonl manifests}"
: "${VLM_SMOKE_OUTPUT:?Use a fresh output directory}"
cd "$VLM_REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1 TQDM_DISABLE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ -f SOURCE_SHA256SUMS ]]; then sha256sum --quiet -c SOURCE_SHA256SUMS; fi
nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv
exec "$VLM_PYTHON" -u tools/gpu_smoke.py --train_manifest "$VLM_SMOKE_DATA/train.jsonl" \
  --val_manifest "$VLM_SMOKE_DATA/validation.jsonl" --output "$VLM_SMOKE_OUTPUT" "$@"
