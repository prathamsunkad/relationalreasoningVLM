#!/usr/bin/env bash
#SBATCH --job-name=vlm-eval
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=10:00:00
#SBATCH --output=logs/%A_%a-vlm-eval.out
#SBATCH --error=logs/%A_%a-vlm-eval.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
cd "${SLURM_SUBMIT_DIR:?}"
source env.sh
: "${VLM_PYTHON:?Set VLM_PYTHON to your verified Python executable}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
runner=()
if [[ ${VLM_USE_CONTAINER:-0} == 1 ]]; then runner=(apptainer exec --nv "$CONTAINER"); fi
# Arrays must be zero-based, contiguous, and step one (e.g. --array=0-3).
index=${SLURM_ARRAY_TASK_ID:-0}
count=${SLURM_ARRAY_TASK_COUNT:-1}
if [[ ${SLURM_ARRAY_TASK_MIN:-0} != 0 || ${SLURM_ARRAY_TASK_STEP:-1} != 1 ]]; then
    echo 'Use a contiguous zero-based array' >&2; exit 2
fi
exec "${runner[@]}" "$VLM_PYTHON" -u lora/eval_vlm.py --shard_idx "$index" --num_shards "$count" "$@"
