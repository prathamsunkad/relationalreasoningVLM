#!/usr/bin/env bash
#SBATCH --job-name=vlm-train
#SBATCH --account=coc
#SBATCH --qos=coc-ice
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%j-vlm-train.out
#SBATCH --error=logs/%j-vlm-train.err
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm; inference/training must not run on a login node}"
cd "${SLURM_SUBMIT_DIR:?}"
source env.sh
: "${VLM_PYTHON:?Set VLM_PYTHON to the Python executable in your verified environment}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
gpus=${SLURM_GPUS_ON_NODE:-1}
[[ "$gpus" =~ ^[1-8]$ ]] || { echo 'Expected 1-8 GPUs on one node' >&2; exit 2; }
# Opt into the existing container when that is where VLM_PYTHON is installed.
runner=()
if [[ ${VLM_USE_CONTAINER:-0} == 1 ]]; then
    runner=(apptainer exec --nv "$CONTAINER")
fi
if [[ "$gpus" == 1 ]]; then
    train_command=("${runner[@]}" "$VLM_PYTHON" -u lora/train_lora.py "$@")
else
    train_command=("${runner[@]}" "$VLM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$gpus" lora/train_lora.py "$@")
fi

# An absolute deadline includes model loading and validation, not just training.
end_time=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
if [[ -n "$end_time" && "$end_time" != Unknown ]]; then
    export VLM_DEADLINE_EPOCH=$(date -d "$end_time" +%s)
fi
# A sentinel reaches all torchrun workers without terminating the launcher.
VLM_STOP_FILE=$(mktemp)
export VLM_STOP_FILE
rm -f "$VLM_STOP_FILE"
trap 'rm -f "$VLM_STOP_FILE"' EXIT
"${train_command[@]}" &
child=$!
relay() { touch "$VLM_STOP_FILE"; }
trap relay USR1 TERM INT
set +e
while true; do
    wait "$child"
    child_result=$?
    kill -0 "$child" 2>/dev/null || break
done
exit "$child_result"
