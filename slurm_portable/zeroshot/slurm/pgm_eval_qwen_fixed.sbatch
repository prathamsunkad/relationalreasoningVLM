#!/bin/bash
#SBATCH -J pgm_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 05:00:00
#SBATCH --array=0-27
#SBATCH -o logs/pgm_eval_%A_%a.log

# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

mkdir -p $WORK/Qwen2BInstruct_PGM_test/logs_v2

CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif

apptainer exec --nv \
    --env HF_HOME=$HF_HOME \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python $REPO_ROOT/zeroshot/pgm_infer.py \
    --npz_dir $DATA/test_full/neutral \
    --output_dir $WORK/Qwen2BInstruct_PGM_test/results_v3_fixed \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards 28 \
    --max_new_tokens 30