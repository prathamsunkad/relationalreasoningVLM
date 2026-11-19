#!/bin/bash
#SBATCH -J pgm_pc_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
# Best compatible GPU: H200. Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, slow V100/RTX6000.
# Shorter queue: sbatch -C "gpu-h200|gpu-h100" ...   Any compatible: -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH -p ice-gpu
#SBATCH -C gpu-h200
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 05:00:00
#SBATCH --array=0-27
#SBATCH -o logs/eval_%x_%A_%a.log

# Sharded per-candidate test evaluation (28 shards over the sorted file list, like pgm_infer.py). Re-running a shard resumes it.
#   LoRA model : sbatch --export=ALL,MODEL_ID=...,TAG=qwen_neutral_100k,REGIME=neutral,ADAPTER=best $REPO_ROOT/slurm_portable/slurm/eval_array.sbatch
#   zero-shot  : sbatch --export=ALL,MODEL_ID=...,TAG=qwen_zeroshot,REGIME=neutral,ADAPTER=none $REPO_ROOT/slurm_portable/slurm/eval_array.sbatch
# Then:  python lora/eval_vlm.py --merge_dir $RUNS/results/<TAG>_<REGIME> --expected_n 200000
# Sanity run on the first 2000 problems with blank context:  EXTRA="--context_mode blank --max_examples 2000" (add "--panel_px N" if training used it) with --array=0-0

# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$REPO_ROOT
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-$DATA}
RUNS=${RUNS:-$LORA_RUNS}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG}
REGIME=${REGIME:-neutral}
ADAPTER=${ADAPTER:-best}                  # best | last | final | none
EXTRA=${EXTRA:-}
OUT=$RUNS/results/${TAG}_${REGIME}${SUFFIX:-}

mkdir -p $RUNS/logs $OUT $RUNS/filecache
cd $REPO
ADAPTER_ARG=""
if [ "$ADAPTER" != "none" ]; then
    ADAPTER_ARG="--adapter $RUNS/$TAG/$ADAPTER"
    if [ ! -d "$RUNS/$TAG/$ADAPTER" ]; then
        echo "adapter $RUNS/$TAG/$ADAPTER does not exist (training not finished? ADAPTER=final only exists after the last step)"; exit 1
    fi
fi
apptainer exec --nv \
    --env HF_HOME=$HF_HOME \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python lora/eval_vlm.py \
    --model_id $MODEL_ID $ADAPTER_ARG \
    --npz_dir $DATA/test_full/$REGIME --file_cache_dir $RUNS/filecache/test_$REGIME \
    --out_dir $OUT --shard_idx $SLURM_ARRAY_TASK_ID --num_shards 28 \
    --batch_size 4 --num_workers 4 $EXTRA
