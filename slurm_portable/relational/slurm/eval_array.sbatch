#!/bin/bash
#SBATCH -J relvlm_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 05:00:00
#SBATCH --array=0-27
#SBATCH -o logs/eval_%x_%A_%a.log

# 28-shard test eval, then:  python eval.py --merge_dir $RUNS/results/<TAG>_<NAME> --expected_n 200000
#   sbatch --export=ALL,TAG=s1_qwen_inter,SOURCE=neutral=matrix:$DATA/test_full/neutral slurm/eval_array.sbatch
#   sanity: EXTRA="--context_mode blank --max_examples 100" (should fall to ~12.5%)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-$DATA}
RUNS=$RELVLM_RUNS
TAG=${TAG:?set TAG}
CKPT=${CKPT:-best.pt}
SOURCE=${SOURCE:-neutral=matrix:$DATA/test_full/neutral}
NAME=${SOURCE%%=*}
EXTRA=${EXTRA:-}
OUT=$RUNS/results/${TAG}_${NAME}${SUFFIX:-}
mkdir -p $RUNS/logs $OUT
cd $REPO
apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
    python eval.py --ckpt $RUNS/$TAG/$CKPT --source "$SOURCE" --out_json $OUT/shard${SLURM_ARRAY_TASK_ID}.json \
    --shard_idx $SLURM_ARRAY_TASK_ID --num_shards ${SLURM_ARRAY_TASK_COUNT:-28} --file_cache_dir $RUNS/filecache $EXTRA
