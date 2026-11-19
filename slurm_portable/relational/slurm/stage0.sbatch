#!/bin/bash
#SBATCH -J relvlm_s0
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, slow V100/RTX6000.
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 05:00:00
#SBATCH -o logs/stage0_%x_%j.log

# Stage 0: slot attention on the VLM's frozen ViT features (one per backbone). Resumable: resubmit / chain.
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=s0_qwen_pgm slurm/stage0.sbatch
#   sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=s0_ivl1b_pgm slurm/stage0.sbatch
#   mixture: TRAIN="--train pgm=matrix:$DATA/train_full/neutral:0.5 --train clevr=jsonl:/path/clevr_art_train.jsonl:0.5"
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-$DATA}
RUNS=$RELVLM_RUNS
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG}
TRAIN=${TRAIN:-"--train pgm=matrix:$DATA/train_full/neutral"}
VAL=${VAL:-"--val pgm=matrix:$DATA/val_full/neutral"}
EXTRA=${EXTRA:-}            # e.g. "--num_slots 16 --slot_dim 32 --grid 12 --max_steps 100000"
mkdir -p $RUNS/logs $RUNS/$TAG $RUNS/filecache
cd $REPO
apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
    python train_stage0.py --model_id $MODEL_ID --out_dir $RUNS/$TAG $TRAIN $VAL \
    --file_cache_dir $RUNS/filecache --time_limit_min ${TIME_LIMIT_MIN:-285} $EXTRA
RC=$?
# With AUTO_RESUBMIT=1 a clean stop resubmits this job (SB_OPTS = its sbatch options) until done.json exists.
if [ $RC -eq 0 ] && [ -n "${AUTO_RESUBMIT:-}" ] && [ ! -f $RUNS/$TAG/done.json ] && [ "${ATTEMPT:-1}" -lt "${MAX_ATTEMPTS:-60}" ]; then
    export ATTEMPT=$((${ATTEMPT:-1} + 1))
    for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_'); do unset "$v"; done
    sbatch --export=ALL -o "$REPO_ROOT/logs/%x_%j.log" ${SB_OPTS:-} $REPO_ROOT/slurm_portable/relational/slurm/stage0.sbatch
fi
exit $RC
