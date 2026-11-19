#!/bin/bash
#SBATCH -J relvlm_s1
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C gpu-h200
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 05:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/stage1_%x_%j.log

# Stage 1 (frozen VLM) or Stage 2 (EXTRA="--lora_r 16 --init_from .../best.pt"). Resumable: resubmit / chain.
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,SLOT=$RUNS/s0_qwen_pgm/best.pt,TAG=s1_qwen_inter,EXTRA="--inject interleave" slurm/stage1.sbatch
# Chain:  J=$(sbatch --parsable ...); sbatch --dependency=afterany:$J ...
REPO=${REPO:-/home/hice1/psunkad3/scratch/relvlm}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/relvlm_runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
SLOT=${SLOT:?set SLOT=<stage0 best.pt>}
TAG=${TAG:?set TAG}
EPOCHS=${EPOCHS:-1}
TRAIN=${TRAIN:-"--train pgm=matrix:$DATA/train_full/neutral"}
VAL=${VAL:-"--val pgm=matrix:$DATA/val_full/neutral"}
EXTRA=${EXTRA:-}
# the 1000 shared validation problems used by the LoRA baseline (same IDs => comparable curves)
VAL_LIST=${VAL_LIST:-/home/hice1/psunkad3/scratch/vlm_lora_pc_runs/filecache/neutral/val1000_shared.tsv}
VAL_ARG=""; [ -f "$VAL_LIST" ] && VAL_ARG="--val_list $VAL_LIST"
mkdir -p $RUNS/logs $RUNS/$TAG $RUNS/filecache
cd $REPO
apptainer exec --nv --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
    python train_stage1.py --model_id $MODEL_ID --slot_ckpt $SLOT --out_dir $RUNS/$TAG $TRAIN $VAL $VAL_ARG \
    --file_cache_dir $RUNS/filecache --epochs $EPOCHS --time_limit_min ${TIME_LIMIT_MIN:-285} $EXTRA
RC=$?
# Long runs under the 50-submitted-jobs limit: with AUTO_RESUBMIT=1 a clean stop resubmits this job until final.pt exists.
# SB_OPTS holds the sbatch options to reuse (-J, -t, -C, --exclude); a crash (RC != 0) does not resubmit.
if [ $RC -eq 0 ] && [ -n "${AUTO_RESUBMIT:-}" ] && [ ! -f $RUNS/$TAG/final.pt ] && [ "${ATTEMPT:-1}" -lt "${MAX_ATTEMPTS:-60}" ]; then
    export ATTEMPT=$((${ATTEMPT:-1} + 1))
    for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_'); do unset "$v"; done
    # Opt-in per-run override: if $RUNS/$TAG/sb_opts exists, its one line replaces SB_OPTS for the next segments
    # (e.g. to add --prefer=gpu-h100|gpu-h200 to a running chain without restarting it).
    if [ -s $RUNS/$TAG/sb_opts ]; then export SB_OPTS="$(head -1 $RUNS/$TAG/sb_opts)"; echo "[chain] SB_OPTS from $RUNS/$TAG/sb_opts"; fi
    sbatch --export=ALL ${SB_OPTS:-} $REPO/slurm/stage1.sbatch
fi
exit $RC
