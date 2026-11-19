#!/bin/bash
# Clean 60k pipeline, step 2: Stage 1a branch (Abstractor + aux head, slots fine-tuned) trained only on the 60k PGM
# problems LoRA alone 5% saw, from the 60k-only slots. Recipe = aux_ft_q_L6_lr3e-4_ep10 (6 layers, head_dim 32,
# slot_lr 3e-5, warmup 500, cosine from 135k, val every 1k) with --train_samples 60000 and 40 epochs = 150,000 steps,
# the same number of steps as the 10-epoch branch on 240k. best.pt = best validation.
# usage: launch_aux60k.sh <abstractor lr> <new tag> <manifest note>
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -euo pipefail
LR=$1; TAG=$2; NOTE=$3
RUNS=$RELVLM_RUNS
REPO=$RELVLM_REPO
SLOT=$RUNS/s0_qwen_pgm60k/best.pt
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -e "$RUNS/s0_qwen_pgm60k/done.json" ] || { echo "s0_qwen_pgm60k has not finished, refusing"; exit 1; }
EXTRA="--inject interleave --objective aux --lr_schedule cosine --decay_start 135000 --val_every 1000 --layers 6 --lr $LR --finetune_slots --slot_lr 3e-5 --train_samples 60000"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$SLOT TAG=$TAG EPOCHS=40 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=150 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO_ROOT/slurm_portable/relational/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=40 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm60k $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
scontrol update JobId=$JID Nice=1 || true   # tier 2: the 50-epoch runs go first (user, Nov 16)
echo "submitted $JID $TAG (lr $LR)"
