#!/bin/bash
# Revised Stage 2, warm start, 5% data, branch dropout: LoRA (baseline recipe) + the aligned branch at 3e-5.
# Same recipe as lora_w5d_q_{int,fuse}_L6ft (Nov 15).
# usage: launch_s2wd.sh <int|fuse> <init .pt to copy> <new tag> <dropout> <manifest note>
set -euo pipefail
MODE=$1; INIT_SRC=$2; TAG=$3; P=$4; NOTE=$5
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
SLOT_CKPT=${SLOT_CKPT:-$RUNS/s0_qwen_pgm/best.pt}   # opt-in override (clean 60k runs use s0_qwen_pgm60k)
PRIO=${PRIO:-0}
REPO=/home/hice1/psunkad3/scratch/relvlm_ft
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
mkdir -p "$RUNS/$TAG"
INIT=$RUNS/$TAG/$(basename $INIT_SRC)
cp "$INIT_SRC" "$INIT"
cmp "$INIT_SRC" "$INIT"
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_lr 1e-4 --lr 3e-5 --proj_lr 3e-5 \
--train_samples 60000 --max_steps 15000 --warmup_steps 20 --lr_schedule cosine --decay_start 11250 --val_every 1000 --branch_dropout $P"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$SLOT_CKPT TAG=$TAG EPOCHS=4 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=60 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=4 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=$(basename $(dirname $SLOT_CKPT)) (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" "$PRIO" <<'EOF'
import json, sys, shutil, time
p = "/home/hice1/psunkad3/scratch/relvlm_runs/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2])
json.dump(d, open(p, "w"), indent=2)
EOF
scontrol update JobId=$JID Nice=$PRIO || true
echo "submitted $JID $TAG from $(basename $INIT_SRC) dropout $P"
