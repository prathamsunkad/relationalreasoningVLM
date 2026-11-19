#!/bin/bash
# Phase 1 "LoRA first, then the branch": start from LoRA alone at its best (5% data, step 14k, 53.4% val) plus the
# aligned 50-epoch branch. Steps 0-500: LoRA frozen, only the connector learns to talk to the LoRA-tuned LLM;
# steps 500-3500: LoRA (2e-5, about its rate at step 14k) and connector train together, cosine to 10%.
# Abstractor and slots stay frozen. Same 60k problems as the LoRA-alone run. Control = same with the branch always
# off (--branch_dropout 1.0, validated off too), so the branch is the only difference.
# usage: launch_lf2.sh <int|fuse> <aligned init .pt> <new tag> <manifest note>   (hybrid only: the branch-off control does not depend on the branch, so lora_lf_q_{int,fuse}_ft50_ctrl are reused; tier 2)
set -euo pipefail
MODE=$1; SRC=$2; TAG=$3; NOTE=$4
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
SLOT_CKPT=${SLOT_CKPT:-$RUNS/s0_qwen_pgm/best.pt}   # opt-in override (clean 60k runs use s0_qwen_pgm60k)
PRIO=${PRIO:-1}
REPO=/home/hice1/psunkad3/scratch/relvlm_ft
ADAPTER=/home/hice1/psunkad3/scratch/vlm_lora_pc_runs/qwen_pc_neutral5pct_recipe/best
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -f "$SRC" ] || { echo "no $SRC"; exit 1; }
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
mkdir -p "$RUNS/$TAG"
INIT=$RUNS/$TAG/$(basename $SRC); cp "$SRC" "$INIT"; cmp "$SRC" "$INIT"
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT --freeze_abstractor \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_from $ADAPTER --freeze_lora_steps 500 --lora_lr 2e-5 \
--lr 3e-5 --proj_lr 1e-4 --train_samples 60000 --max_steps 3500 --warmup_steps 50 --lr_schedule cosine \
--decay_start 500 --val_every 500 --save_every 500"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$SLOT_CKPT TAG=$TAG EPOCHS=4 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=20 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=4 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=$(basename $(dirname $SLOT_CKPT)) (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" "$PRIO" <<'EOF'
import json, sys, shutil, time
p = "/home/hice1/psunkad3/scratch/relvlm_runs/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2])
if "ft276k" not in d["_about"]:
    d["_about"] += " 2026-11-16 (user: 'Yes, queue it'): LoRA-first with the step-276k branch (align_q_*_L6ft276k_lr3e-4, then lora_lf_q_*_ft276k) at 1, below the 50-epoch runs; each alignment drops to 2 once its LoRA-first run is launched."
json.dump(d, open(p, "w"), indent=2)
EOF
scontrol update JobId=$JID Nice=$PRIO || true
echo "submitted $JID $TAG ($MODE) from $(basename $SRC)"
