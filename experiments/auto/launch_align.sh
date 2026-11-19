#!/bin/bash
# Stage 1 alignment: frozen LLM, Abstractor, aux head and slots; only the connector trains. Same recipe as
# align_q_{int,fuse}_L6ft_lr3e-4 (Nov 14), with a different Stage 1a branch checkpoint.
# usage: launch_align.sh <int|fuse> <branch .pt> <new tag> <manifest note>
set -euo pipefail
MODE=$1; INIT=$2; TAG=$3; NOTE=$4
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
REPO=/home/hice1/psunkad3/scratch/relvlm_ft
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -f "$INIT" ] || { echo "no $INIT"; exit 1; }
mkdir -p "$RUNS/$TAG"
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT --freeze_abstractor \
--lr 3e-4 --warmup_steps 200 --lr_schedule cosine --max_steps 10000 --val_every 1000 --proj_lr 3e-4"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$RUNS/s0_qwen_pgm/best.pt TAG=$TAG EPOCHS=1 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=60 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=1 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" <<'EOF'
import json, sys, shutil, time
p = "/home/hice1/psunkad3/scratch/relvlm_runs/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = 0
if "ft50" not in d["_about"]:
    d["_about"] += (" 2026-11-15 (user: 'Yes, both'): the stronger-branch comparison (align_q_*_L6ft50_lr3e-4, then "
                    "lora_w5d_q_*_L6ft50) starts at 0; each alignment drops to 1 once its Stage 2 is launched.")
json.dump(d, open(p, "w"), indent=2)
EOF
scontrol update JobId=$JID Nice=0 || true
echo "submitted $JID $TAG from $(basename $INIT)"
