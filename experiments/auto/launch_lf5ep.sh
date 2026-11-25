#!/bin/bash
# 5% per-epoch (Nov 24, user: "compare the 5% LoRA at different epochs and add the 5% abstractor"; "queue any runs that you feel are important"):
# the H12 recipe (as launch_lf20.sh) on an epoch snapshot of the 5% LoRA rerun (qwen_pc_neutral5pct_ep/step_N), with the
# 60k-only branch; train_samples 60000 = the same 60k puzzles the branch, its slots and the adapter saw (matched 5% data).
# Steps 0-500 LoRA frozen; 500-3500 LoRA 2e-5 + connector; Abstractor and slots frozen.
# usage: launch_lf20.sh <int|fuse> <aligned init .pt> <adapter dir> <new tag> <branch_dropout 0|1.0> <manifest note>
set -euo pipefail
MODE=$1; SRC=$2; ADAPTER=$3; TAG=$4; P=$5; NOTE=$6
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
PRIO=${PRIO:-1}
REPO=/home/hice1/psunkad3/scratch/relvlm_ft
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -f "$SRC" ] || { echo "no $SRC"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter $ADAPTER"; exit 1; }
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
mkdir -p "$RUNS/$TAG"
INIT=$RUNS/$TAG/$(basename $SRC); cp "$SRC" "$INIT"; cmp "$SRC" "$INIT"
DROP=""; [ "$P" != "0" ] && DROP="--branch_dropout $P"
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT --freeze_abstractor \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_from $ADAPTER --freeze_lora_steps 500 --lora_lr 2e-5 \
--lr 3e-5 --proj_lr 1e-4 --train_samples 60000 --max_steps 3500 --warmup_steps 50 --lr_schedule cosine \
--decay_start 500 --val_every 500 --save_every 500 $DROP"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$RUNS/s0_qwen_pgm60k/best.pt TAG=$TAG EPOCHS=4 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=20 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=4 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm60k (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" "$PRIO" <<'PY'
import json, sys, shutil, time
p = "/home/hice1/psunkad3/scratch/relvlm_runs/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2])
if "lf5e" not in d["_about"]:
    d["_about"] += (" 2026-11-24 (user: 'queue any runs that you feel are important'): 5% per-epoch LoRA-first "
                    "runs lora_lf5e{1,2}_q_int_c60 (+ _ctrl) and the extra segments of qwen_pc_neutral5pct_ep at 1.")
json.dump(d, open(p, "w"), indent=2)
PY
scontrol update JobId=$JID Nice=$PRIO || true
echo "submitted $JID $TAG ($MODE, adapter $(basename $ADAPTER), branch_dropout $P)"
