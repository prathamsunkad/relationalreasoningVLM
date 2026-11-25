#!/bin/bash
# Phase 2 (Nov 23, user: 'start all the phases'): LoRA-first on the 20% LoRA (qwen_pc_recipe_e1, all 240k puzzles),
# the H12 recipe unchanged except the adapter and the data: steps 0-500 LoRA frozen, connector only; steps 500-3500
# LoRA (2e-5) + connector, cosine to 10%; Abstractor and slots frozen; train_samples 0 = the same 240k puzzles the
# branch and the adapter were trained on (matched data). Control = same with --branch_dropout 1.0 (validated off too).
# usage: launch_lf20.sh <int|fuse> <aligned init .pt> <adapter dir> <new tag> <branch_dropout 0|1.0> <manifest note>
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -euo pipefail
MODE=$1; SRC=$2; ADAPTER=$3; TAG=$4; P=$5; NOTE=$6
RUNS=$RELVLM_RUNS
PRIO=${PRIO:-1}
REPO=$RELVLM_REPO
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -f "$SRC" ] || { echo "no $SRC"; exit 1; }
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter $ADAPTER"; exit 1; }
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
mkdir -p "$RUNS/$TAG"
INIT=$RUNS/$TAG/$(basename $SRC); cp "$SRC" "$INIT"; cmp "$SRC" "$INIT"
DROP=""; [ "$P" != "0" ] && DROP="--branch_dropout $P"
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT --freeze_abstractor \
--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 --lora_from $ADAPTER --freeze_lora_steps 500 --lora_lr 2e-5 \
--lr 3e-5 --proj_lr 1e-4 --train_samples 0 --max_steps 3500 --warmup_steps 50 --lr_schedule cosine \
--decay_start 500 --val_every 500 --save_every 500 $DROP"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$RUNS/s0_qwen_pgm/best.pt TAG=$TAG EPOCHS=4 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=20 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO_ROOT/slurm_portable/relational/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=4 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" "$PRIO" <<'PY'
import json, os, sys, shutil, time
p = os.environ["RELVLM_RUNS"] + "/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2])
if "ft636k" not in d["_about"]:
    d["_about"] += (" 2026-11-23 (user: 'start all the phases and prioritize the phases accordingly'): phase 0 (diag_5k "
                    "scoring, align_q_*_L6ft636k_lr3e-4) at 0; LoRA-first runs at 5% (lora_lf_q_*_ft636k) and 20% "
                    "(lora_lf20e{1,4}_q_*, with controls) at 1; Stage 2 dropout at 5% (lora_w5d_q_*_L6ft636k) at 2; "
                    "alignments drop to 2 once phases 1-2 are launched (auto/auto_p636.py).")
json.dump(d, open(p, "w"), indent=2)
PY
scontrol update JobId=$JID Nice=$PRIO || true
echo "submitted $JID $TAG ($MODE, adapter $(basename $ADAPTER), branch_dropout $P)"
