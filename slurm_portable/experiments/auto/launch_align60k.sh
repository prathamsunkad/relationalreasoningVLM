#!/bin/bash
# Clean 60k pipeline, step 3: alignment on the 60k problems only. Same recipe as launch_align.sh (frozen LLM,
# Abstractor, aux head and slots; only the connector + gain/lambda train; lr 3e-4, warmup 200, cosine over 10k steps)
# but with --train_samples 60000 (the seed-0 subset LoRA alone 5% used) and the 60k-only slot model. Tier 2.
# usage: launch_align60k.sh <int|fuse> <60k branch .pt> <new tag> <manifest note>
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -euo pipefail
MODE=$1; INIT=$2; TAG=$3; NOTE=$4
RUNS=$RELVLM_RUNS
REPO=$RELVLM_REPO
SLOT=$RUNS/s0_qwen_pgm60k/best.pt
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
[ -f "$INIT" ] || { echo "no $INIT"; exit 1; }
mkdir -p "$RUNS/$TAG"
if [ "$MODE" = int ]; then INJ="--inject interleave"; else INJ="--inject fuse --fuse_lambda 0.9"; fi
EXTRA="$INJ --objective llm --rel_mode pool --layers 6 --head_dim 32 --init_from $INIT --freeze_abstractor \
--lr 3e-4 --warmup_steps 200 --lr_schedule cosine --max_steps 10000 --val_every 1000 --proj_lr 3e-4 --train_samples 60000"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$SLOT TAG=$TAG EPOCHS=1 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=60 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO_ROOT/slurm_portable/relational/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=1 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm60k (fine-tuned copy from init_from) $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" <<'PY'
import json, os, sys, shutil, time
p = os.environ["RELVLM_RUNS"] + "/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = 1
if "c60" not in d["_about"]:
    d["_about"] += (" 2026-11-17 (user: 'can you do H8, H9, H12 for the 60k model?'): clean 60k hybrids at 1 (tier 2): "
                    "align_q_*_L6ft60k_lr3e-4, then lora_w5_q_*_c60 (H8), lora_w5d_q_*_c60 (H9), lora_lf_q_*_c60 (H12); "
                    "each alignment drops to 2 once its runs are launched.")
json.dump(d, open(p, "w"), indent=2)
PY
scontrol update JobId=$JID Nice=1 || true
echo "submitted $JID $TAG from $(basename $INIT)"
