#!/bin/bash
# 60k-only Abstractor smoke runs (Nov 23, user: 'can we check for smoke runs for lesser number of layers of the abstractor
# and some regularization or learning rate'). Same pipeline as aux_ft60k_q_L6_lr1e-3 (60k-only slots s0_qwen_pgm60k,
# --train_samples 60000, objective aux, warmup 500, val every 1k on the shared 1,000 validation puzzles), shorter:
# 20 epochs = 75,000 steps, constant lr until 65k then cosine. The reference run reached 25.6% at 25k, 27.8% at 50k and
# peaked at 30.8% at 71k (train 35%), then overfit (val 27.4%, train 54% at 150k).
# usage: launch_aux60k_smoke.sh <tag> <manifest note> <extra flags...>   (e.g. --layers 4 --abs_dropout 0.1 ...)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -euo pipefail
TAG=$1; NOTE=$2; shift 2; FLAGS="$*"
RUNS=$RELVLM_RUNS
REPO=$RELVLM_REPO
SLOT=$RUNS/s0_qwen_pgm60k/best.pt
PRIO=${PRIO:-1}
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
EXTRA="--inject interleave --objective aux --lr_schedule cosine --decay_start 65000 --val_every 1000 --train_samples 60000 $FLAGS"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO SLOT=$SLOT TAG=$TAG EPOCHS=20 EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" \
      AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=60 TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL $SB_OPTS $REPO_ROOT/slurm_portable/relational/slurm/stage1.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep= epochs=20 REPO=relvlm_ft GPUs=a100/h100/h200/l40s slots=s0_qwen_pgm60k $EXTRA (submitted $(date '+%H:%M %b %-d'): $NOTE)" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" "$PRIO" <<'PY'
import json, os, sys, shutil, time
p = os.environ["RELVLM_RUNS"] + "/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2])
if "aux60s_" not in d["_about"]:
    d["_about"] += (" 2026-11-23 (user: 'can we check for smoke runs for lesser number of layers of the abstractor and some "
                    "regularization'): 60k-only Abstractor smoke runs aux60s_* and the 5% LoRA rerun with per-epoch adapters "
                    "(qwen_pc_neutral5pct_ep) at 1.")
json.dump(d, open(p, "w"), indent=2)
PY
scontrol update JobId=$JID Nice=$PRIO || true
echo "submitted $JID $TAG: $FLAGS"
