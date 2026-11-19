#!/bin/bash
# Clean 60k pipeline, step 1 (user, Nov 16: 'Yes, fully clean'): Stage 0 slots trained only on the images of the
# 60k PGM problems that LoRA alone 5% saw (--train_samples 60000: the same nested seeded subset, data.subset seed 0;
# the cached train lists of both codebases are identical). Everything else = the s0_qwen_pgm recipe (the defaults:
# K 16, T 3, grid 12, slot dim 32, decoder 512, 16 examples/step, lr 4e-4, warmup 10k, 100k steps).
# Starts only if the tiny tests pass (they cover the new opt-in --train_samples flag of train_stage0.py).
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -euo pipefail
RUNS=$RELVLM_RUNS
REPO=$RELVLM_REPO
TAG=s0_qwen_pgm60k
[ -e "$RUNS/$TAG" ] && { echo "$RUNS/$TAG exists, refusing"; exit 1; }
cd $REPO
TESTS=$(REPO=$REPO sbatch --parsable --export=ALL $REPO_ROOT/slurm_portable/relational/slurm/tests.sbatch)
EXTRA="--train_samples 60000 --max_steps 100000"
SB_OPTS="-J $TAG -t 02:00:00 -C gpu-a100|gpu-h100|gpu-h200|gpu-l40s -p coc-gpu,ice-gpu --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0"
JID=$(REPO=$REPO TAG=$TAG EXTRA="$EXTRA" SB_OPTS="$SB_OPTS" AUTO_RESUBMIT=1 ATTEMPT=1 MAX_ATTEMPTS=30 \
      TIME_LIMIT_MIN=110 MODEL_ID=Qwen/Qwen3-VL-2B-Instruct \
      sbatch --parsable --export=ALL --dependency=afterok:$TESTS $SB_OPTS $REPO_ROOT/slurm_portable/relational/slurm/stage0.sbatch)
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $TAG dep=afterok:$TESTS REPO=relvlm_ft GPUs=a100/h100/h200/l40s data=PGM-60k $EXTRA (submitted $(date '+%H:%M %b %-d'): clean 60k pipeline step 1, slots trained only on the images of the 60k problems LoRA alone 5% saw; otherwise the s0_qwen_pgm recipe; starts only if the tiny tests ($TESTS) pass; auto/wait_s0_60k.sh then launches aux_ft60k_q_L6_lr3e-4 and _lr1e-3; user: 'Yes, fully clean')" >> $RUNS/night_queue_2026-11-08.txt
python3 - "$TAG" <<'PY'
import json, os, sys, shutil, time
p = os.environ["RELVLM_RUNS"] + "/priority.json"
shutil.copy(p, p.replace("priority.json", time.strftime(".priority_backup_%m%d_%H%M%S.json")))
d = json.load(open(p))
for t in (sys.argv[1], "aux_ft60k_q_L6_lr3e-4", "aux_ft60k_q_L6_lr1e-3"):
    d[t] = 0
if "60k pipeline" not in d["_about"]:
    d["_about"] += " 2026-11-16 (user: 'Yes, fully clean'): clean 60k pipeline s0_qwen_pgm60k and aux_ft60k_q_L6_lr3e-4/_lr1e-3 at 0."
json.dump(d, open(p, "w"), indent=2)
PY
scontrol update JobId=$JID Nice=0 || true
echo "submitted tests $TESTS and $JID $TAG (after the tests)"
