#!/bin/bash
# Nov 24 (user: 'queue any runs that you feel are important'): wait for the 5% LoRA rerun (qwen_pc_neutral5pct_ep) to
# write final/ (epoch 4), then score its 4 epoch adapters alone on the 5,000 test puzzles in one diag_5k job (same GPU
# type for the per-epoch overlap check against br60/br636k). Detached; log in auto/wait_ep5.log.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -u
RUNS=$RELVLM_RUNS; D=$RUNS/diag_5k
E5=$LORA_RUNS/qwen_pc_neutral5pct_ep
log() { echo "$(date '+%b %-d %H:%M') $*"; }
log "waiting for $E5/final and step_11250"; miss=0
while [ ! -f $E5/final/adapter_config.json ]; do
  if [ -z "$(squeue -u $USER -h -n qwen_pc_neutral5pct_ep -o %i)" ]; then miss=$((miss+1)); else miss=0; fi
  [ $miss -ge 3 ] && { log "chain gone without final/; scoring what exists"; break; }
  sleep 300
done
sleep 90
NAMES=""; for e in 1 2 3 4; do d=$E5/step_$((3750*e)); [ $e = 4 ] && d=$E5/final; [ -f $d/adapter_config.json ] && NAMES="$NAMES lora5ep_e$e"; done
NAMES=$(echo $NAMES); [ -z "$NAMES" ] && { log "nothing to score"; exit 1; }
JID=$(sbatch --parsable -J diag_5k -t 03:00:00 -C 'gpu-a100|gpu-h100|gpu-h200|gpu-l40s' -p coc-gpu,ice-gpu \
      --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0 "$REPO_ROOT/slurm_portable/experiments/diag_5k/diag_5k.sbatch" $NAMES) || { log "sbatch failed"; exit 1; }
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID diag_5k dep= kind=eval REPO=relvlm_ft GPUs=a100/h100/h200/l40s (submitted $(date '+%H:%M %b %-d') by auto/wait_ep5.sh: the 5% LoRA rerun after epochs 1-4 alone on the 5,000 test puzzles: $NAMES; user: 'queue any runs that you feel are important'; tier 0)" >> $RUNS/night_queue_2026-11-08.txt
log "submitted $JID: $NAMES"
while [ -n "$(squeue -h -j $JID -o %i 2>/dev/null)" ]; do sleep 120; done
log "$JID ended: $(sacct -j $JID -X -n -o State | tr -d ' ')"
python3 $D/analyze5k.py > $D/analyze5k_ep5.out 2>&1 && log "analyze5k.py done" || log "analyze5k.py failed"
