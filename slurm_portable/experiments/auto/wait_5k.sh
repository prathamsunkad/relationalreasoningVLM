#!/bin/bash
# Phase 1, step 4 (user: 'okay start it'): wait for a group of runs to finish (final.pt), then submit
# diag_5k/diag_5k.sbatch for their evals (5,000 unseen test problems), log it in the manifest, wait for it, and run
# analyze5k.py. A run whose Slurm chain disappears without a final.pt (3 checks in a row, 5 min apart) is skipped.
# usage: wait_5k.sh <job name: diag_5k|diag_5kw> <time limit> <run=eval,eval>... [-=extra,evals]
# Runs detached (nohup setsid); log in auto/wait_5k_<job name>.log.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -u
RUNS=$RELVLM_RUNS
D=$RUNS/diag_5k
JOB=$1; TL=$2; shift 2
log() { echo "$(date '+%b %-d %H:%M') $*"; }
declare -A EV MISS STATE
ORDER=(); EXTRA=""
for spec in "$@"; do
  r=${spec%%=*}; e=${spec#*=}
  if [ "$r" = "-" ]; then EXTRA="$EXTRA ${e//,/ }"; else EV[$r]=${e//,/ }; ORDER+=("$r"); MISS[$r]=0; fi
done
log "waiting for: ${ORDER[*]}"
while :; do
  pending=0
  for r in "${ORDER[@]}"; do
    [ -n "${STATE[$r]:-}" ] && continue
    if [ -e "$RUNS/$r/final.pt" ]; then STATE[$r]=ok; log "$r finished"; continue; fi
    if q=$(squeue -u "$USER" -h -n "$r" -o %i 2>/dev/null); then
      if [ -z "$q" ]; then MISS[$r]=$((MISS[$r] + 1)); else MISS[$r]=0; fi
    fi
    if [ "${MISS[$r]}" -ge 3 ]; then STATE[$r]=stopped; log "$r: chain stopped without final.pt, skipping ${EV[$r]}"; continue; fi
    pending=1
  done
  [ $pending = 0 ] && break
  sleep 300
done
NAMES="$EXTRA"
for r in "${ORDER[@]}"; do [ "${STATE[$r]}" = ok ] && NAMES="$NAMES ${EV[$r]}"; done
NAMES=$(echo $NAMES)
[ -z "$NAMES" ] && { log "nothing to score"; exit 1; }
sleep 90   # let the last best.pt / final.pt writes settle
JID=$(sbatch --parsable -J "$JOB" -t "$TL" -C 'gpu-a100|gpu-h100|gpu-h200|gpu-l40s' -p coc-gpu,ice-gpu \
      --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0 "$REPO_ROOT/slurm_portable/experiments/diag_5k/diag_5k.sbatch" $NAMES) || { log "sbatch failed"; exit 1; }
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID $JOB dep= kind=eval REPO=relvlm_ft GPUs=a100/h100/h200/l40s (submitted $(date '+%H:%M %b %-d') by auto/wait_5k.sh: Phase 1 step 4, finished runs scored on 5,000 unseen PGM test problems (diag_5k/test5000.tsv) with per-problem records: $NAMES; then diag_5k/analyze5k.py; user: 'okay start it')" >> $RUNS/night_queue_2026-11-08.txt
scontrol update JobId=$JID Nice=0 || true
log "submitted $JID ($JOB): $NAMES"
while [ -n "$(squeue -h -j $JID -o %i 2>/dev/null)" ]; do sleep 120; done
log "$JID ended: $(sacct -j $JID -X -n -o State | tr -d ' ')"
if ls $D/*.json >/dev/null 2>&1; then
  python3 $D/analyze5k.py > $D/analyze5k_${JOB}.out 2>&1 && log "analyze5k.py done: $D/report.md" || log "analyze5k.py failed"
fi
