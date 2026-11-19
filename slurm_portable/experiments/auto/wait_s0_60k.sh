#!/bin/bash
# Clean 60k pipeline: when the 60k-only slot run s0_qwen_pgm60k finishes (done.json), launch the two Stage 1a
# branch runs on the same 60k problems (lr 3e-4 and 1e-3). Gives up if the slot run's chain stops without done.json
# (3 checks in a row, 5 min apart). Runs detached (nohup setsid); log in auto/wait_s0_60k.log.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

set -u
RUNS=$RELVLM_RUNS
A=$REPO_ROOT/slurm_portable/experiments/auto
TAG=s0_qwen_pgm60k
log() { echo "$(date '+%b %-d %H:%M') $*"; }
miss=0
log "waiting for $TAG"
while [ ! -e $RUNS/$TAG/done.json ]; do
  if q=$(squeue -u "$USER" -h -n "$TAG,relvlm_tests" -o %i 2>/dev/null); then
    if [ -z "$q" ]; then miss=$((miss + 1)); else miss=0; fi
  fi
  [ $miss -ge 3 ] && { log "$TAG: chain stopped without done.json; nothing launched"; exit 1; }
  sleep 300
done
sleep 60
log "$TAG finished: $(cat $RUNS/$TAG/done.json 2>/dev/null | cut -c1-200)"
N="clean 60k pipeline step 2, branch trained only on the 60k problems LoRA alone 5% saw, from the 60k-only slots; 40 epochs = 150k steps like the 10-epoch branch; launched by auto/wait_s0_60k.sh; user: 'Yes, fully clean'"
$A/launch_aux60k.sh 3e-4 aux_ft60k_q_L6_lr3e-4 "$N; lr 3e-4 = our best branch recipe" 2>&1 | while read l; do log "$l"; done
$A/launch_aux60k.sh 1e-3 aux_ft60k_q_L6_lr1e-3 "$N; lr 1e-3 = the rate that did slightly better in the earlier 60k runs" 2>&1 | while read l; do log "$l"; done
