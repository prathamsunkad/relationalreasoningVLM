#!/bin/bash
# Clean 60k pipeline: when the 60k-only slot run s0_qwen_pgm60k finishes (done.json), launch the two Stage 1a
# branch runs on the same 60k problems (lr 3e-4 and 1e-3). Gives up if the slot run's chain stops without done.json
# (3 checks in a row, 5 min apart). Runs detached (nohup setsid); log in auto/wait_s0_60k.log.
set -u
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
A=$RUNS/auto
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
