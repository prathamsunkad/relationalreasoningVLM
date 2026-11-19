#!/bin/bash
# Phase 1 follow-up (user: 'okay start it'): once the settled branch aux_ft_q_L6_lr3e-4_settle has finished
# (final.pt), score its best checkpoint on the overlap problems (diag_overlap.sbatch brset), then rerun analyze.py.
# Gives up if the run's chain stops without a final.pt. Runs detached (nohup setsid); log in auto/wait_brset.log.
set -u
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
TAG=aux_ft_q_L6_lr3e-4_settle
D=$RUNS/diag_overlap
miss=0
while [ ! -e $RUNS/$TAG/final.pt ]; do
  if [ -z "$(squeue -u $USER -h -n $TAG -o %i)" ]; then
    miss=$((miss + 1)); [ $miss -ge 3 ] && { echo "$(date) $TAG chain stopped without final.pt; not submitting"; exit 1; }
  else miss=0; fi
  sleep 300
done
sleep 60   # let the final best.pt/best.json writes settle
echo "$(date) $TAG finished: $(cat $RUNS/$TAG/best.json 2>/dev/null)"
JID=$(sbatch --parsable -C 'gpu-a100|gpu-h100|gpu-h200|gpu-l40s' -p coc-gpu,ice-gpu \
      --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0 $D/diag_overlap.sbatch brset) || { echo "$(date) sbatch failed"; exit 1; }
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
echo "$JID diag_overlap dep= kind=eval REPO=relvlm_ft GPUs=a100/h100/h200/l40s (submitted $(date '+%H:%M %b %-d') by auto/wait_brset.sh: overlap check for the settled branch only (brset = aux_ft_q_L6_lr3e-4_settle/best.pt, aux readout) on the same 1,000 shared val problems; then analyze.py; user: 'okay start it')" >> $RUNS/night_queue_2026-11-08.txt
scontrol update JobId=$JID Nice=0 || true
echo "$(date) submitted $JID"
while [ -n "$(squeue -h -j $JID -o %i 2>/dev/null)" ]; do sleep 120; done
st=$(sacct -j $JID -X -n -o State | tr -d ' ')
echo "$(date) $JID ended: $st"
if [ -s $D/brset.json ]; then
  python3 $D/analyze.py > $D/analyze_brset.out 2>&1 && echo "$(date) analyze.py done: $D/report.md" || echo "$(date) analyze.py failed"
else
  echo "$(date) no brset.json; see $RUNS/logs/diag_overlap_$JID.log"
fi
