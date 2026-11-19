#!/bin/bash
# A3: wait until no abl_* job is queued or running, resubmit (once) any run with missing conditions, then run the analysis.
D=/home/hice1/psunkad3/scratch/relvlm_runs/diag_ablate
log() { echo "$(date '+%b %d %H:%M:%S') $*" >> $D/wait_ablate.log; }
log start
for round in 1 2; do
  while [ -n "$(squeue -u $USER -h -o %j | grep '^abl_')" ]; do sleep 300; done
  missing=""
  python3 $D/analyze_ablate.py --dir $D >> $D/wait_ablate.log 2>&1 && log "analysis done (round $round): $D/report.md"
  [ $round = 1 ] && { for R in lora_lf_q_int_ft276k align_q_int_L6ft276k_lr3e-4 lora_lf_q_int_ft50 lora_w5d_q_int_L6ft50 align_q_int_L6ft50_lr3e-4 lora_w5d_q_int_L6ft lora_w5_q_int_L6ft align_q_int_L6ft_lr3e-4 align_q_int_L6ft_lr1e-3 lora_q_int_pool_lr3e-4 lora_q_int_panel_lr3e-4 lora_q_int_slot_lr3e-4 lora_q_int_slot_lr1e-3 lora_q_int_slot_control_lr3e-4; do
      n=$(ls $D/${R}__*.json 2>/dev/null | wc -l); want=5; case $R in *panel*|*slot*) want=6;; esac
      [ $n -lt $want ] && missing="$missing $R"; done
    [ -z "$missing" ] && { log "all conditions present"; break; }
    log "missing conditions for:$missing; resubmitting once"; bash $D/submit_ablate.sh $missing >> $D/wait_ablate.log 2>&1; sleep 120; }
done
log end
