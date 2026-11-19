#!/bin/bash
# A3: submit one diag_ablate.sbatch job per interleave run (H4, H7-H12), log each in the job manifest and give it a
# scheduling tier. Usage: bash submit_ablate.sh [run ...]   (default: all 14 runs below)
set -euo pipefail
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
D=$RUNS/diag_ablate
TIER=${TIER:-1}
BASE="normal zero drop shuffle_cand shuffle_puzzle"
declare -A EXTRA=( [lora_q_int_panel_lr3e-4]=permute [lora_q_int_slot_lr3e-4]=permute [lora_q_int_slot_lr1e-3]=permute
                   [lora_q_int_slot_control_lr3e-4]=permute )
ALL="lora_lf_q_int_ft276k align_q_int_L6ft276k_lr3e-4 lora_lf_q_int_ft50 lora_w5d_q_int_L6ft50 align_q_int_L6ft50_lr3e-4
lora_w5d_q_int_L6ft lora_w5_q_int_L6ft align_q_int_L6ft_lr3e-4 align_q_int_L6ft_lr1e-3 lora_q_int_pool_lr3e-4
lora_q_int_panel_lr3e-4 lora_q_int_slot_lr3e-4 lora_q_int_slot_lr1e-3 lora_q_int_slot_control_lr3e-4"
LIST=${*:-$ALL}
cp $RUNS/night_queue_2026-11-08.txt $RUNS/.night_queue_backup_$(date +%m%d_%H%M%S).txt
cp $RUNS/priority.json $RUNS/.priority_backup_$(date +%m%d_%H%M%S).json
for R in $LIST; do
  C="$BASE ${EXTRA[$R]:-}"
  JID=$(sbatch --parsable -J abl_$R -C 'gpu-a100|gpu-h100|gpu-h200|gpu-l40s' -p coc-gpu,ice-gpu \
        --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0 $D/diag_ablate.sbatch $R $C)
  scontrol update JobId=$JID Nice=$TIER || true
  echo "$JID abl_$R dep= kind=eval REPO=relvlm_ft GPUs=a100/h100/h200/l40s (submitted $(date '+%H:%M %b %-d') by diag_ablate/submit_ablate.sh: A3 relation-token ablation of $R on the 5,000 test puzzles, conditions: $C; then diag_ablate/analyze_ablate.py; user: 'run it on the most important experiments not the beginning ones which did not give results')" >> $RUNS/night_queue_2026-11-08.txt
  python3 - "abl_$R" "$TIER" <<'PY'
import json, sys
p = "/home/hice1/psunkad3/scratch/relvlm_runs/priority.json"
d = json.load(open(p)); d[sys.argv[1]] = int(sys.argv[2]); json.dump(d, open(p, "w"), indent=1)
PY
  echo "submitted $JID abl_$R: $C"
done
