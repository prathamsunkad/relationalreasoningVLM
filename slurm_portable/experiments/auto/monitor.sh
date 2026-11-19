#!/bin/bash
# In-session watcher: exit (and so notify) on the first notable event among our runs; otherwise after MAXH hours.
# usage: monitor.sh <max hours> <eval job id or 0>
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"
mkdir -p "$RELVLM_RUNS/logs" "$RELVLM_RUNS/auto/snap"; touch "$RELVLM_RUNS/night_queue_2026-11-08.txt"
[ -f "$RELVLM_RUNS/priority.json" ] || echo '{"_about": ""}' > "$RELVLM_RUNS/priority.json"

R=~/scratch/relvlm_runs; MAXH=${1:-4}; EJ=${2:-0}
CH="aux_ft_q_L6_lr3e-4_ep50 aux_ft_q_L24_lr8e-5_ep50 aux_ft60k_q_L6_lr3e-4 aux_ft60k_q_L6_lr1e-3 align_q_int_L6ft60k_lr3e-4 align_q_fuse_L6ft60k_lr3e-4 lora_w5_q_int_c60 lora_w5_q_fuse_c60 lora_w5d_q_int_c60 lora_w5d_q_fuse_c60 lora_lf_q_int_c60 lora_lf_q_fuse_c60 align_q_int_L6ftc60e_lr3e-4 align_q_fuse_L6ftc60e_lr3e-4 lora_w5_q_int_c60e lora_w5_q_fuse_c60e lora_w5d_q_int_c60e lora_w5d_q_fuse_c60e lora_lf_q_int_c60e lora_lf_q_fuse_c60e"
end=$(( $(date +%s) + MAXH * 3600 )); declare -A miss seen
for t in $CH; do miss[$t]=0; { [ -e $R/$t/final.pt ] || [ -e $R/$t/done.json ]; } && seen[$t]=1; done
start=$(date +%s)
while [ $(date +%s) -lt $end ]; do
  ev=""
  q=$(squeue -u $USER -h -o '%j %T' 2>/dev/null) || { sleep 120; continue; }
  for t in $CH; do
    [ -d $R/$t ] || continue
    if [ -e $R/$t/final.pt ] || [ -e $R/$t/done.json ]; then [ -z "${seen[$t]:-}" ] && { ev="$ev | $t FINISHED"; seen[$t]=1; }; continue; fi
    if echo "$q" | grep -q "^$t "; then miss[$t]=0; else miss[$t]=$((miss[$t] + 1)); fi
    [ ${miss[$t]} -ge 4 ] && ev="$ev | $t CHAIN STOPPED (no job, not finished)"
    f=$(ls -t $R/logs/stage?_${t}_[0-9]*.log 2>/dev/null | head -1)
    [ -n "$f" ] && [ $(stat -c %Y "$f") -ge $start ] && grep -qE '^Traceback|CUDA out of memory|Killed' "$f" && ev="$ev | $t ERROR in $(basename $f)"
  done
  if [ "$EJ" != 0 ] && [ -z "$(squeue -h -j $EJ -o %i 2>/dev/null)" ]; then ev="$ev | eval job $EJ ended: $(sacct -n -X -j $EJ -o State | tr -d ' ')"; EJ=0; fi
  [ -n "$ev" ] && { echo "$(date '+%b %-d %H:%M') EVENT$ev"; break; }
  sleep 150
done
[ -z "${ev:-}" ] && echo "$(date '+%b %-d %H:%M') no events in $MAXH h"
squeue -u $USER -h -o '%i %j %T %M %y %R' | sort -k2
for t in $CH; do [ -d $R/$t ] || continue; f=$(ls -t $R/logs/stage?_${t}_[0-9]*.log 2>/dev/null | head -1); [ -n "$f" ] && echo "$t: $(grep -E '^\[val\]|^step ' "$f" | tail -1 | cut -c1-150)"; done
