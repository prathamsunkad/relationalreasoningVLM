#!/bin/bash
#SBATCH -J diag_overlap
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 01:30:00
#SBATCH -o logs/diag_overlap_%j.log
# Phase 1 overlap check: per-problem predictions on the same 1,000 shared PGM validation problems from
#   LoRA alone (5% data best, all-data epoch-4 best), read through relvlm's eval.py: a fusion checkpoint with the
#   branch off is the plain VLM, and --lora_from puts the vlm_lora_pc adapter in as its LoRA;
#   the branch alone (aux head, no LLM) at 10 epochs and at two 50-epoch checkpoints;
#   the settled branch (brset: run it by name once aux_ft_q_L6_lr3e-4_settle has finished);
#   and each problem's rules (relation_structure). analyze.py then builds the overlap tables.
# Usage: sbatch [--dependency=afterok:<tests job>] diag_overlap.sbatch [only-these-names...]
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
RUNS=$RELVLM_RUNS
D=$RUNS/diag_overlap
DATA=$DATA/val_full/neutral
LIST=$REPO_ROOT/results/splits/val1000_shared.tsv
LP=$LORA_RUNS
FUSE=$RUNS/auto/snap/init_align_q_fuse_L6ft50_lr3e-4_step3000.pt
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
cd $REPO
X() { apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" "$@"; }
ONLY="$*"
[ -n "$ONLY" ] && echo "=== only: $ONLY"
ev() {  # name, then eval.py args
  local name=$1; shift
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && return
  echo "=== $name ($(date +%H:%M:%S))"
  X python eval.py --source pgm=matrix:$DATA --val_list $LIST --file_cache_dir $RUNS/filecache \
    --out_json $D/$name.json "$@"
}
ev lora5   --ckpt $FUSE --branch_off --lora_from $LP/qwen_pc_neutral5pct_recipe/best
ev lora100 --ckpt $FUSE --branch_off --lora_from $LP/qwen_pc_recipe_e1/best_e4
ev br10    --ckpt $RUNS/aux_ft_q_L6_lr3e-4_ep10/best.pt --readout aux
ev br50_204k --ckpt $RUNS/snapshots/aux_ft_q_L6_lr3e-4_ep50_step204000.pt --readout aux
ev br50_210k --ckpt $RUNS/snapshots/aux_ft_q_L6_lr3e-4_ep50_step210000.pt --readout aux
ev brset   --ckpt $RUNS/aux_ft_q_L6_lr3e-4_settle/best.pt --readout aux   # settled branch (lr 3e-4 to 10% from 231k), after it finishes
if [ -z "$ONLY" ] || [[ " $ONLY " == *" rules "* ]]; then
  echo "=== rules ($(date +%H:%M:%S))"
  X python $D/rules.py $LIST $DATA $D/rules.json
fi
echo "=== done ($(date +%H:%M:%S))"
