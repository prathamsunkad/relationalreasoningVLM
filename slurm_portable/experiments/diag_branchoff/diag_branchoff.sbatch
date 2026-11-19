#!/bin/bash
#SBATCH -J diag_branchoff
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 01:00:00
#SBATCH -o logs/diag_branchoff_%j.log
# Ablation on the warm-started 5% Stage 2 runs (lora_w5_*): score the same 1,000 shared PGM validation problems
# with the branch on and with it cut (fusion: lambda = 1; interleave: zero relation token). Branch-off accuracy is
# what LoRA learned on its own inside the hybrid; compare it with LoRA alone at the same step.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
RUNS=$RELVLM_RUNS
D=$RUNS/diag_branchoff
DATA=$DATA
LIST=$REPO_ROOT/results/splits/val1000_shared.tsv
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
cd $REPO
for run in lora_w5_q_int_L6ft lora_w5_q_fuse_L6ft; do
  for mode in on off; do
    OFF=""; [ $mode = off ] && OFF="--branch_off"
    echo "=== $run branch $mode ($(date +%H:%M:%S))"
    apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
      --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
      python eval.py --ckpt $D/${run}_state.pt --source pgm=matrix:$DATA/val_full/neutral --val_list $LIST \
      --file_cache_dir $RUNS/filecache --out_json $D/${run}_branch_${mode}.json $OFF
  done
done
echo "=== done ($(date +%H:%M:%S))"
