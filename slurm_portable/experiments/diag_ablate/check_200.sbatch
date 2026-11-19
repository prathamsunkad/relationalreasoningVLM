#!/bin/bash
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 01:00:00
#SBATCH -o logs/%x_%j.log
# Real-data check of eval.py --rel_ablate (test copy relvlm_ft_ablate) on 200 test puzzles before the full A3 runs.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$WORK/relvlm_ft_ablate
RUNS=$RELVLM_RUNS
D=$RUNS/diag_ablate/check
X() { apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
      --env TRANSFORMERS_OFFLINE=1 /usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif "$@"; }
cd $REPO
E="--source pgm=matrix:$DATA/test_full/neutral --val_list $RUNS/diag_ablate/test5000.tsv --file_cache_dir $RUNS/filecache"
for C in normal drop shuffle_cand shuffle_puzzle zero; do
  case $C in normal) A="";; zero) A="--branch_off";; *) A="--rel_ablate $C";; esac
  X python eval.py --ckpt $RUNS/lora_lf_q_int_ft276k/best.pt $E --max_examples 200 --out_json $D/lf276_$C.json $A
done
for C in normal permute shuffle_cand; do
  case $C in normal) A="";; *) A="--rel_ablate $C";; esac
  X python eval.py --ckpt $RUNS/lora_q_int_slot_lr3e-4/best.pt $E --max_examples 40 --out_json $D/slot_$C.json $A
done
