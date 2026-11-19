#!/bin/bash
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 03:30:00
#SBATCH -o logs/%x_%j.log
# A3 (user, Nov 19: "ablate by removing & shuffling learned relation tokens during evaluation to check whether the
# decoder actually uses them"): one interleave run scored on the 5,000 test puzzles (test5000.tsv, same as A2) under
# each condition, all in this one job on one GPU type, so the conditions compare like for like:
#   normal          no ablation
#   zero            relation tokens set to 0 (placeholders kept)            = --branch_off
#   drop            placeholders removed from the prompt
#   shuffle_cand    each candidate gets the tokens of another candidate of the same puzzle
#   shuffle_puzzle  each puzzle gets another puzzle's tokens (same candidate slot)
#   permute         the 9 / 144 tokens of a sequence in a fixed random order (panel / slot runs only)
# Writes $D/<run>__<condition>.json (per-puzzle records); existing files are skipped, so a resubmit resumes.
# Usage: sbatch -J abl_<run> diag_ablate.sbatch <run> <condition>...      (submitted by submit_ablate.sh)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
RUNS=$RELVLM_RUNS
D=$RUNS/diag_ablate
DATA=$DATA/test_full/neutral
LIST=$REPO_ROOT/results/splits/test5000.tsv; mkdir -p $D
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
RUN=$1; shift
CKPT=${CKPT_RUNS:-$RUNS}/$RUN/best.pt
[ -f "$CKPT" ] || { echo "no $CKPT"; exit 1; }
cd $REPO
nvidia-smi --query-gpu=name --format=csv,noheader
for C in "$@"; do
  OUT=$D/${RUN}__${C}.json
  [ -s $OUT ] && { echo "=== $RUN $C: exists, skipping"; continue; }
  case $C in
    normal) A="" ;;
    zero)   A="--branch_off" ;;
    *)      A="--rel_ablate $C" ;;
  esac
  echo "=== $RUN $C ($(date +%H:%M:%S))"
  apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
      --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
      python eval.py --ckpt $CKPT --source pgm=matrix:$DATA --val_list $LIST --file_cache_dir $RUNS/filecache \
      --out_json $OUT $A || echo "=== $RUN $C FAILED"
done
echo "=== done ($(date +%H:%M:%S))"
