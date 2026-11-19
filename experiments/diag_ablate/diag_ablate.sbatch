#!/bin/bash
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 03:30:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/%x_%j.log
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
REPO=/home/hice1/psunkad3/scratch/relvlm_ft
RUNS=/home/hice1/psunkad3/scratch/relvlm_runs
D=$RUNS/diag_ablate
DATA=/home/hice1/psunkad3/scratch/pgm_data/test_full/neutral
LIST=$D/test5000.tsv
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
RUN=$1; shift
CKPT=$RUNS/$RUN/best.pt
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
  apptainer exec --nv --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface --env HF_HUB_OFFLINE=1 \
      --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
      python eval.py --ckpt $CKPT --source pgm=matrix:$DATA --val_list $LIST --file_cache_dir $RUNS/filecache \
      --out_json $OUT $A || echo "=== $RUN $C FAILED"
done
echo "=== done ($(date +%H:%M:%S))"
