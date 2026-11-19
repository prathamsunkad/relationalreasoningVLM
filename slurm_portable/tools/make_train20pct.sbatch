#!/bin/bash
#SBATCH -J pgm_train20pct
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o logs/make_train20pct_%j.log

# Replace pgm_data/train_full/neutral with exactly the shared 20% selection (240k .npz), then clean up the overfit files.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

set -e
DATA=$DATA
W=$WORK/pgm20
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
apptainer exec "$CONTAINER" python $W/make_train20pct.py \
    --selection $W/train20pct_seed20261007.tsv --train_dir $DATA/train_full/neutral --tar $DATA/neutral.tar.gz \
    --protect $W/protect_overfit500.txt --work $W \
    --cache_out $LORA_RUNS/filecache/neutral/train_files.txt

# the overfit jobs read the protected files; remove the unselected ones once those jobs have ended
while squeue -j 6303355,6303356 -h 2>/dev/null | grep -q .; do sleep 60; done
n=0
while read f; do
    if ! grep -q "^$f	" $W/train20pct_seed20261007.tsv; then rm -f "$DATA/train_full/neutral/$f"; n=$((n+1)); fi
done < $W/protect_overfit500.txt
echo "removed $n leftover overfit-only files"
echo "train_full/neutral now has $(ls $DATA/train_full/neutral | wc -l) files (expected 240000)"
