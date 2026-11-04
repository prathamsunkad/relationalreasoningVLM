#!/bin/bash
#SBATCH -J pgm_train20pct
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH -o /home/hice1/psunkad3/pgm20/make_train20pct_%j.log

# Replace pgm_data/train_full/neutral with exactly the shared 20% selection (240k .npz), then clean up the overfit files.
set -e
DATA=/home/hice1/psunkad3/scratch/pgm_data
W=/home/hice1/psunkad3/pgm20
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
apptainer exec "$CONTAINER" python $W/make_train20pct.py \
    --selection $W/train20pct_seed20261007.tsv --train_dir $DATA/train_full/neutral --tar $DATA/neutral.tar.gz \
    --protect $W/protect_overfit500.txt --work $W \
    --cache_out /home/hice1/psunkad3/scratch/vlm_lora_pc_runs/filecache/neutral/train_files.txt

# the overfit jobs read the protected files; remove the unselected ones once those jobs have ended
while squeue -j 6303355,6303356 -h 2>/dev/null | grep -q .; do sleep 60; done
n=0
while read f; do
    if ! grep -q "^$f	" $W/train20pct_seed20261007.tsv; then rm -f "$DATA/train_full/neutral/$f"; n=$((n+1)); fi
done < $W/protect_overfit500.txt
echo "removed $n leftover overfit-only files"
echo "train_full/neutral now has $(ls $DATA/train_full/neutral | wc -l) files (expected 240000)"
