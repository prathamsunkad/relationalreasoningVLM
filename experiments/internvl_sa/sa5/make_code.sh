#!/bin/bash
# Rebuilds the sa5/code and sa5/lists folders the 5% Slot Abstractor jobs ran from (Nov 25): the repo's slot_abstractor/*.py
# with one line changed in run_pgm_jsonl.py and pretrain_slot_ae_pgm.py (the contact-sheet loader replaced by
# pgm_npz_data.NpzPGM), plus our exact 5% / val1000 / test5000 lists as paths relative to the pgm_data folder.
# usage: bash make_code.sh <target dir, e.g. $RUNS/sa5>
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REPO=$(cd "$HERE/../../.." && pwd); T=${1:?target dir}
mkdir -p $T/code $T/lists $T/weights $T/logs
cp $REPO/slot_abstractor/*.py $HERE/pgm_npz_data.py $T/code/
for f in run_pgm_jsonl.py pretrain_slot_ae_pgm.py; do
  sed -i 's/^from pgm_jsonl_data import JsonlPGM$/from pgm_npz_data import NpzPGM as JsonlPGM   # Nov 25: our 5% .npz list instead of contact sheets/' $T/code/$f
done
S=$REPO/results/splits
zcat $S/pgm_neutral_train_5pct_60k.txt.gz | sed 's#^#train_full/neutral/#' > $T/lists/train_5pct_60k.txt
grep -v "^#" $S/val1000_shared.tsv | cut -f1 | sed 's#^#val_full/neutral/#' > $T/lists/val1000.txt
grep -v "^#" $S/test5000.tsv | cut -f1 | sed 's#^#test_full/neutral/#' > $T/lists/test5000.txt
cp $HERE/sa5_job.sbatch $T/
echo "built $T (then: sbatch -J sa5_ae $T/sa5_job.sbatch ae; ... train; ... test)"
