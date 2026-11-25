#!/bin/bash
#SBATCH -J pgm-zs-prep
#SBATCH -p ice-cpu
#SBATCH -N1 -n1 -c8
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /home/hice1/khsieh37/scratch/pgm-zeroshot-eval/logs/prep-%j.log
# Downloads the PGM neutral test split (HuggingFaceM4/PGM, pinned revision; 9 parquet files, 6.2 GB) and converts it
# into 28 shard packs for the pgm-lora-v1 evaluator. Tools go in node-local /tmp; nothing in the bundle is touched.
set -euo pipefail
W=$HOME/scratch/pgm-zeroshot-eval
B=$HOME/scratch/pgm-lora-v1
P=$HOME/scratch/pgm-neutral-20percent_data/pgm-neutral-train20pct-seed20261007-v1/pgm-neutral-train20pct-seed20261007-v1
REV=195a03a7e3bc6d6a349b55986afba5319929c3bf
T=${TMPDIR:-/tmp}/pgmzs_$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
module load python/3.11.9
python -m venv $T/venv && source $T/venv/bin/activate
pip install -q --cache-dir $T/pipcache pyarrow==21.0.0 numpy==2.3.4 pillow==11.3.0 huggingface_hub==0.35.3
python - <<PY
from huggingface_hub import hf_hub_download
for k in range(9):
    p = hf_hub_download('HuggingFaceM4/PGM', f'data/test-{k:05d}-of-00009.parquet', repo_type='dataset',
                        revision='$REV', local_dir='$W/hf_parquet')
    print(p, flush=True)
PY
python $W/scripts/build_packs.py --parquet_dir $W/hf_parquet/data --out $W/packs --samples $B/sample_problems \
    --sheet_jsonl $P/test.jsonl
du -sh $W/hf_parquet $W/packs
