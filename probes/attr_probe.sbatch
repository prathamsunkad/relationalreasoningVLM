#!/bin/bash
# Attribute recognition vs rule inference probe for one model (base or fine-tuned, HF format).
# usage (from the folder with pgm_attr_probe.py, pgm_gap.py, layout.json):
#   mkdir -p logs
#   sbatch -J probe_base attr_probe.sbatch OpenGVLab/InternVL3_5-1B-HF
#   sbatch -J probe_ft   attr_probe.sbatch /path/to/finetuned-internvl-hf
# The first job generates the puzzles (attr_probe/puzzles.jsonl); later jobs reuse them, so every model
# is tested on identical puzzles. Generate them once on a login node first if you submit several jobs at once:
#   python pgm_attr_probe.py synth --out attr_probe --n 1600 --template_jsonl $DATA/validation.jsonl --data $DATA --layout layout.json
#SBATCH -J attr_probe                         # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=64GB                    # Memory per GPU
#SBATCH -t4:00:00                             # Duration of the job
#SBATCH --exclude=atl1-1-03-017-30-0           # slow node
#SBATCH -o logs/%x.%j.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ../env_local.sh || { echo "run sbatch from the probes folder (../env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
OUT=attr_probe
N=1600               # puzzles (100 per rule type)

set -euo pipefail
MODEL=${1:?usage: sbatch attr_probe.sbatch <model id or checkpoint folder>}
if [ ! -f "$OUT/puzzles.jsonl" ]; then
  python pgm_attr_probe.py synth --out $OUT --n $N --template_jsonl "$DATA/validation.jsonl" --data "$DATA" --layout layout.json
fi
python pgm_attr_probe.py attrs  --out $OUT --model "$MODEL" --batch_size 32
python pgm_attr_probe.py infer  --out $OUT --model "$MODEL" --batch_size 16 --dump 1
python pgm_attr_probe.py report --out $OUT
