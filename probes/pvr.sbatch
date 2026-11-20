#!/bin/bash
# Simple perception vs reasoning probe for one model, on both input formats.
# usage (from the folder with perceive_vs_reason.py, pgm_gap.py, layout.json):
#   mkdir -p logs
#   sbatch -J pvr_base pvr.sbatch OpenGVLab/InternVL3_5-1B-HF
#   sbatch -J pvr_ft   pvr.sbatch /path/to/finetuned-internvl-hf
#SBATCH -J pvr                                # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=64GB                    # Memory per GPU
#SBATCH -t3:00:00                             # Duration of the job
#SBATCH --exclude=atl1-1-03-017-30-0           # slow node
#SBATCH -o logs/%x.%j.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the gap_hybrid_probes folder (env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
SPLIT=validation

set -euo pipefail
MODEL=${1:?usage: sbatch pvr.sbatch <model id or checkpoint folder>}
for INPUT in crops sheet; do
  python perceive_vs_reason.py run --data "$DATA" --split $SPLIT --model "$MODEL" --input $INPUT --batch_size 16
done
python perceive_vs_reason.py report --split $SPLIT
