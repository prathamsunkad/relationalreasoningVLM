#!/bin/bash
# Real-PGM probe (attribute perception vs rule inference) for one model (base or fine-tuned, HF format).
# usage (from the folder with pgm_real_probe.py, pgm_gap.py, layout.json):
#   python pgm_real_probe.py rules --data $DATA --split validation      # once, on a login node (needs internet)
#   mkdir -p logs
#   sbatch -J real_base real_probe.sbatch OpenGVLab/InternVL3_5-1B-HF
#   sbatch -J real_ft   real_probe.sbatch /path/to/finetuned-internvl-hf
#SBATCH -J real_probe                         # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=64GB                    # Memory per GPU
#SBATCH -t4:00:00                             # Duration of the job
#SBATCH --exclude=atl1-1-03-017-30-0           # slow node
#SBATCH -o logs/%x.%j.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the gap_hybrid_probes folder (env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
SPLIT=validation     # the 1,000 validation puzzles

set -euo pipefail
MODEL=${1:?usage: sbatch real_probe.sbatch <model id or checkpoint folder>}
[ -f real_probe/rules_$SPLIT.json ] || { echo "run 'python pgm_real_probe.py rules --data \$DATA --split $SPLIT' first"; exit 1; }
for stage in order ruleid infer; do
  python pgm_real_probe.py $stage --data "$DATA" --split $SPLIT --model "$MODEL" --batch_size 16
done
python pgm_real_probe.py report --split $SPLIT
