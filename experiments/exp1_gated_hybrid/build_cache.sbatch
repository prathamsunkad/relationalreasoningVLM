#!/bin/bash
# Build the panel cache once (CPU only): crops every puzzle's 16 panels into uint8 arrays so training
# skips PNG decoding / cropping / resizing. Prints the size first and stops if the disk is too small.
# usage (from the folder with train_hybrid.py, pgm_gap.py, layout.json):  mkdir -p logs && sbatch build_cache.sbatch
#SBATCH -J build_cache                        # Job name
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --mem=64GB                            # Memory
#SBATCH -t3:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the gap_hybrid_probes folder (env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
CACHE=$PANEL_CACHE      # needs roughly 240k x 16 x panel pixels bytes
MODEL=OpenGVLab/InternVL3_5-1B-HF                            # only its image processor, for the final check

set -euo pipefail
python train_hybrid.py cache --data "$DATA" --cache "$CACHE" --model "$MODEL" --splits train validation --workers 16
