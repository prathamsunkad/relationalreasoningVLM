#!/bin/bash
# Step 1: pretrain the slot autoencoder on the panels of the 20% training subset only (no labels), 1 GPU.
# Resubmit the same command until the log says "finished"; --resume continues where it stopped.
# usage (from slot_abstractor/):  mkdir -p logs && sbatch sa_pretrain_ae.sbatch
#SBATCH -J sa_pretrain_ae                     # Job name
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%J.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
mkdir -p "$SA_WEIGHTS"

python -u pretrain_slot_ae_pgm.py --jsonl_root "$SHEET_DATA" --layout layout.json \
    --out "$SA_WEIGHTS/slot_ae_pgm20.pth.tar" --steps 300000 --max_hours 5.9 --resume
