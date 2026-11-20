#!/bin/bash
# 4-GPU variant (the paper's layout: 4 x batch 24). The 1-GPU sa_*.sbatch scripts are what produced our results.
# Step 1: pretrain the slot autoencoder on panels from the 20% training subset only.
# Resubmit the same script until the log says "finished"; --resume picks up where it stopped.
#SBATCH -J slot_ae_pgm20
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 08:00:00               # set --max_hours ~0.5 h below this
#SBATCH -o slot_ae_pgm20_%j.log
cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
DATA=$SHEET_DATA

python -u pretrain_slot_ae_pgm.py --jsonl_root $DATA --layout layout.json \
    --out $SA_WEIGHTS/slot_ae_pgm20.pth.tar --steps 300000 --max_hours 7.5 --resume
