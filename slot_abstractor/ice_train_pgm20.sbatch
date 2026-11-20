#!/bin/bash
# 4-GPU variant (the paper's layout: 4 x batch 24). The 1-GPU sa_*.sbatch scripts are what produced our results.
# Step 2: train Slot Abstractor on the 240k-question subset (paper settings: img 80,
# depth 24, lr 8e-5, batch 24 per GPU on 4 GPUs, up to 100 epochs, best-on-validation).
# Resubmit until 100 epochs are done or validation accuracy has plateaued.
#SBATCH -J sa_pgm20
#SBATCH -N 1
#SBATCH --ntasks-per-node=4       # one task per GPU
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 08:00:00
#SBATCH -o sa_pgm20_%j.log
cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
DATA=$SHEET_DATA
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS

srun python -u run_pgm_jsonl.py --mode train --jsonl_root $DATA --layout layout.json \
    --slot_checkpoint $SA_WEIGHTS/slot_ae_pgm20.pth.tar --save_path $SA_WEIGHTS/ --tag pgm20 \
    --img_size 80 --batch_size 24 --depth 24 --learning_rate 0.00008 --num_epochs 100 \
    --max_hours 7.5 --resume
