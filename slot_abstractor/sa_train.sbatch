#!/bin/bash
# Step 2: Slot Abstractor on PGM 20% (240k), 1 H200, 6 h per job. Submit again after it ends; it resumes
# (mid-epoch too) until NUM_EPOCHS are done. Batch 96 on 1 GPU = the paper's global batch (4 GPUs x 24).
# usage (from slot_abstractor/):  mkdir -p logs && sbatch sa_train.sbatch
#SBATCH -J sa_train                           # Job name
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%J.log                     # Output file
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
NUM_EPOCHS=80

python -u run_pgm_jsonl.py --mode train \
    --jsonl_root "$SHEET_DATA" --layout layout.json \
    --slot_checkpoint "$SA_WEIGHTS/slot_ae_pgm20.pth.tar" --save_path "$SA_WEIGHTS/" \
    --tag "$SA_TAG" --batch_size 96 --learning_rate 0.00008 \
    --num_workers 14 --num_epochs $NUM_EPOCHS \
    --max_hours 5.9 --resume --snapshot_every 5
