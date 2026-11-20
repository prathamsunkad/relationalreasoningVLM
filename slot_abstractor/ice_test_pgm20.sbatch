#!/bin/bash
# 4-GPU variant (the paper's layout: 4 x batch 24). The 1-GPU sa_*.sbatch scripts are what produced our results.
# Step 3: evaluate the best checkpoint on validation (1k) and the full neutral test (200k).
# Slot initialisation is random in the released model, so evaluate a few seeds with the
# SAME number of GPUs and report mean +- sd.
#SBATCH -J sa_pgm20_test
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH -t 04:00:00
#SBATCH -o sa_pgm20_test_%j.log
cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
DATA=$SHEET_DATA
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$SLURM_NTASKS
CKPT=$SA_WEIGHTS/slot_abstractor_pgm_pgm20_d24_lr8e-05_run0_best.pth.tar

for SEED in 0 1 2; do
  srun python -u run_pgm_jsonl.py --mode test --jsonl_root $DATA --layout layout.json \
      --slot_checkpoint $SA_WEIGHTS/slot_ae_pgm20.pth.tar --model_checkpoint $CKPT --save_path $SA_WEIGHTS/ \
      --img_size 80 --batch_size 24 --depth 24 --seed $SEED
done
# writes <checkpoint>.seed<k>.eval.json and per-question <checkpoint>.seed<k>.test_preds.csv
