#!/bin/bash
# Step 3: score the best checkpoint on validation (1k) and the full test (200k), 3 seeds (slot initialisation
# is random), 1 GPU. Seeds that already have an .eval.json are skipped, so resubmit after a time-out.
# usage (from slot_abstractor/):  mkdir -p logs && sbatch sa_test.sbatch            # best checkpoint
#                                 sbatch sa_test.sbatch <path/to/checkpoint.pth.tar>  # any other one
#SBATCH -J sa_test                            # Job name
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=64GB                    # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%J.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the slot_abstractor folder"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$SA_CONDA_ENV"
CKPT=${1:-$SA_WEIGHTS/slot_abstractor_pgm_${SA_TAG}_d24_lr8e-05_run0_best.pth.tar}
[ -f "$CKPT" ] || { echo "checkpoint $CKPT not found"; exit 1; }

for SEED in 0 1 2; do
  OUT="$SA_WEIGHTS/$(basename "$CKPT").seed$SEED.eval.json"
  if [ -f "$OUT" ]; then echo "seed $SEED done: $OUT"; continue; fi
  python -u run_pgm_jsonl.py --mode test --jsonl_root "$SHEET_DATA" --layout layout.json \
      --slot_checkpoint "$SA_WEIGHTS/slot_ae_pgm20.pth.tar" --model_checkpoint "$CKPT" --save_path "$SA_WEIGHTS/" \
      --img_size 80 --batch_size 96 --depth 24 --num_workers 14 --seed $SEED
done
python summarize_runs.py --eval_glob "$SA_WEIGHTS/$(basename "$CKPT").seed*.eval.json"
