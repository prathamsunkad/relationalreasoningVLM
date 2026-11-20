#!/bin/bash
# Stage-0 slot-source ablation on PGM 20%: one job per slot source, 1 H200 each, 6 h.
# Submit again after a time-out; unfinished variants resume from their last checkpoint, finished ones exit.
# usage (from experiments/exp1_gated_hybrid/; pgm_gap.py and layout.json come from probes/ via ../../env_local.sh):
#   mkdir -p logs
#   sbatch -J s0_vit_L12 stage0_ablation.sbatch 2                  # one variant as its own job
#   for i in 0 1 2 3 4 5 6; do sbatch -J s0_$i stage0_ablation.sbatch $i; done   # all 7 as separate jobs
#   sbatch --array=0-6 stage0_ablation.sbatch                       # or all 7 as one array job
#SBATCH -J s0_ablation                        # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file (job name, job id)
#SBATCH --mail-type=BEGIN,END,FAIL            # Mail preferences

cd $SLURM_SUBMIT_DIR
source ../../env_local.sh || { echo "run sbatch from experiments/exp1_gated_hybrid (../../env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"   # the env with transformers + peft

DATA=$SHEET_DATA                                         # set in env_local.sh
MODEL=OpenGVLab/InternVL3_5-1B-HF

#            index:  0          1          2           3           4          5     6
VARIANTS=(  "vit 0"    "vit 6"    "vit 12"    "vit 18"    "vit -1"   "cnn" "resnet" )
NAMES=(     vit_L0     vit_L6     vit_L12     vit_L18     vit_last   cnn   resnet   )
ABL_TRAIN=60000      # training puzzles per variant (identical for all)
EPOCHS=2
RECON=1.0            # slot reconstruction loss weight (identical for all)
SEED=0

# --------------------------------------------------------------------------------------------
set -euo pipefail
i=${1:-${SLURM_ARRAY_TASK_ID:-}}
if [[ -z "$i" || ! "$i" =~ ^[0-6]$ ]]; then
  echo "give the variant index 0-6: sbatch stage0_ablation.sbatch <index>  (or use --array=0-6)"; exit 1
fi
read -r SRC LAYER <<< "${VARIANTS[$i]}"
LAYER=${LAYER:--1}
NAME=${NAMES[$i]}
OUT=ckpt/ablation/$NAME

for f in train_hybrid.py relational_fusion.py "$PROBES/pgm_gap.py" "$LAYOUT"; do
  [ -f "$f" ] || { echo "$f not found in $PWD"; exit 1; }
done
[ -f "$DATA/train.jsonl" ] || { echo "No train.jsonl in DATA=$DATA"; exit 1; }
if [ -f "$OUT/summary.json" ]; then echo "$NAME already finished ($OUT/summary.json)"; exit 0; fi

echo "== variant $i: $NAME (source $SRC, layer $LAYER) on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_hybrid.py stage0 \
  --data "$DATA" --model "$MODEL" --seed "$SEED" --out "$OUT" \
  --slot_source "$SRC" --slot_layer "$LAYER" --recon_weight "$RECON" \
  --train_limit "$ABL_TRAIN" --epochs "$EPOCHS" --select acc --workers 12

echo "== $NAME done: $(date)"
