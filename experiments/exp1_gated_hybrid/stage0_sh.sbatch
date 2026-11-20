#!/bin/bash
# Stage-0 architecture search by successive halving: one job = one slot source trained up to MAX_STEPS.
# The learning rate is flat after warmup, so the next rung simply re-runs the same variant with a larger
# MAX_STEPS and it continues from its checkpoint. Also resubmit the same command after a time-out.
# usage (from experiments/exp1_gated_hybrid/; pgm_gap.py and layout.json come from probes/ via ../../env_local.sh):
#   sbatch -J sh_vit_L6 stage0_sh.sbatch 1 20000        # variant 1 (vit_L6) up to 20,000 steps, seed 0
#   for i in 0 1 2 3 4 5 6; do sbatch -J sh_$i stage0_sh.sbatch $i 20000; done      # rung 1: all variants
#   sbatch -J sh_vit_L6_s1 stage0_sh.sbatch 1 60000 1   # a second seed (own folder ckpt/sh/seed1/)
# Results: ckpt/sh_v2/seed<SEED>/<variant>/ ; rank them with  python rank_stage0.py ckpt/sh_v2/seed0
# v2 = with the abstractor fix (qk_norm). The rung-1 runs without it are left untouched in ckpt/sh/.
#SBATCH -J stage0_sh                          # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file (job name, job id)
#SBATCH --exclude=atl1-1-03-017-30-0           # its GPU ran the ViT ~4x slower than other nodes in rung 1
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ../../env_local.sh || { echo "run sbatch from experiments/exp1_gated_hybrid (../../env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
CACHE=$PANEL_CACHE                                       # built by build_cache.sbatch; set CACHE="" to skip
MODEL=OpenGVLab/InternVL3_5-1B-HF

#            index:  0          1          2           3           4          5     6
VARIANTS=(  "vit 0"    "vit 6"    "vit 12"    "vit 18"    "vit -1"   "cnn" "resnet" )
NAMES=(     vit_L0     vit_L6     vit_L12     vit_L18     vit_last   cnn   resnet   )
TAG=v2               # results folder ckpt/sh_$TAG ; change it for a new search so old checkpoints are not resumed
QK_NORM=1            # the abstractor fix (1); 0 = the rung-1 behaviour
HOLDOUT=10000        # validation = last 10,000 rows of train.jsonl (not trained on); identical for all
RECON=1.0            # slot reconstruction loss weight; identical for all
LR=1e-4              # 3e-4 collapsed in lr_check; rung-1 v2 used 1e-4
WARMUP=500
EVAL_EVERY=2500

# --------------------------------------------------------------------------------------------
set -euo pipefail
i=${1:-}
MAX_STEPS=${2:-}
SEED=${3:-0}
if [[ ! "$i" =~ ^[0-6]$ || ! "$MAX_STEPS" =~ ^[0-9]+$ || ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "usage: sbatch stage0_sh.sbatch <variant 0-6> <max_steps> [seed]"; exit 1
fi
read -r SRC LAYER <<< "${VARIANTS[$i]}"
LAYER=${LAYER:--1}
NAME=${NAMES[$i]}
OUT=ckpt/sh_$TAG/seed$SEED/$NAME
for f in train_hybrid.py relational_fusion.py "$PROBES/pgm_gap.py" "$LAYOUT"; do
  [ -f "$f" ] || { echo "$f not found in $PWD"; exit 1; }
done
CACHE_FLAG=""
[ -n "$CACHE" ] && CACHE_FLAG="--cache $CACHE"

echo "== $NAME (source $SRC, layer $LAYER, seed $SEED) up to step $MAX_STEPS on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python train_hybrid.py stage0 \
  --data "$DATA" --model "$MODEL" --seed "$SEED" --out "$OUT" $CACHE_FLAG \
  --slot_source "$SRC" --slot_layer "$LAYER" --recon_weight "$RECON" --qk_norm "$QK_NORM" \
  --schedule constant --max_steps "$MAX_STEPS" --warmup_steps "$WARMUP" --lr_rel "$LR" \
  --val_from_train "$HOLDOUT" --eval_every "$EVAL_EVERY" --select loss \
  --eval_batch_size 32 --workers 12

echo "== $NAME reached step $MAX_STEPS: $(date)"
