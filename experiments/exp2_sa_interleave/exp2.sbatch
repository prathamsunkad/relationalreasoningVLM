#!/bin/bash
# Exp2: pretrained Slot Abstractor interleaved into the fine-tuned InternVL3.5-1B (two MLP connectors).
# Submit from gap_hybrid_probes/ (paths: env_local.sh here, and the repo's env.sh for the .npz data):
#   mkdir -p logs
#   sbatch -J e2_inspect    exp2/exp2.sbatch inspect        # SA alone on the 1,000 shared val problems: must give ~62.8% (its best val)
#   sbatch -J e2_comp       exp2/exp2.sbatch complement     # SA vs VLM on val1000 + test5000: 2x2 table, ensemble
#   sbatch -J e2_a          exp2/exp2.sbatch train_a        # Exp2a: only the connectors train
#   sbatch -J e2_a_ctrl     exp2/exp2.sbatch train_a_ctrl   #   control: each puzzle gets another puzzle's SA features
#   sbatch -J e2_b          exp2/exp2.sbatch train_b        # Exp2b: connectors + LoRA, branch dropout 0.5, from e2a's best
#   sbatch -J e2_b_ctrl     exp2/exp2.sbatch train_b_ctrl   #   control: same steps and prompt, tokens always zero (LoRA alone)
#   sbatch -J e2_eval       exp2/exp2.sbatch eval exp2a none    # test5000; ablations: none zero shuffle_cand shuffle_puzzle
# Training resumes from its state.pt: resubmit the same command after a time-out.
#SBATCH -J exp2                               # Job name (override with -J)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=8                     # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=64GB                    # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file (job name, job id)
#SBATCH --exclude=atl1-1-03-017-30-0,atl1-1-03-018-2-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0   # the repo's BAD_NODES
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the gap_hybrid_probes folder (env_local.sh not found)"; exit 1; }
source ../env.sh || { echo "the repo's env.sh is not one folder up"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1      # models must already be in $HF_HOME, as for the repo's jobs

NPZ=${NPZ_DATA:-$DATA}                        # repo env.sh: {train_full,val_full,test_full}/neutral (train_full = the 240k set)
MODEL=$INTERNVL_MODEL                         # OpenGVLab/InternVL3_5-1B-HF (pinned in env.sh)
TEST_LIST=$REPO/results/splits/test5000.tsv   # the 5,000 test problems every hybrid in the repo is scored on
A_STEPS=6000          # Exp2a length (the repo's alignment runs read the branch within ~1k steps)
B_EPOCHS=1            # Exp2b length, passes over the 240k
DROPOUT=0.5           # Exp2b branch dropout (the repo's H9 recipe; 0 gave a shortcut in H8)

set -euo pipefail
MODE=${1:-}
for f in exp2/exp2_sa_interleave.py exp2/sa_interface.py; do [ -f "$f" ] || { echo "$f not found in $PWD"; exit 1; }; done
[ -d "$NPZ/val_full/neutral" ] || { echo "no .npz data at $NPZ/val_full/neutral. If DATA is exported for the contact sheets, unset it
  or export NPZ_DATA=<folder with train_full/val_full/test_full>; the shared copy needs the grant in 00_START_HERE/CHECKPOINTS.md"; exit 1; }
[ -f "$SA_CKPT" ] || { echo "Slot Abstractor checkpoint $SA_CKPT not found: set SA_WEIGHTS (or SA_CKPT) in env_local.sh"; exit 1; }
python -c "import einops, colorlog" 2>/dev/null || { echo "the Slot Abstractor code needs: pip install einops colorlog (in $CONDA_ENV)"; exit 1; }
[ -d "$IVL_ADAPTER" ] || { echo "fine-tuned adapter $IVL_ADAPTER not found"; exit 1; }
mkdir -p "$EXP2_RUNS"
echo "== $MODE on $(hostname), $(date)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

PY="python exp2/exp2_sa_interleave.py"
SA="--sa_builder $SA_BUILDER --sa_ckpt $SA_CKPT"
VLM="--model_id $MODEL --adapter $IVL_ADAPTER"
TRAIN="--train_dir $NPZ/train_full/neutral --val_dir $NPZ/val_full/neutral --val_list $VAL_LIST
       --file_cache_dir $EXP2_RUNS/filecache --time_limit_min 340 --num_workers 6"

case $MODE in
  inspect)
    $PY inspect $SA --val_dir $NPZ/val_full/neutral --val_list $VAL_LIST --n 1000 ;;
  complement)
    $PY complement $SA $VLM --npz_dir $NPZ/val_full/neutral  --file_list $VAL_LIST   --out_dir $EXP2_RUNS/complement_val1000 --batch_size 4
    $PY complement $SA $VLM --npz_dir $NPZ/test_full/neutral --file_list $TEST_LIST --out_dir $EXP2_RUNS/complement_test5000 --batch_size 4 ;;
  train_a)
    $PY train --stage a $SA $VLM $TRAIN --max_steps $A_STEPS --out_dir $EXP2_RUNS/exp2a ;;
  train_a_ctrl)
    $PY train --stage a $SA $VLM $TRAIN --max_steps $A_STEPS --control shuffle_puzzle --out_dir $EXP2_RUNS/exp2a_ctrl ;;
  train_b)
    [ -f $EXP2_RUNS/exp2a/best.pt ] || { echo "run train_a first (exp2b starts from its connectors)"; exit 1; }
    $PY train --stage b $SA $VLM $TRAIN --epochs $B_EPOCHS --branch_dropout $DROPOUT \
        --init_from $EXP2_RUNS/exp2a/best.pt --out_dir $EXP2_RUNS/exp2b ;;
  train_b_ctrl)
    [ -f $EXP2_RUNS/exp2a/best.pt ] || { echo "run train_a first"; exit 1; }
    $PY train --stage b $SA $VLM $TRAIN --epochs $B_EPOCHS --control zero \
        --init_from $EXP2_RUNS/exp2a/best.pt --out_dir $EXP2_RUNS/exp2b_ctrl ;;
  eval)
    TAG=${2:?usage: exp2.sbatch eval <run tag, e.g. exp2a> <none|zero|shuffle_cand|shuffle_puzzle>}
    ABL=${3:-none}
    $PY eval $SA --ckpt $EXP2_RUNS/$TAG/best.pt --npz_dir $NPZ/test_full/neutral --file_list $TEST_LIST \
        --ablate $ABL --out_dir $EXP2_RUNS/$TAG/test5000_$ABL --batch_size 4
    python ../lora/eval_vlm.py --merge_dir $EXP2_RUNS/$TAG/test5000_$ABL --expected_n 5000 ;;
  *)
    echo "usage: sbatch exp2/exp2.sbatch inspect|complement|train_a|train_a_ctrl|train_b|train_b_ctrl|eval <tag> <ablate>"; exit 1 ;;
esac
echo "== $MODE done: $(date)"
