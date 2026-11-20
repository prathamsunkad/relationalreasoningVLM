#!/bin/bash
#SBATCH --job-name=pgm-hybrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=pgm_hybrid_%j.log
#
# Hybrid (InternVL + gated relational module) and its matched LoRA baseline.
# Run from the folder holding run_hybrid.sh, train_hybrid.py, relational_fusion.py, pgm_gap.py, layout.json:
#
# Run run_stage0_ablation.sh first: it writes stage0_choice.txt (where the slots get their features),
# which stage 0 here uses. Without it, stage 0 uses the defaults (last ViT layer, no reconstruction).
#
#   bash run_hybrid.sh all        # every stage in order (finished stages are skipped)
#   bash run_hybrid.sh stage0     # or one step: stage0 | stage1 | stage2 | baseline | eval
#   sbatch run_hybrid.sh all      # resubmit the same command after a time-out: it resumes
#
# ---------------- settings ----------------
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/env_local.sh"
DATA="$SHEET_DATA"                     # set in env_local.sh
MODEL="OpenGVLab/InternVL3_5-1B-HF"
SEED=0
EXTRA=""        # extra flags for every training stage, e.g. "--lora_r 16 --lora_alpha 32 --lr_lora 1e-4"
                # (set the LoRA flags to the values your earlier baselines used)
# ------------------------------------------
set -euo pipefail
STEP=${1:-all}
OUT=ckpt/seed$SEED
for f in train_hybrid.py relational_fusion.py pgm_gap.py layout.json; do
  [ -f "$f" ] || { echo "$f not found: run from the folder that holds the scripts and layout.json"; exit 1; }
done
[ -f "$DATA/train.jsonl" ] || { echo "No train.jsonl in DATA=$DATA: edit the settings at the top"; exit 1; }

stage() {   # stage <name> <mode> [flags]: skipped once its summary.json exists
  local name=$1; shift
  if [ -f "$OUT/$name/summary.json" ]; then echo "== $name: done, skipping"; return; fi
  echo; echo "== $name"
  python train_hybrid.py "$@" --data "$DATA" --model "$MODEL" --seed "$SEED" --out "$OUT/$name" $EXTRA
}
evaluate() {   # evaluate <checkpoint> [control flag]
  echo; echo "== eval $1 ${2:-}"
  python train_hybrid.py eval --data "$DATA" --model "$MODEL" --init "$OUT/$1" --split test ${2:-}
}

if [[ $STEP == all || $STEP == stage0 ]]; then
  S0=""
  if [ -f stage0_choice.txt ]; then S0=$(cat stage0_choice.txt); echo "stage 0 architecture (stage0_choice.txt): $S0"
  else echo "stage 0 architecture: defaults (no stage0_choice.txt; run run_stage0_ablation.sh to choose)"; fi
  stage stage0_A stage0 --half A $S0
  stage stage0_B stage0 --half B $S0
fi
[[ $STEP == all || $STEP == stage1 ]]   && stage stage1 stage1 --branches "$OUT/stage0_A" "$OUT/stage0_B"
[[ $STEP == all || $STEP == stage2 ]]   && stage stage2 stage2 --init "$OUT/stage1"
[[ $STEP == all || $STEP == baseline ]] && stage baseline baseline
if [[ $STEP == all || $STEP == eval ]]; then
  evaluate stage2
  evaluate stage2 --no_fusion
  evaluate stage2 --shuffle_relational
  evaluate baseline
fi
echo; echo "Done. Checkpoints and eval_test*.jsonl are in $OUT/"
