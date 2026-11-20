#!/bin/bash
#SBATCH --job-name=pgm-gap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=pgm_gap_%j.log
#
# Perception-vs-reasoning experiment for both VLMs.
# Run from the folder that holds run_gap.sh, pgm_gap.py and infer_layout.py:
#
#   bash run_gap.sh pilot     # 50 validation puzzles + debug images; check these first
#   bash run_gap.sh full      # everything
#   sbatch run_gap.sh full    # same, as a SLURM job
#
# ---------------- settings: edit these three lines ----------------
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/env_local.sh"
DATA="$SHEET_DATA"                     # set in env_local.sh
QWEN="Qwen/Qwen3-VL-2B-Instruct"         # base model, or the path to your fine-tuned checkpoint
INTERNVL="OpenGVLab/InternVL3_5-1B-HF"   # base model, or the path to your fine-tuned checkpoint (HF format)
# -------------------------------------------------------------------
set -euo pipefail

MODE=${1:-pilot}
case $MODE in
  pilot) LIM="--limit 50 --dump 5"; TRAIN_LIM="--limit 300";   SOLVE="--seeds 1 --epochs 10"; PROBE="--probe_train 200" ;;
  full)  LIM="";                    TRAIN_LIM="--limit 20000"; SOLVE="";                      PROBE="--probe_train 1000" ;;
  *) echo "usage: bash run_gap.sh pilot|full"; exit 1 ;;
esac
OUT=runs/$MODE

for f in pgm_gap.py infer_layout.py; do
  [ -f "$f" ] || { echo "$f not found: run this from the folder that holds the scripts"; exit 1; }
done
[ -f "$DATA/validation.jsonl" ] || { echo "No validation.jsonl in DATA=$DATA: edit the settings at the top"; exit 1; }

if [ ! -f layout.json ]; then
  echo "== inferring panel layout"
  python infer_layout.py --data "$DATA" --out layout.json
  echo "   check layout_check.png before trusting the results"
fi

run() { echo; echo "== $*"; python pgm_gap.py "$@" --data "$DATA" --layout layout.json --out "$OUT"; }

for M in "$QWEN" "$INTERNVL"; do
  echo; echo "################ $M ($MODE)"
  run image    --model "$M" --split validation $LIM
  run image    --model "$M" --split validation $LIM --variant crops --batch_size 4
  run match    --model "$M" --split validation $LIM --variant sheet
  run match    --model "$M" --split validation $LIM --variant crops
  run samediff --model "$M" --split validation $LIM
  run probe    --model "$M" --split validation $LIM $PROBE
  run describe --model "$M" --split validation $LIM
  run describe --model "$M" --split train $TRAIN_LIM
  run text     --model "$M" --split validation
  run text     --model "$M" --split validation --rules || echo "   (skipped --rules: no relation structure in the metadata)"
  python pgm_gap.py solve --model "$M" --split validation --out "$OUT" $SOLVE
done

python pgm_gap.py report --split validation --out "$OUT"
echo; echo "Done. Table: $OUT/report_validation.md"
