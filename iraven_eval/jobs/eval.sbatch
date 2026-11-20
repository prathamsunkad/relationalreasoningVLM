#!/bin/bash
#SBATCH -J iraven-eval
#SBATCH -p ice-gpu
#SBATCH -N1 -n1 -c8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -C intel
#SBATCH -t 06:00:00
#SBATCH -o /home/hice1/khsieh37/iraven_eval/logs/eval-%x-%j.log
# Usage: sbatch --export=ALL,KEY=lfm25-pgm-best jobs/eval.sbatch
# "-C intel" keeps the job on NVIDIA nodes (the AMD-GPU nodes on ICE are all AMD-CPU).
set -euo pipefail
W=$HOME/iraven_eval
KEY=${KEY:?set KEY to an adapter key}
BUNDLE=$HOME/scratch/pgm-best-lora-2026-11-01-v1
PGM=$HOME/scratch/pgm-neutral-20percent_data/pgm-neutral-train20pct-seed20261007-v1/pgm-neutral-train20pct-seed20261007-v1
DATA=$W/data/i-raven-test-seed1234-v1
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
module load python/3.11.9
source "$W/venv/bin/activate"
nvidia-smi --query-gpu=name,memory.total --format=csv
if [ ! -f "$DATA/test.jsonl" ]; then
  python "$W/scripts/build_sheets.py" --raw "$W/data/raw" --template "$W/template/pgm_contact_sheet_template.png" \
    --prompts "$BUNDLE/prompts.json" --out "$DATA" --generator-commit "$(cat $W/generator/SRAN_COMMIT.txt)"
fi
# 1) Pipeline check: first 1,000 PGM neutral test questions (published full-test accuracy is in INDEX.json).
python "$W/scripts/eval_adapter.py" --bundle "$BUNDLE" --key "$KEY" --manifest "$PGM/test.jsonl" --tag pgm-neutral-test-first1000 \
  --limit 1000 --out-dir "$W/results"
# 2) I-RAVEN test split (14,000 puzzles, 7 configurations).
python "$W/scripts/eval_adapter.py" --bundle "$BUNDLE" --key "$KEY" --manifest "$DATA/test.jsonl" --tag i-raven-test \
  --out-dir "$W/results"
