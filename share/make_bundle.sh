#!/bin/bash
# Assemble the shareable adapter bundle (code + adapters for every epoch + results + sample problems + setup/check
# scripts; no packages, base models or datasets: the recipient's setup.sbatch installs those).
#   bash share/make_bundle.sh [OUT_DIR]          default OUT_DIR: ~/scratch/share/pgm-lora-v1
# Afterwards record the reference scores once from this (original) environment:
#   sbatch share/record_expected.sbatch OUT_DIR
set -euo pipefail
cd "$(dirname "$0")/.."; source env.sh
OUT=${1:-/home/hice1/psunkad3/scratch/share/pgm-lora-v1}
[ -e "$OUT" ] && { echo "$OUT already exists; remove it or choose another folder"; exit 1; }
mkdir -p "$OUT"/{code/tools,code/lora,adapters,results,sample_problems}

# code: only what inference + evaluation need
cp -r pgmvlm "$OUT/code/"; cp lora/__init__.py lora/eval_vlm.py "$OUT/code/lora/"; cp tools/demo_predict.py "$OUT/code/tools/"
cp share/check.py "$OUT/code/"; find "$OUT/code" -name __pycache__ -prune -exec rm -rf {} +
# entry points + docs
cp share/run.sh share/setup.sbatch share/check.sbatch share/eval_test.sbatch share/requirements-lock.txt "$OUT/"
cp share/BUNDLE_README.md "$OUT/README.md"

# adapters: the end-of-epoch adapter of every epoch, plus a best-validation checkpoint where it differs from the final one
add () {  # src dst
    mkdir -p "$OUT/adapters/$2"; cp "$1"/adapter_config.json "$1"/adapter_model.safetensors "$OUT/adapters/$2/"
}
Q=$RUNS/qwen_pc_recipe_e1; I=$RUNS/ivl1b_pc_recipe_e1
for e in 1 2 3 4; do add $Q/final_e$e qwen3vl2b_pgm_ep$e; add $I/final_e$e internvl35_1b_pgm_ep$e; done
add $Q/final_e5 qwen3vl2b_pgm_ep5; add $I/final_e6 internvl35_1b_pgm_ep6
same () { cmp -s "$1/adapter_model.safetensors" "$2/adapter_model.safetensors"; }
same $Q/best_e5 $Q/final_e5 || add $Q/best_e5 qwen3vl2b_pgm_ep5_bestval
same $I/best_e6 $I/final_e6 || add $I/best_e6 internvl35_1b_pgm_ep6_bestval

# results (tables, curves, validation loss); no per-problem dumps
cp results/lora/lora_pgm_results.csv results/lora/lora_pgm_results.md results/lora/val_loss_*.json \
   results/lora/qwen_curves_ep0-5.png results/lora/internvl_curves_ep0-6.png results/lora/demo_*.png "$OUT/results/"
[ -f results/lora/test_ep5_ep6.json ] && cp results/lora/test_ep5_ep6.json "$OUT/results/"

# 10 test problems (fixed choice) for the demo and the installation check
python3 - "$RUNS/filecache/test_neutral/eval_files.txt" "$DATA/test_full/neutral" "$OUT/sample_problems" <<'PY'
import random, shutil, sys
files = [l.strip() for l in open(sys.argv[1]) if l.strip()]
for f in sorted(random.Random(2026).sample(files, 10)):
    shutil.copy(f"{sys.argv[2]}/{f}", sys.argv[3])
PY
chmod +x "$OUT/run.sh"
echo "bundle assembled in $OUT ($(du -sh "$OUT" | cut -f1)); next: sbatch share/record_expected.sbatch $OUT"
