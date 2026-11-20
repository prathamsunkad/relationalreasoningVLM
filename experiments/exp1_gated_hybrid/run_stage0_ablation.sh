#!/bin/bash
#SBATCH --job-name=stage0-ablation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=stage0_ablation_%j.log
#
# Where should the slots get their features? Tests whether InternViT's self-attention dilutes
# the per-object information slot attention needs.
#
#   1. probe    linear same/different probe on InternViT layers 0 (patch embeddings) .. 24   (minutes)
#   2. train    stage 0 (relational branch alone, no LLM) once per slot source, identical budget
#   3. choose   table of validation accuracy; the winner is written to stage0_choice.txt,
#               which run_hybrid.sh then uses for the full stage 0 (both halves) and later stages
#
#   bash run_stage0_ablation.sh            # all three steps; finished runs are skipped
#   bash run_stage0_ablation.sh choose     # only re-print the table / re-pick
#
# Run from the folder holding the scripts and layout.json.
# ---------------- settings ----------------
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/env_local.sh"
DATA="$SHEET_DATA"                     # set in env_local.sh
MODEL="OpenGVLab/InternVL3_5-1B-HF"
VARIANTS="vit:0 vit:6 vit:12 vit:18 vit:-1 cnn resnet"   # vit:L = InternViT layer L (-1 = last); cnn; resnet
ABL_TRAIN=60000      # training puzzles per variant (first N of the pool; the same for every variant)
ABL_EPOCHS=2
RECON=1.0            # slot reconstruction loss weight, identical for all variants (0 = off)
SEED=0
# ------------------------------------------
set -euo pipefail
STEP=${1:-all}
OUT=ckpt/ablation
for f in train_hybrid.py relational_fusion.py pgm_gap.py layout.json; do
  [ -f "$f" ] || { echo "$f not found: run from the folder that holds the scripts and layout.json"; exit 1; }
done
[ -f "$DATA/train.jsonl" ] || { echo "No train.jsonl in DATA=$DATA: edit the settings at the top"; exit 1; }

if [[ $STEP == all || $STEP == probe ]]; then
  echo "== probe: InternViT layers"
  python pgm_gap.py probe --data "$DATA" --layout layout.json --model "$MODEL" --split validation \
    --out runs/full --probe_layers 0,6,12,18,24 --probe_train 1000
fi

if [[ $STEP == all || $STEP == train ]]; then
  if [[ " $VARIANTS " == *" resnet "* ]]; then   # fetch the ImageNet weights once (compute nodes may be offline)
    python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')" \
      || echo "   could not download ResNet-18 weights; the resnet variant will fail"
  fi
  for v in $VARIANTS; do
    src=${v%%:*}; layer=-1; [[ $v == *:* ]] && layer=${v#*:}
    name=$src; [[ $src == vit ]] && name="vit_L$layer"
    if [ -f "$OUT/$name/summary.json" ]; then echo "== $name: done, skipping"; continue; fi
    echo; echo "== stage 0, slots from $v"
    python train_hybrid.py stage0 --data "$DATA" --model "$MODEL" --seed "$SEED" --out "$OUT/$name" \
      --slot_source "$src" --slot_layer "$layer" --recon_weight "$RECON" \
      --train_limit "$ABL_TRAIN" --epochs "$ABL_EPOCHS" --select acc \
      || echo "   $name failed; continuing with the other variants"
  done
fi

if [[ $STEP == all || $STEP == choose || $STEP == train ]]; then
  python - "$OUT" "$RECON" <<'EOF'
import json, math, os, sys
out, recon = sys.argv[1], sys.argv[2]
rows = []
for name in sorted(os.listdir(out)):
    p = os.path.join(out, name, "summary.json")
    if not os.path.exists(p):
        continue
    b = json.load(open(p))["best"]
    n, acc = b["n"], b["acc"]
    z = 1.96; den = 1 + z * z / n; c = (acc + z * z / (2 * n)) / den
    h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
    rows.append((acc, name, c - h, c + h, b["loss"], b["epoch"]))
if not rows:
    sys.exit("no finished ablation runs yet")
rows.sort(reverse=True)
print("\nStage-0 ablation (validation, chance 12.5%):\n")
print("| slots from | accuracy [95% CI] | val loss | best epoch |\n|---|---|---|---|")
for acc, name, lo, hi, loss, ep in rows:
    print(f"| {name} | {acc:.1%} [{lo:.1%}, {hi:.1%}] | {loss:.3f} | {ep} |")
best = rows[0]
ties = [r[1] for r in rows[1:] if r[0] >= best[2]]
if ties:
    print(f"\nNote: {', '.join(ties)} fall inside the winner's 95% interval, so the validation set cannot "
          "separate them. Confirm with a second seed or a larger --val_limit before relying on the ordering.")
name = best[1]
flags = (f"--slot_source vit --slot_layer {name[len('vit_L'):]}" if name.startswith("vit_L")
         else f"--slot_source {name}") + f" --recon_weight {recon}"
open("stage0_choice.txt", "w").write(flags + "\n")
print(f"\nChosen: {name}. Wrote stage0_choice.txt: {flags}")
print("Next: bash run_hybrid.sh stage0   (trains the chosen branch on both halves for cross-fitting)")
EOF
fi
