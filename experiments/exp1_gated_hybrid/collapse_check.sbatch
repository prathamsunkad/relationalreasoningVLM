#!/bin/bash
# Quick check that the abstractor fix (qk_norm) stops the collapse, on the real data, before rung 1 is redone.
# CNN slots (the variant that collapsed from the start in rung 1, and the fastest: ~330 puzzles/s on an H200).
# Four short runs, one after another, each 6,000 steps on the full pool (~5-10 min each):
#   cnn_old       qk_norm 0  (as in rung 1)                    expected: collapses again
#   cnn_fix       qk_norm 1                                    the fix
#   cnn_fix_r01   qk_norm 1, reconstruction weight 0.1         does a weaker reconstruction loss help?
#   cnn_fix_lr1   qk_norm 1, learning rate 1e-4                does a lower learning rate help?
# usage (from gap_hybrid_probes/):  mkdir -p logs && sbatch collapse_check.sbatch
#SBATCH -J collapse_check                     # Job name
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t2:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file
#SBATCH --mail-type=END,FAIL                  # Mail preferences

cd $SLURM_SUBMIT_DIR
source ./env_local.sh || { echo "run sbatch from the gap_hybrid_probes folder (env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

DATA=$SHEET_DATA                                         # set in env_local.sh
CACHE=pgm_panel_cache
MODEL=OpenGVLab/InternVL3_5-1B-HF

set -euo pipefail
COMMON="--data $DATA --model $MODEL --cache $CACHE --slot_source cnn --schedule constant --warmup_steps 500
        --max_steps 6000 --val_from_train 10000 --eval_every 1000 --log_every 100 --select loss
        --eval_batch_size 32 --workers 12 --seed 0"
run() { name=$1; shift; echo; echo "== $name: $*"; python train_hybrid.py stage0 $COMMON --out ckpt/collapse/$name "$@"; }

run cnn_old     --qk_norm 0 --recon_weight 1.0 --lr_rel 3e-4
run cnn_fix     --qk_norm 1 --recon_weight 1.0 --lr_rel 3e-4
run cnn_fix_r01 --qk_norm 1 --recon_weight 0.1 --lr_rel 3e-4
run cnn_fix_lr1 --qk_norm 1 --recon_weight 1.0 --lr_rel 1e-4

echo; python rank_stage0.py ckpt/collapse
echo; echo "Collapse check (last 10 log lines of each run): identical-score share, answer loss, attention saturation"
python - <<'PY'
import glob, json, os
for d in sorted(glob.glob("ckpt/collapse/*")):
    rows = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl")) if l.startswith('{"step"')][-10:]
    if not rows: continue
    flat = sum(r.get("spread", 1) < 1e-3 for r in rows) / len(rows)
    ce = sum(r["ce"] for r in rows) / len(rows)
    am = sum(r.get("attn_max", float("nan")) for r in rows) / len(rows)
    print(f"{os.path.basename(d):12s} identical scores {flat:4.0%}   answer loss {ce:.3f}   attn_max {am:.3f}   spread {rows[-1].get('spread')}")
PY
