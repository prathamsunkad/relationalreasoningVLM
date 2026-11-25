#!/bin/bash
#SBATCH -J pgm-zs-eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --exclude=atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0
#SBATCH --mem=32G
#SBATCH -t 10:00:00
#SBATCH --array=0-27
#SBATCH -o /home/hice1/khsieh37/scratch/pgm-zeroshot-eval/logs/eval-%x-%A_%a.log
# Zero-shot (no LoRA adapter) evaluation on the 200k PGM neutral test split, with the pgm-lora-v1 bundle's own code,
# packages, base models and container: the same per-candidate Yes/No input and score as the LoRA runs and eval_test.sbatch.
# The only differences from `run.sh eval-shard`: no --adapter, and each shard's .npz files are unpacked into node-local /tmp.
# Each array task scores its shard with every model in MODELS (default: internvl then qwen), so both models fit in one
# 28-task array (ICE allows 50 submitted jobs per user). Re-running a task resumes where it stopped.
#   sbatch jobs/eval_zeroshot.sbatch            or   sbatch --export=ALL,MODELS=internvl jobs/eval_zeroshot.sbatch
set -euo pipefail
MODELS=${MODELS:-internvl qwen}
W=$HOME/scratch/pgm-zeroshot-eval
B=$HOME/scratch/pgm-lora-v1
K=$SLURM_ARRAY_TASK_ID
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
py () { apptainer exec --nv --env PYTHONPATH="$B/env:$B/code" --env PYTHONNOUSERSITE=1 --env HF_HOME="$B/hf_cache" \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" python "$@"; }
D=${TMPDIR:-/tmp}/pgmzs_${SLURM_JOB_ID}; mkdir -p $D; trap 'rm -rf $D' EXIT
py - <<PY
import numpy as np
z = np.load('$W/packs/pack$(printf %02d $K).npz')
for img, t, f in zip(z['image'], z['target'], z['name']):
    np.savez('$D/' + str(f), image=img, target=np.int64(t))
print('unpacked', len(z['name']), 'problems', flush=True)
PY
nvidia-smi --query-gpu=name --format=csv,noheader
for MODEL in $MODELS; do
  case $MODEL in qwen) BASE=$B/models/Qwen3-VL-2B-Instruct ;; internvl) BASE=$B/models/InternVL3_5-1B-HF ;;
                 *) echo "unknown model $MODEL"; exit 2 ;; esac
  OUT=$W/results/${MODEL}_zeroshot; mkdir -p $OUT/shard$K
  echo "=== $MODEL shard $K ($(date))"
  py "$B/code/lora/eval_vlm.py" --model_id "$BASE" --npz_dir "$D" --out_dir "$OUT/shard$K" --shard_idx 0 --num_shards 1 \
     --batch_size 4 --num_workers 4
  cp "$OUT/shard$K/results_shard0.json" "$OUT/results_shard$K.json"
  cp "$OUT/shard$K/eval_config.json" "$OUT/eval_config.json"
done
echo "=== done ($(date))"
