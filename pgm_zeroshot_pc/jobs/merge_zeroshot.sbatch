#!/bin/bash
#SBATCH -J pgm-zs-merge
#SBATCH -N1 --ntasks-per-node=1 --cpus-per-task=2 --mem=16G -t 0:30:00
#SBATCH -p ice-cpu
#SBATCH -o /home/hice1/khsieh37/scratch/pgm-zeroshot-eval/logs/merge-%j.log
# Runs after the eval array (sbatch --dependency=afterany:<array id>): merges each model's 28 shard files with the
# bundle's own eval_vlm.py --merge_dir (expected 200,000 problems), then writes the per-model accuracy, Wilson 95% CI,
# Yes-rate and predicted-position histogram to results/zeroshot_summary.txt.
set -uo pipefail
W=$HOME/scratch/pgm-zeroshot-eval
B=$HOME/scratch/pgm-lora-v1
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
py () { apptainer exec --env PYTHONPATH="$B/env:$B/code" --env PYTHONNOUSERSITE=1 --env HF_HOME="$B/hf_cache" \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" python "$@"; }
for M in internvl qwen; do
  echo "$M: $(ls $W/results/${M}_zeroshot/results_shard*.json | wc -l) shard files"
  py "$B/code/lora/eval_vlm.py" --merge_dir "$W/results/${M}_zeroshot" --expected_n 200000
done
py "$W/scripts/summ.py" "$W/results" | tee "$W/results/zeroshot_summary.txt"
echo "=== merge done ($(date))"
