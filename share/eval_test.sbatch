#!/bin/bash
#SBATCH -J pgm_bundle_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-h200|gpu-h100|gpu-a100|gpu-l40s"
#SBATCH --exclude=atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH --array=0-27
#SBATCH -o eval_%A_%a.log
# Full test-set evaluation of one adapter, in 28 parallel shards (each problem scored with its 8 completed matrices).
#   sbatch --export=ALL,MODEL=qwen,EPOCH=5,DATA_DIR=/path/to/test_full/neutral,OUT_DIR=$PWD/results_qwen_ep5 eval_test.sbatch
#   then:  bash run.sh merge $PWD/results_qwen_ep5          (accuracy + 95% CI; re-running a shard resumes it)
# DATA_DIR = a folder of PGM .npz files (the 200,000-problem neutral test split for the reference numbers).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${MODEL:?set MODEL=qwen or internvl}" "${EPOCH:?set EPOCH}" "${DATA_DIR:?set DATA_DIR}" "${OUT_DIR:?set OUT_DIR}"
mkdir -p "$OUT_DIR"
bash run.sh eval-shard "$MODEL" "$EPOCH" "$DATA_DIR" "$OUT_DIR" "$SLURM_ARRAY_TASK_ID" 28
