#!/bin/bash
#SBATCH -J pgm_bundle_record
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-h200|gpu-h100"
#SBATCH --exclude=atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_pc_runs/logs/bundle_record_%j.log
# Maintainers only: record the reference scores of every adapter on the bundle's sample problems from the ORIGINAL
# training environment, then write SHA256SUMS.   sbatch share/record_expected.sbatch <bundle_dir>
set -euo pipefail
OUT=${1:?bundle dir}
source /home/hice1/psunkad3/scratch/pgm-vlm-reasoning/env.sh
apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env PYTHONPATH=$OUT/code \
    $CONTAINER python $OUT/code/check.py --bundle $OUT --record --base qwen=$QWEN_MODEL --base internvl=$INTERNVL_MODEL
cd $OUT && find . -type f ! -name SHA256SUMS ! -path './env/*' ! -path './models/*' ! -path './hf_cache/*' -print0 \
    | sort -z | xargs -0 sha256sum > SHA256SUMS
echo "recorded; SHA256SUMS written ($(wc -l < SHA256SUMS) files)"
