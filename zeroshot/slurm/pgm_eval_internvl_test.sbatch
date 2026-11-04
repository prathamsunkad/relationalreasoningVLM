#!/bin/bash
#SBATCH -J pgm_eval_internvl
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 05:00:00
#SBATCH --array=0-27
#SBATCH -o /home/hice1/psunkad3/scratch/InternVL_PGM_test/logs_v2/pgm_eval_%A_%a.log

mkdir -p /home/hice1/psunkad3/scratch/InternVL_PGM_test/logs_v2

CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif

apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python /home/hice1/psunkad3/scratch/pgm_infer.py \
    --npz_dir /home/hice1/psunkad3/scratch/pgm_data/test_full/neutral \
    --output_dir /home/hice1/psunkad3/scratch/InternVL_PGM_test/results_v2 \
    --model_id OpenGVLab/InternVL3_5-1B-HF \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards 28 \
    --max_new_tokens 30