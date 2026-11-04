#!/bin/bash
#SBATCH -J pgm_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t4:00:00
#SBATCH --array=0-9
#SBATCH -o /home/hice1/psunkad3/scratch/logs/pgm_eval_%A_%a.log

mkdir -p /home/hice1/psunkad3/scratch/logs

CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif

apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python /home/hice1/psunkad3/scratch/pgm_infer.py \
    --npz_dir /home/hice1/psunkad3/scratch/pgm_data/test_full/neutral \
    --output_dir /home/hice1/psunkad3/scratch/pgm_data/results \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards 10 \
    --max_examples 500 \
    --max_new_tokens 20