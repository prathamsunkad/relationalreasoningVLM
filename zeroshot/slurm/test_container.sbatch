#!/bin/bash
#SBATCH -J test_container
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=16GB
#SBATCH -t00:10:00
#SBATCH -o /home/hice1/psunkad3/scratch/logs/test_container_%j.log

apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    /usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif \
    python /home/hice1/psunkad3/scratch/pgm-vlm-reasoning/zeroshot/pgm_infer.py \
    --npz_dir /home/hice1/psunkad3/scratch/pgm_data/test_full/neutral \
    --output_dir /home/hice1/psunkad3/scratch/pgm_data/results_test \
    --model_id Qwen/Qwen3-VL-2B-Instruct \
    --shard_idx 0 \
    --num_shards 1000 \
    --max_examples 3 \
    --max_new_tokens 20