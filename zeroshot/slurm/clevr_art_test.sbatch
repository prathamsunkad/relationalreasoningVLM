#!/bin/bash
#SBATCH -J clevr_qwen_rmts
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 02:00:00
#SBATCH --array=0-7
#SBATCH -o /home/hice1/psunkad3/scratch/InternVL3.51B_CLEVRART_test/logs_rmts/%A_%a.log

mkdir -p /home/hice1/psunkad3/scratch/InternVL3.51B_CLEVRART_test/logs_rmts

CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif

apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python /home/hice1/psunkad3/scratch/pgm-vlm-reasoning/zeroshot/clevr_art_infer.py \
    --dataset_root /home/hice1/psunkad3/scratch/clevr_art_data/clevr-art-ood-test-seed1-v1 \
    --task rmts \
    --output_dir /home/hice1/psunkad3/scratch/InternVL3.51B_CLEVRART_test/results_internvl_rmts \
    --model_id OpenGVLab/InternVL3_5-1B-HF \
    --shard_idx $SLURM_ARRAY_TASK_ID \
    --num_shards 8 \
    --max_new_tokens 16