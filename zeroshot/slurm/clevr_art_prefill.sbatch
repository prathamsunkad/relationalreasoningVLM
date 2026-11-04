#!/bin/bash
#SBATCH -J clevr_prefill
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=32GB
#SBATCH -t 01:00:00
#SBATCH --array=0-31
#SBATCH -o /home/hice1/psunkad3/scratch/clevr_prefill_logs/%A_%a.log

# 4 configs x 8 shards: array index -> (config, shard)
CONFIG=$((SLURM_ARRAY_TASK_ID / 8))
SHARD=$((SLURM_ARRAY_TASK_ID % 8))

case $CONFIG in
  0) MODEL=Qwen/Qwen3-VL-2B-Instruct;   TASK=rmts;    OUT=Qwen2BInstruct_CLEVRART_test/results_qwen_rmts_prefill ;;
  1) MODEL=Qwen/Qwen3-VL-2B-Instruct;   TASK=idrules; OUT=Qwen2BInstruct_CLEVRART_test/results_qwen_idrules_prefill ;;
  2) MODEL=OpenGVLab/InternVL3_5-1B-HF; TASK=rmts;    OUT=InternVL3.51B_CLEVRART_test/results_internvl_rmts_prefill ;;
  3) MODEL=OpenGVLab/InternVL3_5-1B-HF; TASK=idrules; OUT=InternVL3.51B_CLEVRART_test/results_internvl_idrules_prefill ;;
esac

mkdir -p /home/hice1/psunkad3/scratch/clevr_prefill_logs

CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif

apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 \
    --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python /home/hice1/psunkad3/scratch/pgm-vlm-reasoning/zeroshot/clevr_art_infer.py \
    --dataset_root /home/hice1/psunkad3/scratch/clevr_art_data/clevr-art-ood-test-seed1-v1 \
    --task $TASK \
    --output_dir /home/hice1/psunkad3/scratch/$OUT \
    --model_id $MODEL \
    --shard_idx $SHARD \
    --num_shards 8 \
    --max_new_tokens 16 \
    --prefill_answer
