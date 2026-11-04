#!/bin/bash
#SBATCH -J pgm_lora
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 05:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_runs/logs/train_%x_%j.log

# LoRA fine-tuning on one PGM regime. Resumable: submit it again (or chain it) and it continues from last/.
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
#   sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=ivl1b_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
# Chain 3 jobs:   J=$(sbatch --parsable ...);  J=$(sbatch --parsable --dependency=afterany:$J ...); ...
# For the paper's 20% vs 100% comparison use slurm/submit_table5.sh instead.
# Both HF models must already be in the offline cache (huggingface-cli download <id> on a login node).

# EDIT THESE
REPO=${REPO:-/home/hice1/psunkad3/scratch/vlm_lora}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}          # expects $DATA/{train_full,val_full,test_full}/<regime>
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/vlm_lora_runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG, e.g. qwen_neutral_100k}
REGIME=${REGIME:-neutral}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-100000}      # 0 = all 1.2M problems
EPOCHS=${EPOCHS:-1}
EXTRA=${EXTRA:-}                            # e.g. "--lora_vision --lora_projector --loss both --batch_size 4 --grad_accum 8"

mkdir -p $RUNS/logs $RUNS/$TAG $RUNS/filecache
cd $REPO
apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python train_lora.py \
    --model_id $MODEL_ID \
    --train_dir $DATA/train_full/$REGIME --val_dir $DATA/val_full/$REGIME \
    --file_cache_dir $RUNS/filecache/$REGIME \
    --out_dir $RUNS/$TAG \
    --train_samples $TRAIN_SAMPLES --epochs $EPOCHS \
    --time_limit_min 285 --num_workers 6 \
    $EXTRA
