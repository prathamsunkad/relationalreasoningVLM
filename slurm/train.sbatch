#!/bin/bash
#SBATCH -J pgm_pc
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Best compatible GPU: H200. Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, slow V100/RTX6000.
# Shorter queue: sbatch -C "gpu-h200|gpu-h100" ...   Any compatible: -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH -p ice-gpu
#SBATCH -C gpu-h200
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 05:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_pc_runs/logs/train_%x_%j.log

# Per-candidate LoRA fine-tuning on one PGM regime. Resumable: submit it again (or chain it) and it continues from last/.
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=qwen_pc_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
#   sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=ivl1b_pc_neutral_100k,REGIME=neutral,TRAIN_SAMPLES=100000 slurm/train.sbatch
# Chain 3 jobs:   J=$(sbatch --parsable ...);  J=$(sbatch --parsable --dependency=afterany:$J ...); ...
# For the paper's 20% vs 100% comparison use slurm/submit_table5.sh instead.
# Both HF models must already be in the offline cache (huggingface-cli download <id> on a login node).

# EDIT THESE
REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}          # expects $DATA/{train_full,val_full,test_full}/<regime>
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/vlm_lora_pc_runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG, e.g. qwen_neutral_100k}
REGIME=${REGIME:-neutral}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-0}           # 0 = every problem in train_full/<regime> (the 240k 20% set)
EPOCHS=${EPOCHS:-1}
EXTRA=${EXTRA:-}                            # e.g. "--lora_vision --lora_projector --loss both --panel_px 224 --grad_ckpt"
# validate on the 1000 shared validation problems of the 20% dataset when the list exists (same IDs as the teammate runs)
VAL_LIST=${VAL_LIST:-$RUNS/filecache/$REGIME/val1000_shared.tsv}
VAL_ARG=""; [ -f "$VAL_LIST" ] && VAL_ARG="--val_list $VAL_LIST"

mkdir -p $RUNS/logs $RUNS/$TAG $RUNS/filecache
cd $REPO
apptainer exec --nv \
    --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface \
    --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
    "$CONTAINER" \
    python lora/train_lora.py \
    --model_id $MODEL_ID \
    --train_dir $DATA/train_full/$REGIME --val_dir $DATA/val_full/$REGIME \
    --file_cache_dir $RUNS/filecache/$REGIME \
    --out_dir $RUNS/$TAG \
    --train_samples $TRAIN_SAMPLES --epochs $EPOCHS \
    --time_limit_min 285 --num_workers 6 $VAL_ARG \
    $EXTRA
