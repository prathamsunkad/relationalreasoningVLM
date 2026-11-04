#!/bin/bash
#SBATCH -J pgm_pc_overfit
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 03:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_pc_runs/logs/overfit_%x_%j.log

# Per-candidate overfit test: train on a tiny fixed set of N problems for many epochs and watch accuracy ON THOSE SAME
# PROBLEMS ([fit] lines in the log, train_fit_acc in metrics.jsonl). Held-out [val] accuracy is printed too.
#   [fit] climbs well above 12.5%  -> per-candidate scoring gives the model a learning signal; go to a real run
#   [fit] stays near 12.5%         -> it still cannot separate the 8 completed matrices: resolution (--panel_px) next
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=overfit_pc_qwen slurm/overfit.sbatch
#   sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=overfit_pc_ivl1b slurm/overfit.sbatch
# Defaults: 500 problems x 20 epochs = 313 steps at global batch 32 problems (= 256 sequences), LLM-only LoRA, ce8 loss.
# Variants: EXTRA="--lora_vision --lora_projector"  |  EXTRA="--panel_px 224"

REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/vlm_lora_pc_runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG, e.g. overfit_pc_qwen}
REGIME=${REGIME:-neutral}
N=${N:-500}                                  # training problems (also the number checked for fit)
EPOCHS=${EPOCHS:-20}
EXTRA=${EXTRA:-}

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
    --train_samples $N --epochs $EPOCHS --fit_eval_samples $N \
    --loss ce8 --batch_size 2 --grad_accum 16 --warmup_steps 20 \
    --val_every 25 --val_samples 500 --val_batch_size 4 --save_every 100000 --log_every 5 \
    --time_limit_min 170 --num_workers 6 \
    $EXTRA
