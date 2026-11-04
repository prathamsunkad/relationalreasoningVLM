#!/bin/bash
#SBATCH -J pgm_overfit
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# Avoid Blackwell (ice-bw-gpu: unsupported by torch 2.6/cu124), AMD MI210, and slow V100/RTX6000
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 03:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/vlm_lora_runs/logs/overfit_%x_%j.log

# Overfit test: train on a tiny fixed set of N problems for many epochs and watch accuracy ON THOSE SAME PROBLEMS
# ([fit] lines in the log, train_fit_acc_logits in metrics.jsonl). Held-out [val] accuracy is printed too.
#   [fit] climbs towards 100%  -> the model can tell the inputs apart; the full run's problem is generalisation/formulation
#   [fit] stays near 12.5%     -> it cannot even memorise: perception/resolution bottleneck (or a pipeline bug)
#   sbatch --export=ALL,MODEL_ID=Qwen/Qwen3-VL-2B-Instruct,TAG=overfit_qwen slurm/overfit.sbatch
#   sbatch --export=ALL,MODEL_ID=OpenGVLab/InternVL3_5-1B-HF,TAG=overfit_ivl1b slurm/overfit.sbatch
# Defaults: 500 problems x 40 epochs = 625 steps at global batch 32, LoRA on LLM + vision tower + projector, digit8 loss.
# For comparison with the original setup (LLM-only LoRA): EXTRA="" TAG=overfit_qwen_llmonly

REPO=${REPO:-/home/hice1/psunkad3/scratch/vlm_lora}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/vlm_lora_runs}
MODEL_ID=${MODEL_ID:-Qwen/Qwen3-VL-2B-Instruct}
TAG=${TAG:?set TAG, e.g. overfit_qwen}
REGIME=${REGIME:-neutral}
N=${N:-500}                                  # training problems (also the number checked for fit)
EPOCHS=${EPOCHS:-40}
EXTRA=${EXTRA---lora_vision --lora_projector}   # set EXTRA="" for LLM-only LoRA

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
    --train_samples $N --epochs $EPOCHS --fit_eval_samples $N \
    --loss digit8 --batch_size 4 --grad_accum 8 --warmup_steps 20 \
    --val_every 50 --val_samples 500 --save_every 100000 --log_every 10 \
    --time_limit_min 170 --num_workers 6 \
    $EXTRA
