#!/bin/bash
#SBATCH -J relvlm_smoke
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40"
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 01:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/smoke_%j.log

# Real-weight smoke test on SYNTHETIC panels (does not read pgm_data): stage 0 -> stage 1 (interleave, then fuse) -> eval,
# for both backbones. Prints memory and speed.   sbatch slurm/smoke.sbatch
REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning/relational}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
RUNS=${RUNS:-/home/hice1/psunkad3/scratch/relvlm_runs}
S=$RUNS/smoke; rm -rf $S; mkdir -p $S
cd $REPO
RUN="apptainer exec --nv --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 $CONTAINER python"
set -e
nvidia-smi --query-gpu=name,memory.total --format=csv
$RUN tools/make_fake_npz.py $S/data 32
for M in Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-1B-HF; do
  T=$(basename $M)
  echo "=================== $M"
  $RUN train_stage0.py --model_id $M --out_dir $S/s0_$T --train pgm=matrix:$S/data --val pgm=matrix:$S/data \
      --val_samples 16 --batch_size 16 --max_steps 40 --val_every 20 --log_every 10 --warmup_steps 5 --num_workers 4
  for INJ in interleave fuse; do
    $RUN train_stage1.py --model_id $M --slot_ckpt $S/s0_$T/best.pt --out_dir $S/s1_${T}_$INJ --train pgm=matrix:$S/data \
        --val pgm=matrix:$S/data --val_samples 8 --inject $INJ --rel_tokens 2 --objective both \
        --batch_size 2 --grad_accum 2 --max_steps 6 --log_every 2 --val_every 6 --warmup_steps 2 --num_workers 4
    $RUN eval.py --ckpt $S/s1_${T}_$INJ/best.pt --source pgm=matrix:$S/data --out_json $S/ev_${T}_$INJ/shard0.json \
        --max_examples 8 --num_workers 2
  done
done
echo SMOKE_OK
