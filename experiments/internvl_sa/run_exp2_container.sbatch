#!/bin/bash
# Nov 25 (user: 'yes start phase 1', InternVL + the shared Slot Abstractor): runs Pranav's
# experiments/exp2_sa_interleave/exp2_sa_interleave.py inside the PACE pytorch container (no conda env on this account;
# einops/colorlog/transformers/peft come from ~/.local/lib/python3.10). Every argument is passed to the script.
# SA_WEIGHTS defaults to the shared 20% Slot Abstractor (skomaragiri8 public-models, copied + sha256-checked Nov 25).
#   sbatch -J e2_inspect run_exp2.sbatch inspect --sa_builder ... (see submit lines in the job log)
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64GB
#SBATCH -t 06:00:00
#SBATCH -o /home/hice1/psunkad3/scratch/relvlm_runs/logs/exp2_%x_%j.log
R=/storage/ice1/4/7/psunkad3/relationalreasoningVLM
export SA_WEIGHTS=${SA_WEIGHTS:-/home/hice1/psunkad3/scratch/external_models/slotabstractor-pgm20-2026-11-18-v1/pgm/weights}
source $R/env_local.sh; source $R/env.sh
export EXP2_RUNS=/home/hice1/psunkad3/scratch/relvlm_runs/exp2_runs
cd $R/experiments/exp2_sa_interleave
echo "== $* on $(hostname), $(date); SA_CKPT=$SA_CKPT SA_SLOT_CKPT=$SA_SLOT_CKPT SA_ANTIALIAS=$SA_ANTIALIAS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv --env HF_HOME=/home/hice1/psunkad3/.cache/huggingface --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
  --env SA_CODE_DIR=$SA_CODE_DIR --env SA_CKPT=$SA_CKPT --env SA_SLOT_CKPT=$SA_SLOT_CKPT --env SA_ANTIALIAS=$SA_ANTIALIAS \
  /usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif python exp2_sa_interleave.py "$@"
echo "== done rc=$? $(date)"
