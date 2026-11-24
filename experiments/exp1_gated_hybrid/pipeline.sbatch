#!/bin/bash
# One step of pipeline.sh as a 6 h job; `bash pipeline.sh submit` submits these with dependencies.
# By hand (from experiments/exp1_gated_hybrid/):  sbatch -J e1_s0A.myrun pipeline.sbatch s0A
# Settings come from pipeline_config.sh plus whatever you exported when submitting (sbatch passes the environment).
#SBATCH -J e1_step                            # Job name (pipeline.sh sets e1_<step>.<run folder name>)
#SBATCH -N1 --ntasks-per-node=1               # Number of nodes and tasks
#SBATCH --cpus-per-task=16                    # CPU cores
#SBATCH --gres=gpu:H200:1                     # GPU type and number of GPUs
#SBATCH --mem-per-gpu=224GB                   # Memory per GPU
#SBATCH -t6:00:00                             # Duration of the job
#SBATCH -o logs/%x.%j.log                     # Output file (job name, job id)
#SBATCH --exclude=atl1-1-03-017-30-0           # its GPU ran the ViT ~4x slower than other nodes
#SBATCH --mail-type=FAIL                      # Mail preferences

cd "$SLURM_SUBMIT_DIR"
source ../../env_local.sh || { echo "submit from experiments/exp1_gated_hybrid (../../env_local.sh not found)"; exit 1; }
module load anaconda3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash pipeline.sh run "${1:?usage: sbatch pipeline.sbatch <step>}"
