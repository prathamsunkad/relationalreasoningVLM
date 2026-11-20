#!/bin/bash
#SBATCH -J iraven-env
#SBATCH -p ice-cpu
#SBATCH -N1 -n1 -c4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH -o /home/hice1/khsieh37/iraven_eval/logs/env-%j.log
# Builds the Python environment with the versions pinned in the adapter bundle's
# requirements-reference.txt and downloads the pinned base model(s) into ~/iraven_eval/hf.
set -euo pipefail
W=$HOME/iraven_eval
export PIP_CACHE_DIR=/tmp/$USER-pip-$SLURM_JOB_ID HF_HOME=$W/hf HF_HUB_DISABLE_TELEMETRY=1
trap 'rm -rf "$PIP_CACHE_DIR"' EXIT
module load python/3.11.9
python --version
[ -d "$W/venv" ] || python -m venv "$W/venv"
source "$W/venv/bin/activate"
pip install -q --upgrade pip
pip install -q torch==2.7.1 torchvision==0.22.1
pip install -q transformers==5.1.0 peft==0.18.1 accelerate==1.12.0 safetensors==0.7.0 Pillow==11.3.0 num2words==0.5.14 numpy
pip list 2>/dev/null | grep -i -E "^(torch|torchvision|transformers|peft|accelerate|safetensors|pillow|num2words|numpy|huggingface.hub|tokenizers) "
python "$W/scripts/download_base.py" ${KEYS:-lfm25-pgm-best}
du -sh "$W/venv" "$W/hf"
