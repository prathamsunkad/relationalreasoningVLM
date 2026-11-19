#!/bin/bash
#SBATCH -J pgm_pc_viz
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o logs/viz_%j.log

# Render exactly what the model receives for one PGM problem (8 completed matrices + tokenised input), per model.
#   sbatch $REPO_ROOT/slurm_portable/slurm/viz.sbatch                         (problem 0 of val_full/neutral, both models)
#   sbatch --export=ALL,INDEX=5,PANEL_PX=224 $REPO_ROOT/slurm_portable/slurm/viz.sbatch
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$REPO_ROOT
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
DATA=${DATA:-$DATA}
RUNS=${RUNS:-$LORA_RUNS}
REGIME=${REGIME:-neutral}
INDEX=${INDEX:-0}
PANEL_PX=${PANEL_PX:-160}

cd $REPO
for M in Qwen/Qwen3-VL-2B-Instruct OpenGVLab/InternVL3_5-1B-HF; do
    OUT=$RUNS/viz/$(basename $M)_p${INDEX}_px${PANEL_PX}
    apptainer exec \
        --env HF_HOME=$HF_HOME \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        "$CONTAINER" \
        python tools/visualize_input.py --dir $DATA/val_full/$REGIME --index $INDEX --panel_px $PANEL_PX \
        --model_id $M --out_dir $OUT
done
