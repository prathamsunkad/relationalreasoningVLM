#!/bin/bash
#SBATCH -J repo_plots
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
#SBATCH -o repo_plots_%j.log
# Redraw results/plots/*.png from the records in results/ (CPU only):  sbatch slurm/plots.sbatch   (from the repo root)
source "$(pwd)/env.sh"
apptainer exec "$CONTAINER" python tools/plot_relational.py
