#!/bin/bash
#SBATCH -J clevr_art_render
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 06:00:00
#SBATCH --array=0-19
#SBATCH -o logs/clevr_art_%A_%a.log

# CLEVR-ART train renders (CPU, Blender 2.79b). Run `python datasets/make_clevr_art.py setup` once first.
# 80 chunks of 250 problems: chunks 0-39 = RMTS problems [250*c, 250*c+250), chunks 40-79 = identity rules.
# The QOS allows 50 submitted jobs per user, so each of the 20 array tasks renders 4 chunks in parallel (4 threads each).
# Idempotent: finished chunks are skipped, so resubmitting (or `--array=<ids>`) only redoes what is missing.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
export RELVLM_DATA=${RELVLM_DATA:-$RELVLM_DATA}
cd $REPO
# Host python3 (it has numpy), not the container: Blender 2.79b needs libGLU / X11 libraries the container lacks.
pids=()
for k in 0 1 2 3; do
  C=$((SLURM_ARRAY_TASK_ID * 4 + k))
  if [ $C -lt 40 ]; then TASK=rmts; START=$((C*250)); else TASK=idrules; START=$(((C-40)*250)); fi
  python3 datasets/make_clevr_art.py render --task $TASK --start $START --count 250 --threads 4 &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait $p || fail=1; done
exit $fail
