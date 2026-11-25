#!/usr/bin/env bash
#SBATCH -J internvl-connectors-pilot
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:h200:1
#SBATCH -p ice-gpu
#SBATCH -A coc
#SBATCH --qos=coc-ice
#SBATCH -t 00:30:00
#SBATCH --no-requeue
#SBATCH -o pilot-%j.out
#SBATCH -e pilot-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
BUNDLE=/storage/ice1/9/8/yluo432/pgm-lora-v1
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
cd "$ROOT"
sha256sum -c source.tar.gz.sha256 --quiet
sha256sum -c plan.json.sha256 --quiet
(cd "$BUNDLE" && sha256sum -c SHA256SUMS --quiet && sha256sum -c runtime-env.tar.sha256 --quiet)
stage=$(mktemp -d "${TMPDIR:-/tmp}/internvl-connectors.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
fstype=$(df -PT "$stage" | tail -1 | awk '{print $2}')
case "$fstype" in lustre|nfs|nfs4) echo 'Temporary storage must be node-local'; exit 1;; esac
free_kib=$(df -Pk "$stage" | tail -1 | awk '{print $4}')
(( free_kib >= 16*1024*1024 )) || { echo 'Need 16 GiB node-local temporary space'; exit 1; }
tar -xf "$BUNDLE/runtime-env.tar" -C "$stage" --no-same-owner
mkdir "$stage/source"
tar -xzf "$ROOT/source.tar.gz" -C "$stage/source" --no-same-owner
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
py() {
    apptainer exec --nv --env PYTHONPATH="$stage/env:$stage/source:$stage/source/tests" \
        --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        "$CONTAINER" python "$@"
}
# Exercise the native tiny-model tests under the exact GPU runtime too.
py -m unittest discover -s "$stage/source/tests" -p test_internvl_connectors.py -v
py "$stage/source/tools/internvl_connector_pilot.py" --plan "$ROOT/plan.json" \
    --output "$ROOT/pilot" --seconds 1500
