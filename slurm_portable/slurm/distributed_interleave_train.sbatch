#!/usr/bin/env bash
#SBATCH -J ivl4-train
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:4
#SBATCH -p ice-gpu
#SBATCH -A coc
#SBATCH --qos=coc-ice
#SBATCH -t 04:00:00
#SBATCH --no-requeue
#SBATCH -o train-%j.out
#SBATCH -e train-%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?}"
BUNDLE=/storage/ice1/9/8/yluo432/pgm-lora-v1
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
cd "$ROOT"
sha256sum -c source.tar.gz.sha256 --quiet
sha256sum -c campaign.json.sha256 ep6-plan.json.sha256 ep4-plan.json.sha256 --quiet
(cd "$BUNDLE" && sha256sum -c SHA256SUMS --quiet && sha256sum -c runtime-env.tar.sha256 --quiet)
stage=$(mktemp -d "${TMPDIR:-/tmp}/internvl-connectors.XXXXXXXX")
quota_pid=''
cleanup() {
    if [[ -n "$quota_pid" ]]; then
        kill "$quota_pid" 2>/dev/null || true
        wait "$quota_pid" 2>/dev/null || true
    fi
    rm -rf -- "$stage"
}
trap cleanup EXIT
fstype=$(df -PT "$stage" | tail -1 | awk '{print $2}')
case "$fstype" in lustre|nfs|nfs4) echo 'Temporary storage must be node-local'; exit 1;; esac
free_kib=$(df -Pk "$stage" | tail -1 | awk '{print $4}')
(( free_kib >= 16*1024*1024 )) || { echo 'Need 16 GiB node-local temporary space'; exit 1; }
tar -xf "$BUNDLE/runtime-env.tar" -C "$stage" --no-same-owner
mkdir "$stage/source"
tar -xzf "$ROOT/source.tar.gz" -C "$stage/source" --no-same-owner
# Pure-Python wheels, pinned by the same hashes used for the validated teacher cache.
(cd "$ROOT/wheels" && sha256sum -c SHA256SUMS --quiet)
python3 - "$ROOT/wheels" "$stage/reference-deps" <<'PYDEPS'
import sys, zipfile
from pathlib import Path
for wheel in sorted(Path(sys.argv[1]).glob('*.whl')):
    with zipfile.ZipFile(wheel) as archive:
        for name in archive.namelist():
            if name.startswith('/') or '..' in Path(name).parts:
                raise RuntimeError('Unsafe wheel path')
        archive.extractall(sys.argv[2])
PYDEPS
CAMPAIGN=$(python3 -c 'import json,sys;from pathlib import Path;print(Path(json.load(open(sys.argv[1]))["incremental_cache"]).parent)' "$ROOT/ep6-plan.json")
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$ROOT/pilot" "$CAMPAIGN/epoch6" "$CAMPAIGN/cache"
# lfs is a host utility, absent from this CUDA container. Query it live on the
# host for every storage check; only this job's output tree is accessible.
python3 "$stage/source/tools/storage_quota_bridge.py" --root "$CAMPAIGN" \
    --socket "$stage/quota.sock" --ready "$stage/quota-ready.json" >"$ROOT/quota-bridge.log" 2>&1 &
quota_pid=$!
for ((attempt=0;attempt<300;attempt++)); do
    [[ -f "$stage/quota-ready.json" ]] && break
    kill -0 "$quota_pid" 2>/dev/null || { cat "$ROOT/quota-bridge.log"; exit 1; }
    sleep 0.2
done
[[ -f "$stage/quota-ready.json" ]] || { echo 'Host quota service did not become ready'; exit 1; }
py() {
    apptainer exec --nv --bind "$stage:$stage" --env PYTHONPATH="$stage/reference-deps:$stage/env:$stage/source:$stage/source/tests" \
        --env PGMVLM_STORAGE_SOCKET="$stage/quota.sock" \
        --env PYTHONNOUSERSITE=1 --env PYTHONDONTWRITEBYTECODE=1 \
        --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 \
        "$CONTAINER" python "$@"
}
py -c 'import json,sys; from pgmvlm.storage import require_headroom; print(json.dumps(require_headroom(sys.argv[1],reserve_files=2000)))' "$ROOT/pilot"
py "$stage/source/tools/check_pretrained_runtime.py" --plan "$ROOT/ep6-plan.json" \
    --output "$ROOT/runtime-preflight-$SLURM_JOB_ID.json"
py "$stage/source/tools/dispatch_interleave_allocation.py" --campaign "$ROOT/campaign.json" \
    --source "$stage/source" --mode train --seconds 13800
# Release only unused pending segments from this exact campaign when both models finish.
python3 - "$ROOT" <<'PYFINISH'
import json,os,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);summary=root/'campaign-summary.json'
if summary.exists() and json.loads(summary.read_text()).get('mode')=='train' and json.loads(summary.read_text()).get('complete'):
    jobs=json.loads((root/'jobs.json').read_text())
    for job in jobs.get('segments',[]):
        if job==os.environ.get('SLURM_JOB_ID'): continue
        row=subprocess.check_output(['squeue','-h','-j',job,'-o','%T|%j'],text=True).strip()
        if row.startswith('PENDING|ivl4-train'):
            subprocess.run(['scancel',job],check=True)
PYFINISH
