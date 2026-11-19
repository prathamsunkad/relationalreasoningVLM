#!/bin/bash
#SBATCH -J pgm_bundle_setup
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o setup_%j.log
# One-time setup of the PGM LoRA adapter bundle (CPU only, ~20-40 min, needs ~12 GB in this folder):
#   1. installs the exact Python packages (requirements-lock.txt) into env/   (nothing outside this folder is touched)
#   2. downloads the two base models at the exact revisions into models/
#   3. checks that everything imports
# Run it from inside the bundle folder:   cd pgm-lora-v1 && sbatch setup.sbatch      (log: setup_<jobid>.log)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

set -euo pipefail
B=${SLURM_SUBMIT_DIR:-$(pwd)}
cd "$B"
[ -f requirements-lock.txt ] || { echo "run this from inside the bundle folder"; exit 1; }
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
X="apptainer exec --env PYTHONNOUSERSITE=1 --env HF_HOME=$B/hf_cache $CONTAINER"

echo "[1/3] installing packages into $B/env"
$X python -m pip install --no-deps --no-cache-dir --disable-pip-version-check --target "$B/env" \
    -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu124

echo "[2/3] downloading base models into $B/models"
$X env PYTHONPATH="$B/env" python - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev, name in [("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "Qwen3-VL-2B-Instruct"),
                        ("OpenGVLab/InternVL3_5-1B-HF", "9191dbccf312b537016f041b25d61c72e7c5c9f3", "InternVL3_5-1B-HF")]:
    path = snapshot_download(repo, revision=rev, local_dir=f"models/{name}")
    print(f"  {repo} @ {rev[:12]} -> {path}")
PY

echo "[3/3] checking imports"
$X env PYTHONPATH="$B/env:$B/code" python -c "
import torch, transformers, peft, numpy, PIL
import pgmvlm.models, pgmvlm.data
print('  torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| numpy', numpy.__version__)
"
echo "setup done. Next: sbatch check.sbatch"
