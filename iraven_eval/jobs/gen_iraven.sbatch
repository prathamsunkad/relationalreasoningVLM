#!/bin/bash
#SBATCH -J iraven-gen
#SBATCH -p ice-cpu
#SBATCH -N1 -n1 -c4
#SBATCH --mem=16G
#SBATCH -t 16:00:00
#SBATCH -o /home/hice1/khsieh37/iraven_eval/logs/gen-%j.log
# Generates I-RAVEN with the authors' unmodified generator (husheng12345/SRAN, I-RAVEN/),
# default settings (seed 1234, 10,000 per configuration, 6:2:2 split), under Python 2.7 with
# the pinned requirements. Everything is written to node-local /tmp; only the test split
# (2,000 x 7 = 14,000 puzzles) is packed into ~/iraven_eval/data/raw (7 files).
set -euo pipefail
W=$HOME/iraven_eval
T=/tmp/$USER-iraven-gen-$SLURM_JOB_ID
mkdir -p "$T" "$W/data/raw" "$W/generator"
trap 'rm -rf "$T"' EXIT
export CONDA_PKGS_DIRS=$T/conda-pkgs PIP_CACHE_DIR=$T/pip-cache
module load anaconda3
conda create -y -q -p "$T/py27" python=2.7 pip
PY="$T/py27/bin/python"
"$PY" -m pip install -q numpy==1.16.6 tqdm==4.54.1 scipy==1.2.3 opencv-python==4.2.0.32 pillow==6.2.2
"$PY" -c "import sys, numpy, scipy, cv2, PIL; print(sys.version); print(numpy.__version__, scipy.__version__, cv2.__version__, PIL.__version__)"
git clone -q https://github.com/husheng12345/SRAN "$T/SRAN"
git -C "$T/SRAN" rev-parse HEAD | tee "$W/generator/SRAN_COMMIT.txt"
cp -r "$T/SRAN/I-RAVEN" "$W/generator/" 2>/dev/null || true
cd "$T/SRAN/I-RAVEN"
NUM=${NUM_SAMPLES:-10000}
date
"$PY" main.py --save-dir "$T/I-RAVEN" --seed 1234 --num-samples "$NUM"
date
"$PY" "$W/scripts/pack_test.py" "$T/I-RAVEN" "$W/data/raw"
ls -la "$W/data/raw"
date
