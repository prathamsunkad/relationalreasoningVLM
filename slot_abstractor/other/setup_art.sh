#!/bin/bash
# Run from the slotabstractor-main folder. Adds the ART task generators the code imports
# (same_diff, RMTS, dist3, identity_rules come from the authors' OCRA repo) and makes them
# work with numpy >= 1.24 (np.int was removed). Safe to run twice.
set -e
[ -d art ] || { echo "run this from slotabstractor-main/"; exit 1; }
[ -d ../OCRA ] || git clone --depth 1 https://github.com/Shanka123/OCRA.git ../OCRA
cp ../OCRA/{same_diff,RMTS,dist3,identity_rules}.py art/
sed -i 's/astype(np\.int)/astype(int)/g; s/np\.int)/int)/g' art/{same_diff,RMTS,dist3,identity_rules}.py
python - <<'EOF'
import importlib, sys
sys.path.insert(0, "art")
for m in ["same_diff", "RMTS", "dist3", "identity_rules"]:
    importlib.import_module(m)
import torch, torchvision, einops, colorlog, imageio, scipy, numpy, PIL
print("ART generators OK | torch", torch.__version__, "| torchvision", torchvision.__version__,
      "| numpy", numpy.__version__, "| CUDA available:", torch.cuda.is_available())
EOF
