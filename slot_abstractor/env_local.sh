# Paths for slot_abstractor/. Every .sbatch in this folder sources this file; edit the defaults or export your own.
export SHEET_DATA=${SHEET_DATA:-/home/hice1/skomaragiri8/scratch/datasets/pgm-neutral-train20pct-seed20261007-v1/pgm-neutral-train20pct-seed20261007-v1}  # JSONL + contact sheets
export SA_CONDA_ENV=${SA_CONDA_ENV:-slotabs}       # conda env with torch, torchvision, einops, colorlog
export SA_WEIGHTS=${SA_WEIGHTS:-weights}           # checkpoints (not in git); point it at an existing weights/ folder to reuse it
export SA_TAG=${SA_TAG:-pgm20_1gpu}                # run name; the checkpoints are slot_abstractor_pgm_<tag>_d24_lr8e-05_run0_{best,last}.pth.tar
