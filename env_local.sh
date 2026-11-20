# Paths for probes/, experiments/exp1_gated_hybrid/ and experiments/exp2_sa_interleave/.
# Their .sbatch and run_*.sh files source this file. Edit the defaults, or export your own values first.
# (slot_abstractor/ has its own env_local.sh.)
#
# Note: the repo's env.sh also defines DATA, but there it is the folder of PGM .npz files. The probes and Exp1
# read the contact-sheet package (JSONL + PNG sheets) instead, so it has its own name: SHEET_DATA.
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SHEET_DATA=${SHEET_DATA:-/storage/ice1/7/9/skomaragiri8/pgm-neutral-train20pct-seed20261007-v1}  # train/validation/test .jsonl + sheets
export CONDA_ENV=${CONDA_ENV:-$HOME/scratch/envs/vlmprobe}       # conda env with torch, transformers, peft, einops, colorlog
export PROBES=${PROBES:-$_REPO/probes}                           # pgm_gap.py (shared VLM wrapper) lives here
export LAYOUT=${LAYOUT:-$PROBES/layout.json}                     # panel boxes on a contact sheet (one file for everything)
export PANEL_CACHE=${PANEL_CACHE:-pgm_panel_cache}               # Exp1 panel cache (relative = inside experiments/exp1_gated_hybrid/)
# export HF_HOME=$HOME/scratch/hf_cache                          # uncomment if your Hugging Face models are cached elsewhere

# ---- Exp2: pretrained Slot Abstractor + the repo's fine-tuned InternVL. The .npz data, VAL_LIST and
# SHARED_ROOT come from the repo's env.sh, which exp2.sbatch also sources.
export SA_CODE_DIR=${SA_CODE_DIR:-$_REPO/slot_abstractor}                  # the Slot Abstractor code (in this repo)
export SA_WEIGHTS=${SA_WEIGHTS:-$_REPO/slot_abstractor/weights}            # its weights/ folder (not in git; set it to yours)
export SA_CKPT=${SA_CKPT:-$SA_WEIGHTS/slot_abstractor_pgm_pgm20_1gpu_d24_lr8e-05_run0_best.pth.tar}   # trained on the 240k only
export SA_SLOT_CKPT=${SA_SLOT_CKPT:-$SA_WEIGHTS/slot_ae_pgm20.pth.tar}     # its slot autoencoder (pretrained on the 240k only)
export SA_BUILDER=${SA_BUILDER:-$_REPO/experiments/exp2_sa_interleave/sa_builder.py}   # connects Exp2 to it
export SA_ANTIALIAS=${SA_ANTIALIAS:-1}                                     # 1 if the slotabs env has torchvision >= 0.17, else 0
export IVL_ADAPTER=${IVL_ADAPTER:-/storage/ice1/4/7/psunkad3/vlm_lora_pc_runs/ivl1b_pc_recipe_e1/final_e6}   # 60.31% test
export EXP2_RUNS=${EXP2_RUNS:-exp2_runs}                                    # outputs (relative = inside experiments/exp2_sa_interleave/)
