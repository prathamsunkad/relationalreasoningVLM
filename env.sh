# Every path the scripts use. `source env.sh` from the repo root (or export your own values first).
# Two kinds of paths:
#   SHARED_ROOT: Pratham's PACE scratch, read-only for teammates (data, checkpoints, adapters; see 00_START_HERE/CHECKPOINTS.md)
#   WORK:        your own scratch, where your runs, outputs and logs go
# For Pratham both resolve to the same folder, so the original runs are used in place.
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}                 # this repo
export REPO_ROOT=${REPO_ROOT:-$REPO}                                         # this repo (used by slurm_portable/)
export SHARED_ROOT=${SHARED_ROOT:-/storage/ice1/4/7/psunkad3}                # shared, read-only (needs the access grant in CHECKPOINTS.md)
export WORK=${WORK:-$HOME/scratch}                                           # yours: outputs and runs
# inputs (shared)
export DATA=${DATA:-$SHARED_ROOT/pgm_data}                                   # PGM {train_full,val_full,test_full}/<regime>/*.npz
export RELVLM_DATA=${RELVLM_DATA:-$SHARED_ROOT/relvlm_data}                  # packed mixture datasets (CLEVR, VSR, ART, CLEVR-ART)
export CLEVR_ART_DATA=${CLEVR_ART_DATA:-$SHARED_ROOT/clevr_art_data}         # CLEVR-ART sheets for the zero-shot runs
export LORA_ADAPTER_5PCT=${LORA_ADAPTER_5PCT:-$SHARED_ROOT/vlm_lora_pc_runs/qwen_pc_neutral5pct_recipe/best}   # start of every LoRA-first run
export VAL_LIST=${VAL_LIST:-$REPO/results/splits/val1000_shared.tsv}       # the 1,000 shared validation problems
# outputs (yours). To read or re-score our runs instead, point these at $SHARED_ROOT/vlm_lora_pc_runs and $SHARED_ROOT/relvlm_runs
export RUNS=${RUNS:-$WORK/vlm_lora_pc_runs}                                  # LoRA runs, results, logs, file lists
export LORA_RUNS=${LORA_RUNS:-$RUNS}                                        # same, under a name the relational scripts do not reuse
export RELVLM_RUNS=${RELVLM_RUNS:-$WORK/relvlm_runs}                         # relational runs: checkpoints, logs, snapshots
# Relational code: its own (non-portable) jobs read REPO=<repo>/relational and RUNS=<relational runs>; pass them per job,
# e.g.  REPO=$RELVLM_REPO RUNS=$RELVLM_RUNS sbatch ... relational/slurm/stage1.sbatch   (slurm_portable/ does this for you)
export RELVLM_REPO=${RELVLM_REPO:-$REPO/relational}
export CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}                          # your model cache (download the two models once)
# GPU nodes that were broken or thermally throttled (Oct 2026); excluded from training/eval jobs
export BAD_NODES=${BAD_NODES:-atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0}
# Pinned base-model revisions (what every adapter was trained on)
export QWEN_MODEL=Qwen/Qwen3-VL-2B-Instruct QWEN_REVISION=89644892e4d85e24eaac8bacfd4f463576704203
export INTERNVL_MODEL=OpenGVLab/InternVL3_5-1B-HF INTERNVL_REVISION=9191dbccf312b537016f041b25d61c72e7c5c9f3
