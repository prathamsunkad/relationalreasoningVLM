# Paths used by the slurm scripts and tools. `source env.sh` (or export your own values first).
# Everything defaults to the original locations, so existing runs, adapters and data are used in place.
export REPO=${REPO:-/home/hice1/psunkad3/scratch/pgm-vlm-reasoning}
export DATA=${DATA:-/home/hice1/psunkad3/scratch/pgm_data}                 # {train_full,val_full,test_full}/<regime>/*.npz
export RUNS=${RUNS:-/home/hice1/psunkad3/scratch/vlm_lora_pc_runs}         # LoRA runs, results, logs, file lists
export CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
export HF_HOME=${HF_HOME:-/home/hice1/psunkad3/.cache/huggingface}
# GPU nodes that were broken or thermally throttled (Oct 2026); excluded from training/eval jobs
export BAD_NODES=${BAD_NODES:-atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0,atl1-1-03-017-16-0}
# Pinned base-model revisions (what every adapter was trained on)
export QWEN_MODEL=Qwen/Qwen3-VL-2B-Instruct QWEN_REVISION=89644892e4d85e24eaac8bacfd4f463576704203
export INTERNVL_MODEL=OpenGVLab/InternVL3_5-1B-HF INTERNVL_REVISION=9191dbccf312b537016f041b25d61c72e7c5c9f3
