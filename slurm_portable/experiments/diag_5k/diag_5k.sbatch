#!/bin/bash
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 02:30:00
#SBATCH -o logs/%x_%j.log
# Phase 1, step 4 (user: 'okay start it'): score finished runs on 5,000 unseen PGM test problems (test5000.tsv,
# never used for training or checkpoint selection), one eval.py per run with per-problem records, so that each
# hybrid can be compared problem by problem with its own control (McNemar). analyze5k.py writes report.md.
#   lora5        LoRA alone, 5% data, its best (step 14k): vlm_lora_pc adapter in a fusion ckpt with the branch off
#   lf_*         LoRA-first runs (best.pt); *_ctrl are their branch-always-off controls, scored with the branch off;
#                *_off is the hybrid scored with the branch off (what its own LoRA learned)
#   w5_* / w5d_* Stage 2 from the 10-epoch branch, without / with branch dropout 0.5 (best.pt)
#   brset        the settled branch alone (aux head), for the late-fusion reference
#   w5d50_*      Stage 2 with dropout from the 50-epoch branch (step 204k); *_off = scored with the branch off
#   br10/br204k/br276k  the branch alone at 10 epochs and at two points of the 50-epoch run
#   lf276_*      LoRA-first with the step-276k branch (controls: lf_*_ctrl, which do not depend on the branch)
#   c60_*        clean 60k hybrids (slots, branch, alignment all on the 60k only): H8 (w5), H9 (w5d), H12 (lf) recipes;
#                br60 = the 60k branch alone (snapshots/aux_ft60k_best.pt, written by auto/auto_c60.py)
#   c60e_*, br60e  the same with the early 60k branch (aux_ft60k_q_L6_lr3e-4 at step 43k, 28.9% val)
#   rules        rules.json for these problems
# Usage: sbatch -J <diag_5k|diag_5kw> [-t T] diag_5k.sbatch name...   (submitted by ../auto/wait_5k.sh)
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
RUNS=$RELVLM_RUNS
D=$RUNS/diag_5k
DATA=$DATA/test_full/neutral
LIST=$D/test5000.tsv
LP=$LORA_RUNS
FUSE=$RUNS/auto/snap/init_align_q_fuse_L6ft50_lr3e-4_step3000.pt
CONTAINER=/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif
cd $REPO
X() { apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" "$@"; }
ONLY="$*"
[ -n "$ONLY" ] && echo "=== only: $ONLY"
ev() {  # name, then eval.py args
  local name=$1; shift
  [ -n "$ONLY" ] && [[ " $ONLY " != *" $name "* ]] && return
  [ -s $D/$name.json ] && { echo "=== $name: $D/$name.json exists, skipping"; return; }
  echo "=== $name ($(date +%H:%M:%S))"
  X python eval.py --source pgm=matrix:$DATA --val_list $LIST --file_cache_dir $RUNS/filecache \
    --out_json $D/$name.json "$@"
}
ev lora5       --ckpt $FUSE --branch_off --lora_from $LP/qwen_pc_neutral5pct_recipe/best
ev lf_int      --ckpt $RUNS/lora_lf_q_int_ft50/best.pt
ev lf_int_ctrl --ckpt $RUNS/lora_lf_q_int_ft50_ctrl/best.pt --branch_off
ev lf_int_off  --ckpt $RUNS/lora_lf_q_int_ft50/best.pt --branch_off
ev lf_fuse      --ckpt $RUNS/lora_lf_q_fuse_ft50/best.pt
ev lf_fuse_ctrl --ckpt $RUNS/lora_lf_q_fuse_ft50_ctrl/best.pt --branch_off
ev lf_fuse_off  --ckpt $RUNS/lora_lf_q_fuse_ft50/best.pt --branch_off
ev brset       --ckpt $RUNS/aux_ft_q_L6_lr3e-4_settle/best.pt --readout aux
ev w5_int      --ckpt $RUNS/lora_w5_q_int_L6ft/best.pt
ev w5_fuse     --ckpt $RUNS/lora_w5_q_fuse_L6ft/best.pt
ev w5d_int     --ckpt $RUNS/lora_w5d_q_int_L6ft/best.pt
ev w5d_fuse    --ckpt $RUNS/lora_w5d_q_fuse_L6ft/best.pt
ev w5d_int_off  --ckpt $RUNS/lora_w5d_q_int_L6ft/best.pt --branch_off
ev w5d_fuse_off --ckpt $RUNS/lora_w5d_q_fuse_L6ft/best.pt --branch_off
ev w5d50_int      --ckpt $RUNS/lora_w5d_q_int_L6ft50/best.pt
ev w5d50_int_off  --ckpt $RUNS/lora_w5d_q_int_L6ft50/best.pt --branch_off
ev w5d50_fuse     --ckpt $RUNS/lora_w5d_q_fuse_L6ft50/best.pt
ev w5d50_fuse_off --ckpt $RUNS/lora_w5d_q_fuse_L6ft50/best.pt --branch_off
ev lf276_int      --ckpt $RUNS/lora_lf_q_int_ft276k/best.pt
ev lf276_int_off  --ckpt $RUNS/lora_lf_q_int_ft276k/best.pt --branch_off
ev lf276_fuse     --ckpt $RUNS/lora_lf_q_fuse_ft276k/best.pt
ev lf276_fuse_off --ckpt $RUNS/lora_lf_q_fuse_ft276k/best.pt --branch_off
ev br10   --ckpt $RUNS/aux_ft_q_L6_lr3e-4_ep10/best.pt --readout aux
ev br204k --ckpt $RUNS/snapshots/aux_ft_q_L6_lr3e-4_ep50_step204000.pt --readout aux
ev br276k --ckpt $RUNS/snapshots/aux_ft_q_L6_lr3e-4_ep50_step276000.pt --readout aux
ev c60_lf_int       --ckpt $RUNS/lora_lf_q_int_c60/best.pt
ev c60_lf_int_off   --ckpt $RUNS/lora_lf_q_int_c60/best.pt --branch_off
ev c60_lf_fuse      --ckpt $RUNS/lora_lf_q_fuse_c60/best.pt
ev c60_lf_fuse_off  --ckpt $RUNS/lora_lf_q_fuse_c60/best.pt --branch_off
ev c60_w5_int       --ckpt $RUNS/lora_w5_q_int_c60/best.pt
ev c60_w5_int_off   --ckpt $RUNS/lora_w5_q_int_c60/best.pt --branch_off
ev c60_w5_fuse      --ckpt $RUNS/lora_w5_q_fuse_c60/best.pt
ev c60_w5_fuse_off  --ckpt $RUNS/lora_w5_q_fuse_c60/best.pt --branch_off
ev c60_w5d_int      --ckpt $RUNS/lora_w5d_q_int_c60/best.pt
ev c60_w5d_int_off  --ckpt $RUNS/lora_w5d_q_int_c60/best.pt --branch_off
ev c60_w5d_fuse     --ckpt $RUNS/lora_w5d_q_fuse_c60/best.pt
ev c60_w5d_fuse_off --ckpt $RUNS/lora_w5d_q_fuse_c60/best.pt --branch_off
ev br60   --ckpt $RUNS/snapshots/aux_ft60k_best.pt --readout aux
ev c60e_lf_int       --ckpt $RUNS/lora_lf_q_int_c60e/best.pt
ev c60e_lf_int_off   --ckpt $RUNS/lora_lf_q_int_c60e/best.pt --branch_off
ev c60e_lf_fuse      --ckpt $RUNS/lora_lf_q_fuse_c60e/best.pt
ev c60e_lf_fuse_off  --ckpt $RUNS/lora_lf_q_fuse_c60e/best.pt --branch_off
ev c60e_w5_int       --ckpt $RUNS/lora_w5_q_int_c60e/best.pt
ev c60e_w5_int_off   --ckpt $RUNS/lora_w5_q_int_c60e/best.pt --branch_off
ev c60e_w5_fuse      --ckpt $RUNS/lora_w5_q_fuse_c60e/best.pt
ev c60e_w5_fuse_off  --ckpt $RUNS/lora_w5_q_fuse_c60e/best.pt --branch_off
ev c60e_w5d_int      --ckpt $RUNS/lora_w5d_q_int_c60e/best.pt
ev c60e_w5d_int_off  --ckpt $RUNS/lora_w5d_q_int_c60e/best.pt --branch_off
ev c60e_w5d_fuse     --ckpt $RUNS/lora_w5d_q_fuse_c60e/best.pt
ev c60e_w5d_fuse_off --ckpt $RUNS/lora_w5d_q_fuse_c60e/best.pt --branch_off
ev br60e  --ckpt $RUNS/snapshots/aux_ft60k_q_L6_lr3e-4_step43000.pt --readout aux
if [ -z "$ONLY" ] || [[ " $ONLY " == *" rules "* ]]; then
  if [ -s $D/rules.json ]; then echo "=== rules: exists, skipping"
  else echo "=== rules ($(date +%H:%M:%S))"; X python $RUNS/diag_overlap/rules.py $LIST $DATA $D/rules.json; fi
fi
echo "=== done ($(date +%H:%M:%S))"
