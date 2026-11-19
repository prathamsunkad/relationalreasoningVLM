#!/bin/bash
#SBATCH -J s0_recon_eval
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -p ice-gpu
#SBATCH -C "gpu-a100|gpu-h100|gpu-h200|gpu-l40s|gpu-a40|gpu-v100|gpu-rtx6000"
#SBATCH --exclude=atl1-1-03-004-21-0,atl1-1-03-013-8-0
#SBATCH --mem-per-gpu=48GB
#SBATCH -t 01:30:00
#SBATCH -o logs/s0_recon_eval_%j.log

# Reconstruction of every finished Stage 0 model on every dataset. Trains nothing; results in $RUNS/$OUT/results.jsonl.
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

REPO=$RELVLM_REPO
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
RUNS=$RELVLM_RUNS
P=$DATA
D=$RELVLM_DATA
OUT=${OUT:-s0_recon_eval}
SOURCES=(
  pgm=matrix:$P/val_full/neutral
  clevr=jsonl:$D/clevr/relational_yesno_val.jsonl
  vsr=jsonl:$D/vsr/random_dev.jsonl
  art_sd=jsonl:$D/art/sd/val.jsonl
  art_rmts=jsonl:$D/art/rmts/val.jsonl
  art_dist3=jsonl:$D/art/dist3/val.jsonl
  art_id=jsonl:$D/art/idrules/val.jsonl
  cart_rmts=jsonl:$D/clevr_art/rmts_val.jsonl
  cart_id=jsonl:$D/clevr_art/idrules_val.jsonl
  coco_seen=jsonl:$D/coco/stage0_val2017.jsonl
  art_id_test=jsonl:$D/art/idrules/test.jsonl
  cart_rmts_ood=jsonl:$D/clevr_art_ood/rmts_cand.jsonl
  vsr_zeroshot=jsonl:$D/vsr/zeroshot_test.jsonl
  whatsup_a=jsonl:$D/whatsup/Controlled_Images_A.jsonl
  cvbench_2d=jsonl:$D/cvbench/test_2d.jsonl
  cvbench_3d=jsonl:$D/cvbench/test_3d.jsonl
  blink=jsonl:$D/blink/val.jsonl
  mmvp=jsonl:$D/mmvp/test.jsonl
)
SRC_ARGS=(); for s in "${SOURCES[@]}"; do SRC_ARGS+=(--source "$s"); done
cd $REPO
run() {
    apptainer exec --nv --env HF_HOME=$HF_HOME --env HF_HUB_OFFLINE=1 \
        --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" \
        python tools/eval_stage0.py --out_dir $RUNS/$OUT --file_cache_dir $RUNS/filecache "${SRC_ARGS[@]}" "$@"
}
run --model_id Qwen/Qwen3-VL-2B-Instruct --ckpt qwen_pgm32=$RUNS/s0_qwen_pgm/best.pt --ckpt qwen_mix32=$RUNS/s0_qwen_mix20/best.pt \
    --ckpt qwen_pgm256=$RUNS/s0_qwen_pgm_d256/best.pt --ckpt qwen_mix256=$RUNS/s0_qwen_mix20_d256/best.pt
run --model_id OpenGVLab/InternVL3_5-1B-HF --ckpt ivl_pgm32=$RUNS/s0_ivl1b_pgm/best.pt --ckpt ivl_mix32=$RUNS/s0_ivl1b_mix20/best.pt
