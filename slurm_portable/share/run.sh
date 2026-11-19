#!/bin/bash
# Entry point of the PGM LoRA adapter bundle. Runs Python inside the shared PACE container with this bundle's own
# packages (env/), base models (models/) and code (code/); your personal ~/.local packages are ignored.
#
#   bash run.sh predict <problem.npz> [qwen|internvl|both] [epoch]   one problem: the 8 candidate scores and the answer
#   bash run.sh eval-shard <qwen|internvl> <epoch> <data_dir> <out_dir> <shard> [num_shards]   (used by eval_test.sbatch)
#   bash run.sh merge <out_dir>                                      combine eval shards -> summary.json + accuracy
#   bash run.sh check                                                (used by check.sbatch)
# GPU commands need a GPU allocation (sbatch the .sbatch files, or an interactive `salloc --gres=gpu:1`).
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

set -e
B="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}

py () {
    apptainer exec --nv --env PYTHONPATH="$B/env:$B/code" --env PYTHONNOUSERSITE=1 --env HF_HOME="$B/hf_cache" \
        --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 "$CONTAINER" python "$@"
}
base () { case $1 in qwen) echo "$B/models/Qwen3-VL-2B-Instruct" ;; internvl) echo "$B/models/InternVL3_5-1B-HF" ;;
          *) echo "unknown model '$1' (use qwen or internvl)" >&2; exit 2 ;; esac; }
adapter () {  # model epoch
    local a; case $1 in qwen) a="$B/adapters/qwen3vl2b_pgm_ep$2" ;; internvl) a="$B/adapters/internvl35_1b_pgm_ep$2" ;; esac
    [ -d "$a" ] || { echo "no adapter for $1 epoch $2 (see adapters/)" >&2; exit 2; }; echo "$a"
}
default_epoch () { case $1 in qwen) echo 5 ;; internvl) echo 6 ;; esac; }
for d in env models; do [ -d "$B/$d" ] || { echo "$B/$d is missing: run 'sbatch setup.sbatch' first" >&2; exit 1; }; done

cmd=${1:?usage: bash run.sh predict|eval-shard|merge|check ...}; shift
case $cmd in
  predict)
    npz=$(readlink -f "${1:?usage: bash run.sh predict <problem.npz> [qwen|internvl|both] [epoch]}"); which=${2:-both}
    specs=()
    for m in $([ "$which" = both ] && echo qwen internvl || echo "$which"); do
        e=${3:-$(default_epoch $m)}; specs+=(--model "$m epoch $e=$(base $m)@$(adapter $m $e)")
    done
    py "$B/code/tools/demo_predict.py" --npz_dir "$(dirname "$npz")" --file "$(basename "$npz")" "${specs[@]}" \
       --out_dir "${OUT_DIR:-$PWD/predict_$(basename "$npz" .npz)}" ;;
  eval-shard)
    m=$1; e=$2; data=$3; out=$4; shard=$5; n=${6:-28}
    py "$B/code/lora/eval_vlm.py" --model_id "$(base $m)" --adapter "$(adapter $m $e)" --npz_dir "$data" \
       --out_dir "$out" --shard_idx "$shard" --num_shards "$n" --batch_size 4 --num_workers 4 ;;
  merge)
    py "$B/code/lora/eval_vlm.py" --merge_dir "${1:?usage: bash run.sh merge <out_dir>}" ;;
  check)
    py "$B/code/check.py" --bundle "$B" ;;
  *) echo "unknown command '$cmd'" >&2; exit 2 ;;
esac
