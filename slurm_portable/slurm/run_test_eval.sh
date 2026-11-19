#!/bin/bash
# Run the 28-shard 200k test evaluation for one adapter, resubmit failed shards (excluding the node they failed on),
# then merge. Runs on the login node; it only submits and waits on Slurm jobs.
#   bash slurm/run_test_eval.sh <MODEL_ID> <TAG> <ADAPTER> <SUFFIX>
#   bash slurm/run_test_eval.sh Qwen/Qwen3-VL-2B-Instruct qwen_pc_recipe_e1 final_e3 _ep3
# Results: $RUNS/results/<TAG>_neutral<SUFFIX>/summary.json
# portable copy (tools/make_portable.py): every path comes from env.sh; logs go to $REPO_ROOT/logs/
source "${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}/env.sh" || { echo "cannot find env.sh: run from the repo root or export REPO_ROOT"; exit 1; }
mkdir -p "$REPO_ROOT/logs"

MODEL_ID=$1; TAG=$2; ADAPTER=$3; SUFFIX=$4
REPO=$REPO_ROOT
RUNS=${RUNS:-$LORA_RUNS}
CONTAINER=${CONTAINER:-/usr/local/pace-apps/manual/packages/ngc-images/pytorch-2.1.0.sif}
BAD=${BAD:-atl1-1-03-018-2-0,atl1-1-03-017-30-0,atl1-1-03-013-3-0}   # known bad H100/H200 nodes
GPUS=${GPUS:-gpu-h200|gpu-h100}
OUT=$RUNS/results/${TAG}_neutral${SUFFIX}
cd $REPO

submit () {  # $1 = array spec; retries while the per-user submit limit refuses the job
    local a
    for try in $(seq 1 120); do
        a=$(env -u EXTRA sbatch --parsable --array=$1 --exclude=$BAD -C "$GPUS" --job-name=eval_${TAG}${SUFFIX} \
            --export=ALL,MODEL_ID=$MODEL_ID,TAG=$TAG,REGIME=neutral,ADAPTER=$ADAPTER,SUFFIX=$SUFFIX \
            $REPO_ROOT/slurm_portable/slurm/eval_array.sbatch 2>/dev/null) && [ -n "$a" ] && { echo $a; return 0; }
        sleep 60
    done
    return 1
}

shards=0-27
for round in 1 2 3; do
    A=$(submit $shards) || { echo "could not submit"; exit 1; }
    echo "[$TAG$SUFFIX] round $round: array $A (shards $shards)"
    while squeue -j $A -h 2>/dev/null | grep -q .; do sleep 60; done
    failed=$(sacct -j $A -X -n -o JobID%24,State,NodeList%30 | awk '$2!="COMPLETED"')
    [ -z "$failed" ] && break
    echo "$failed"
    shards=$(echo "$failed" | awk '{split($1,a,"_"); print a[2]}' | paste -sd,)
    newbad=$(echo "$failed" | awk '{print $3}' | grep -v None | sort -u | paste -sd,)
    [ -n "$newbad" ] && BAD=$BAD,$newbad
    echo "retrying shards $shards, now excluding $BAD"
done

M=$(sbatch --parsable --wait -N1 -n1 --cpus-per-task=2 --mem=16G -t 00:30:00 --job-name=merge_${TAG}${SUFFIX} \
    -o $RUNS/logs/merge_${TAG}${SUFFIX}_%j.log \
    --wrap "cd $REPO && apptainer exec $CONTAINER python lora/eval_vlm.py --merge_dir $OUT --expected_n 200000")
grep -h "problems from\|acc \|histogram\|scored" $RUNS/logs/merge_${TAG}${SUFFIX}_$M.log
