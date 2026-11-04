#!/bin/bash
# Paper Table 5 analogue: train on 20% and on 100% of one regime's training data, then evaluate both on the full test split.
# Submits, per fraction: PARTS chained training jobs (each resumes and stops itself before the 5h limit), then the 28-shard
# eval array (ADAPTER=final) that starts when the last training job ends.
#
#   MODEL_ID=Qwen/Qwen3-VL-2B-Instruct  MODEL_TAG=qwen  REGIME=neutral  PARTS=6  bash slurm/submit_table5.sh
#   MODEL_ID=OpenGVLab/InternVL3_5-1B-HF MODEL_TAG=ivl1b REGIME=neutral PARTS=6  bash slurm/submit_table5.sh
# EPOCHS_20 / EPOCHS_100 set the passes over each subset (default 1 and 1). The paper does not state the epochs of its 20% run;
# Extra train flags for both runs: USER_EXTRA="--lora_vision --lora_projector" bash slurm/submit_table5.sh
# to separate "less data" from "fewer steps" also run EPOCHS_20=5 (same number of steps as 100% for 1 epoch).
# Check samples/s from a first run (tools/preflight.py or the log) to choose PARTS: hours = problems*epochs / (samples_per_s*3600).
set -e
MODEL_ID=${MODEL_ID:?set MODEL_ID}
MODEL_TAG=${MODEL_TAG:?set MODEL_TAG}
REGIME=${REGIME:-neutral}
PARTS=${PARTS:-4}
EPOCHS_20=${EPOCHS_20:-1}
EPOCHS_100=${EPOCHS_100:-1}
HERE=$(cd "$(dirname "$0")" && pwd)

for FRAC in 0.2 1.0; do
    if [ "$FRAC" = "0.2" ]; then PCT=20; EP=$EPOCHS_20; else PCT=100; EP=$EPOCHS_100; fi
    TAG=${MODEL_TAG}_${REGIME}_f${PCT}_e${EP}
    PREV=""
    for p in $(seq 1 $PARTS); do
        DEP=""; [ -n "$PREV" ] && DEP="--dependency=afterany:$PREV"
        # settings travel as environment variables (--export=ALL), which is safe for values containing spaces
        PREV=$(MODEL_ID=$MODEL_ID TAG=$TAG REGIME=$REGIME TRAIN_SAMPLES=0 EPOCHS=$EP EXTRA="--train_fraction $FRAC ${USER_EXTRA:-}" \
            sbatch --parsable $DEP --job-name=${TAG} --export=ALL $HERE/train.sbatch)
        echo "train $TAG part $p/$PARTS -> job $PREV"
    done
    EV=$(MODEL_ID=$MODEL_ID TAG=$TAG REGIME=$REGIME ADAPTER=final EXTRA="" \
        sbatch --parsable --dependency=afterany:$PREV --job-name=eval_${TAG} --export=ALL $HERE/eval_array.sbatch)
    echo "eval  $TAG -> array job $EV (after $PREV)"
done
echo "afterwards: python lora/eval_vlm.py --merge_dir \$RUNS/results/<TAG>_$REGIME --expected_n 200000 ; python tools/make_table.py --table5 --row \"$MODEL_TAG:20=...,100=...\""
