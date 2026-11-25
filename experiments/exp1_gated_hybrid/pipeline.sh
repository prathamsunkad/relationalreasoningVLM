#!/bin/bash
# Exp1, the whole recipe: relational branch -> LoRA baseline -> alignment -> hybrid fine-tuning + equal-step
# control -> test evaluation with controls. Settings: pipeline_config.sh. Run from experiments/exp1_gated_hybrid/.
#
#   bash pipeline.sh smoke          # every step for a few steps on an interactive GPU (~15 min): checks the plumbing
#   bash pipeline.sh submit         # submit every unfinished step as chained 6 h jobs (safe to repeat)
#   bash pipeline.sh status         # progress, ETA and best validation of each step
#   bash pipeline.sh report         # the final table (written by the eval step to $RUN/report_test.md)
#   bash pipeline.sh run <step>     # one step in the foreground (interactive GPU, or what each job runs)
#   bash pipeline.sh print          # the settings and the output folder
#
# Steps (RECIPE=lorafirst, the default):
#   s0A, s0B  stage 0: the branch alone on half A / half B of the pool (cross-fitting)     \
#   base      LoRA on the 16 crops, no branch (BASE_EPOCHS)                               / these 3 run in parallel
#   s1        stage 1: both branches + the base LoRA, frozen; only the fusion head trains (S1_EPOCHS)
#   s2        stage 2: LoRA + head + branch (low LR) + aux loss + branch dropout (FT_EPOCHS)
#   ctrl      equal-step control: the base LoRA trained FT_EPOCHS more, no branch
#   eval      test: base, ctrl, s1, s2, s2 with the branch off, s2 with shuffled relational features; report
# RECIPE=joint: s1 without LoRA, s2 = fresh LoRA + branch for BASE_EPOCHS, and base itself is the control.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$HERE"
source ../../env_local.sh
CMD=${1:-print}
if [ "$CMD" = smoke ]; then   # tiny budgets, own folder; everything else as in the real run
  export RUN=${RUN:-ckpt/pipeline_smoke} S0_STEPS=40 S0_DECAY=10 S0_EVAL_EVERY=20 S0_VAL=64 LLM_VAL=16 \
         LLM_EVAL_EVERY=10 SMOKE_STEPS=12 TEST_LIMIT=24 MIN_BRANCH_ACC=0 MAX_HOURS=0
fi
source ./pipeline_config.sh
S0_VAL=${S0_VAL:-0}            # stage-0 validation rows (0 = all HOLDOUT rows)
SMOKE_STEPS=${SMOKE_STEPS:-0}
JOBTAG=$(basename "$RUN")      # job names e1_<step>.<JOBTAG>: one run's jobs can be told apart from another's

CACHE_FLAG=()
if [ "$USE_CACHE" = 1 ]; then CACHE_FLAG=(--cache "$PANEL_CACHE"); fi
COMMON=(--data "$SHEET_DATA" --model "$MODEL" --seed "$SEED" --layout "$LAYOUT" --workers "$WORKERS")
TRAIN=(--val_from_train "$HOLDOUT" --train_limit "$TRAIN_LIMIT" --select loss --max_hours "$MAX_HOURS"
       "${CACHE_FLAG[@]}")
if [ "$SHUFFLE" = 1 ]; then TRAIN+=(--shuffle); fi
LLM=(--batch_size "$BATCH" --grad_accum "$ACCUM" --val_limit "$LLM_VAL" --eval_every "$LLM_EVAL_EVERY"
     --eval_batch_size 8 --save_every 500 --schedule cosine)

say() { echo "[$(date '+%m-%d %H:%M')] $*"; }
finished() {   # finished <step>: 0 if the step has nothing left to do
  case $1 in
    eval) [ -f "$RUN/report_test.md" ] ;;
    ctrl) [ "$RECIPE" = joint ] || [ -f "$RUN/ctrl/summary.json" ] ;;
    *)    [ -f "$RUN/$1/summary.json" ] ;;
  esac
}
need() {       # need <step>...: stop unless those steps are finished
  for s in "$@"; do
    finished "$s" || { say "step '$s' is not finished yet ($RUN/$s): run or submit it first"; exit 1; }
  done
}
length() {     # length <epochs>: the training length of an LLM stage (a few steps in the smoke test)
  if [ "$SMOKE_STEPS" -gt 0 ]; then echo "--max_steps $SMOKE_STEPS"; else echo "--epochs $1"; fi
}
train() {      # train <step> <mode> [flags]
  local name=$1 mode=$2; shift 2
  if finished "$name"; then say "$name: finished, skipping"; return; fi
  say "$name ($mode) -> $RUN/$name on $(hostname)"
  python -u train_hybrid.py "$mode" "${COMMON[@]}" "${TRAIN[@]}" --out "$RUN/$name" "$@"
  if finished "$name"; then say "$name: finished"
  else say "$name: not finished (time budget); the next job continues from $RUN/$name/last.pt"; fi
}
evaluate() {   # evaluate <step> <tag> [control flag]
  local name=$1 tag=$2; shift 2
  local out="$RUN/$name/eval_test${tag}_summary.json"
  if [ -f "$out" ]; then say "eval $name$tag: done"; return; fi
  say "eval $name$tag on the first $TEST_LIMIT test puzzles"
  python -u train_hybrid.py eval "${COMMON[@]}" "${CACHE_FLAG[@]}" --init "$RUN/$name" --split test \
      --limit "$TEST_LIMIT" --eval_batch_size 8 "$@"
}
check_branches() {   # stage 1 only starts from branches that learned something
  python - "$RUN" "$MIN_BRANCH_ACC" <<'EOF'
import json, sys
run, floor = sys.argv[1], float(sys.argv[2])
bad = []
for h in "AB":
    b = json.load(open(f"{run}/s0{h}/summary.json"))["best"]
    print(f"branch {h}: held-out loss {b['loss']:.4f}, accuracy {b['acc']:.2%} (step {b['step']}; chance 12.5%)")
    if b["acc"] < floor:
        bad.append(h)
if bad:
    sys.exit(f"branch(es) {', '.join(bad)} below MIN_BRANCH_ACC={floor}: the hybrid would get little from them. Train "
             f"stage 0 longer (raise S0_STEPS and resubmit) or set MIN_BRANCH_ACC=0 to go on anyway.")
EOF
}

run_step() {
  local step=$1
  for f in train_hybrid.py relational_fusion.py "$PROBES/pgm_gap.py" "$LAYOUT"; do
    [ -f "$f" ] || { echo "$f not found"; exit 1; }
  done
  [ -f "$SHEET_DATA/train.jsonl" ] || { echo "no train.jsonl in SHEET_DATA=$SHEET_DATA (env_local.sh)"; exit 1; }
  if [ "$USE_CACHE" = 1 ] && [ ! -f "$PANEL_CACHE/train_meta.json" ]; then
    echo "PANEL_CACHE=$PANEL_CACHE has no train_meta.json: sbatch build_cache.sbatch first, or USE_CACHE=0 (slower)"
    exit 1
  fi
  mkdir -p "$RUN"
  local S0=(--slot_source "$SLOT_SOURCE" --slot_layer "$SLOT_LAYER" --abs_layers "$ABS_LAYERS" --n_slots "$N_SLOTS"
            --rel_dim "$REL_DIM" --slot_iters "$SLOT_ITERS" --qk_norm "$QK_NORM" --recon_weight "$RECON"
            --lr_rel "$S0_LR" --batch_size "$S0_BATCH" --grad_accum 1 --schedule constant
            --warmup_steps "$S0_WARMUP" --max_steps "$S0_STEPS" --decay_steps "$S0_DECAY"
            --patience "$S0_PATIENCE" --eval_every "$S0_EVAL_EVERY" --val_limit "$S0_VAL" --eval_batch_size 32
            --save_every 1000)
  case $step in
    s0A) train s0A stage0 --half A "${S0[@]}" ;;
    s0B) train s0B stage0 --half B "${S0[@]}" ;;
    base) train base baseline "${LLM[@]}" $(length "$BASE_EPOCHS") --lr_lora "$LR_LORA" ;;
    s1)
      need s0A s0B
      finished s1 || check_branches
      if [ "$RECIPE" = lorafirst ]; then
        need base
        train s1 stage1 --branches "$RUN/s0A" "$RUN/s0B" --lora_from "$RUN/base" "${LLM[@]}" \
            $(length "$S1_EPOCHS") --lr_head "$LR_HEAD"
      else
        train s1 stage1 --branches "$RUN/s0A" "$RUN/s0B" "${LLM[@]}" $(length "$S1_EPOCHS") --lr_head "$LR_HEAD"
      fi ;;
    s2)
      need s1
      local E=$FT_EPOCHS; [ "$RECIPE" = joint ] && E=$BASE_EPOCHS
      train s2 stage2 --init "$RUN/s1" "${LLM[@]}" $(length "$E") --lr_lora "$LR_LORA" --lr_head "$LR_HEAD" \
          --lr_rel "$LR_REL" --aux_weight "$AUX" --branch_dropout "$BRANCH_DROPOUT" ;;
    ctrl)
      if [ "$RECIPE" = joint ]; then say "ctrl: RECIPE=joint uses base as the control"; return; fi
      need base
      train ctrl baseline --init "$RUN/base" "${LLM[@]}" $(length "$FT_EPOCHS") --lr_lora "$LR_LORA" ;;
    eval)
      need base s1 s2 ctrl
      [ -f "$SHEET_DATA/test.jsonl" ] || { echo "no test.jsonl in $SHEET_DATA"; exit 1; }
      evaluate base ""
      [ "$RECIPE" = joint ] || evaluate ctrl ""
      evaluate s1 ""
      evaluate s2 ""
      evaluate s2 _nofusion --no_fusion
      evaluate s2 _shuffled --shuffle_relational
      python pipeline_report.py compare "$RUN" --recipe "$RECIPE" ;;
    *) echo "unknown step '$step' (s0A s0B base s1 s2 ctrl eval)"; exit 1 ;;
  esac
}

submit_chain() {   # submit_chain <step> <copies> [job ids to wait for]: prints the last job id ("" if finished)
  local step=$1 n=$2; shift 2
  if finished "$step"; then echo ""; return; fi
  local dep; dep=$(IFS=:; echo "$*")
  local jid=""
  for _ in $(seq "$n"); do
    jid=$(sbatch --parsable -J "e1_$step.$JOBTAG" ${dep:+--dependency=afterany:$dep} pipeline.sbatch "$step")
    dep=$jid
  done
  echo "$jid"
}

case $CMD in
  print)
    echo "RUN=$RUN  RECIPE=$RECIPE  branch: $SLOT_TAG, $ABS_LAYERS layers, $S0_STEPS steps per half"
    echo "data: $SHEET_DATA (train_limit $TRAIN_LIMIT, holdout $HOLDOUT)  cache: $([ "$USE_CACHE" = 1 ] && echo "$PANEL_CACHE" || echo off)"
    echo "LLM: batch $BATCH x $ACCUM, lr $LR_LORA, base $BASE_EPOCHS ep, s1 $S1_EPOCHS ep, s2/ctrl $FT_EPOCHS ep, branch dropout $BRANCH_DROPOUT"
    for s in s0A s0B base s1 s2 ctrl eval; do printf "  %-5s %s\n" "$s" "$(finished "$s" && echo finished || echo todo)"; done ;;
  run)    run_step "${2:?usage: bash pipeline.sh run <step>}" ;;
  smoke)
    say "smoke test -> $RUN (delete it afterwards: rm -r $RUN)"
    for s in s0A s0B base s1 s2 ctrl eval; do run_step "$s"; done
    say "smoke test passed: every step ran. Now: bash pipeline.sh submit" ;;
  submit)
    mkdir -p logs
    command -v sbatch >/dev/null || { echo "sbatch not found: submit from a PACE login node"; exit 1; }
    queued=$(squeue -h -u "${USER:-$(id -un)}" -o %j 2>/dev/null | grep -c "\.$JOBTAG\$" || true)
    if [ "$queued" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
      echo "$queued job(s) of this run ($JOBTAG) are already queued or running; not submitting twice."
      echo "Wait for them, or cancel them first:  squeue -u \$USER -o '%i %j' | grep '\.$JOBTAG\$' | cut -d' ' -f1 | xargs scancel"
      exit 1
    fi
    a=$(submit_chain s0A "$JOBS_S0"); b=$(submit_chain s0B "$JOBS_S0"); base=$(submit_chain base "$JOBS_BASE")
    if [ "$RECIPE" = lorafirst ]; then s1=$(submit_chain s1 "$JOBS_S1" $a $b $base); else s1=$(submit_chain s1 "$JOBS_S1" $a $b); fi
    s2=$(submit_chain s2 "$JOBS_S2" $s1)
    ctrl=""; [ "$RECIPE" = joint ] || ctrl=$(submit_chain ctrl "$JOBS_CTRL" $base)
    ev=$(submit_chain eval "$JOBS_EVAL" $base $s1 $s2 $ctrl)
    echo "RUN=$RUN"
    for s in s0A:$a s0B:$b base:$base s1:$s1 s2:$s2 ctrl:$ctrl eval:$ev; do
      printf "  %-5s %s\n" "${s%%:*}" "${s#*:}"; done
    echo "(last job of each chain; empty = already finished). Logs: logs/e1_<step>.$JOBTAG.<jobid>.log"
    echo "Check with: bash pipeline.sh status ; squeue -u \$USER" ;;
  status) python pipeline_report.py status "$RUN" --recipe "$RECIPE" ;;
  report) cat "$RUN/report_test.md" ;;
  *) sed -n 2,22p "$0"; exit 1 ;;
esac
