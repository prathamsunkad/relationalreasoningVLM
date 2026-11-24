# Exp1 recipe: every setting the pipeline uses. pipeline.sh sources this file.
# Each line is VAR=${VAR:-default}, so an exported value wins:  SLOT_SOURCE=vit SLOT_LAYER=12 bash pipeline.sh submit
# (unset a variable before changing its default here, or the exported value is still used).

MODEL=${MODEL:-OpenGVLab/InternVL3_5-1B-HF}
SEED=${SEED:-0}
RECIPE=${RECIPE:-lorafirst}       # lorafirst: LoRA baseline -> add the branch (the H11/H12 recipe)
                                  # joint: fresh LoRA trained together with the branch (the original stage 2)

# ---- data (the same puzzles for every arm)
TRAIN_LIMIT=${TRAIN_LIMIT:-0}     # 0 = all of train.jsonl; 60000 = the 5% scale (first 60k rows, nested order)
HOLDOUT=${HOLDOUT:-10000}         # the last 10k rows of train.jsonl: validation for every stage, never trained on
WORKERS=${WORKERS:-12}
SHUFFLE=${SHUFFLE:-1}             # reshuffle every epoch (seeded, resumable); 0 = file order every epoch
USE_CACHE=${USE_CACHE:-1}         # 1 = read panels from PANEL_CACHE (env_local.sh; build_cache.sbatch); 0 = PNG sheets

# ---- relational branch (stage 0): the best settings from the stage-0 search
SLOT_SOURCE=${SLOT_SOURCE:-cnn}   # cnn | vit | resnet.  vit_L12 would be SLOT_SOURCE=vit SLOT_LAYER=12
SLOT_LAYER=${SLOT_LAYER:--1}      # vit only: InternViT layer feeding the slots
ABS_LAYERS=${ABS_LAYERS:-24}      # depth test: 24 > 12 > 6 > 2 at 20k steps, and 24 was still improving
N_SLOTS=${N_SLOTS:-10}
REL_DIM=${REL_DIM:-256}
SLOT_ITERS=${SLOT_ITERS:-3}
QK_NORM=${QK_NORM:-1}             # the collapse fix
RECON=${RECON:-1.0}               # slot reconstruction loss weight (0.1 brought the collapse back)
S0_LR=${S0_LR:-1e-4}              # 3e-4 collapsed; lower did not help
S0_WARMUP=${S0_WARMUP:-500}
S0_BATCH=${S0_BATCH:-16}
S0_STEPS=${S0_STEPS:-300000}      # per half: 300k x 16 = 4.8M puzzles, ~42 epochs of a 115k half (the Slot
                                  # Abstractor needed ~50 epochs to level off). ~7 h at 195 puzzles/s.
S0_DECAY=${S0_DECAY:-30000}       # last 30k steps: LR decays linearly to 0 (or right after a plateau)
S0_EVAL_EVERY=${S0_EVAL_EVERY:-2500}
S0_PATIENCE=${S0_PATIENCE:-12}    # 12 validations (30k steps) without a new best -> decay, then stop
MIN_BRANCH_ACC=${MIN_BRANCH_ACC:-0.25}   # stage 1 does not start if a branch is below this held-out accuracy
                                         # (a branch near chance cannot help; 0 = no check)

# ---- LLM stages (baseline, stage 1, stage 2, control): LoRA r16 / alpha 32 / dropout 0.05 on the LLM
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-4}                 # 16 puzzles per optimizer step, as the repo's LoRA runs
LR_LORA=${LR_LORA:-1e-4}
BASE_EPOCHS=${BASE_EPOCHS:-3}     # LoRA-on-crops baseline (contact-sheet InternVL: 52.3% after 3 epochs)
S1_EPOCHS=${S1_EPOCHS:-0.25}      # alignment: only the fusion head trains (LLM + LoRA frozen)
LR_HEAD=${LR_HEAD:-1e-3}
FT_EPOCHS=${FT_EPOCHS:-1}         # lorafirst: stage 2 and the equal-step control both train this long
LR_REL=${LR_REL:-3e-5}            # stage 2: the branch keeps training slowly (trunk and decoder frozen)
AUX=${AUX:-0.2}                   # stage 2: weight of the branch's own 8-way score loss
BRANCH_DROPOUT=${BRANCH_DROPOUT:-0.5}    # stage 2: relational path off for half the puzzles (H9: stops the shortcut)
LLM_VAL=${LLM_VAL:-2000}          # LLM stages validate on the first 2,000 held-out rows
LLM_EVAL_EVERY=${LLM_EVAL_EVERY:-2000}

# ---- final evaluation
TEST_LIMIT=${TEST_LIMIT:-50000}   # first 50k test puzzles (+-0.44 points); 0 = all 200k

# ---- jobs (PACE: 6 h each, resumed from last.pt). A copy that finds its step finished exits at once.
MAX_HOURS=${MAX_HOURS:-5.4}       # leaves time for loading the model and the last save
JOBS_S0=${JOBS_S0:-3}
JOBS_BASE=${JOBS_BASE:-5}
JOBS_S1=${JOBS_S1:-1}
JOBS_S2=${JOBS_S2:-3}
JOBS_CTRL=${JOBS_CTRL:-3}
JOBS_EVAL=${JOBS_EVAL:-2}

# ---- outputs: one folder per recipe + branch + data size + seed
SLOT_TAG=$SLOT_SOURCE
if [ "$SLOT_SOURCE" = vit ]; then SLOT_TAG=vit_L$SLOT_LAYER; fi
N_TAG=""
if [ "$TRAIN_LIMIT" != 0 ]; then N_TAG=_n$TRAIN_LIMIT; fi
RUN=${RUN:-ckpt/pipeline/${RECIPE}_${SLOT_TAG}_abs${ABS_LAYERS}${N_TAG}_seed$SEED}
