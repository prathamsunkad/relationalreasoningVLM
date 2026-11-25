# InternVL epoch-6 interleave with incremental frozen features

This experiment starts from shared InternVL3.5-1B PGM epoch-6 LoRA and the
external pretrained Slot Abstractor. It reuses `RelationalVLM.relation_tokens`
and `_lm_hook`: one pooled token, two-layer GELU projector, word-embedding RMS
normalization and learned gain, native `Relational summary: <|rel|>` prompt.
The VLM vision/projector/base language weights and the teacher stay frozen.
Only the connector trains for 500 updates; then it trains jointly with existing
LoRA. This is not full-parameter LLM fine-tuning.

## Data and objective

Same fixed 240,000 official PGM neutral training questions and 1,000 validation
questions. Each question has eight completed 520x520 matrices, native InternVL
crop_to_patches=True (preserves the shared baseline), Yes-minus-No FP32 scores,
and eight-way CE. Global batch16 means16questions,128candidate sequences.
One epoch is15000updates:500connector-only then14500joint. The cached60k
block is shuffled first, then the remaining180k with the same fixed seed.

## Cache and resume

`pgmvlm/incremental_relations.py` reuses the immutable61k cache and extracts only
requested missing questions. The extra180k occupies about1.66GB of FP32 values
in approximately704small shards. Partial shards are checksum-verified and
atomically published; no240k individual files. All later checkpoints use the
same teacher/manifest/seed identity. Cache writes precede training commits.
The trainer saves LoRA, connector, optimizer, schedule position, CUDA/CPU RNG,
validation selection and question cursor together. Interruptions may recompute
uncommitted features but never silently advance the training cursor.

## Recipe

AdamW; connector LR1e-4, LoRA LR2e-5;50updateconnector warmup and cosine decay
fromupdate500 to10%ofpeak byupdate15000;clip1;weight decay.01excludingbias/gain.
Validate every1500updates, selectminimum full-validationCE and reportaccuracy.
Noaugmentation,branchdropout orAbstractorupdates. The teammateNov24branch
supports these learning rates and500stepattachment stage, but its runs began
from an already aligned connector. Our connector starts fresh. Our trueLoRA
freeze,CEselection andextendedone-epoch schedule are intentional differences.

## Entry point and submission

`tools/train_cached_interleave.py --plan PLAN --output OUT --mode pilot --seconds 1500`
uses256trainingquestions(128cached/128missing),32validation,4connector-only
and12jointupdates. It checks native inputs, cache parity, candidate permutation,
trainable/frozen weights and exactdiskcheckpoint next-update replay.
Use `--mode train --seconds 57000` for a16hourallocation; add`--resume` for the
same experiment after a checkpointed time boundary. A passing exact-planpilot
is mandatory. Slurm wrappers and immutable source/plan hashes are archived
under `benchmarks/.local/campaigns/internvl-interleave-20pct-2026-11-24` in the
original VLM workspace. No fusion or full-test evaluation is included.
