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

## Runtime dependency repair

Pilot 7300780 stopped before training because the InternVL runtime omitted the external teacher's colorlog dependency. Both job wrappers now load isolated, hash-pinned colorlog 6.9.0 and einops 0.8.1 wheels from `requirements-pretrained-connectors.txt`. `tools/check_pretrained_runtime.py` checks the real teacher imports, source/weight hashes and strict CPU checkpoint loading in the job container before InternVL loading. The GPU pilot still requires fresh/cache feature parity and exact checkpoint replay. No model, image protocol, cache identity or training hyperparameters changed. Repair bundles include the requirement pins, wheels and checksums; the shared runtime is untouched.

## Four-GPU revision (November25)

User priority changed to results within12–24hours. The pending single-H200 job is replaced by a four-H200 correctness pilot and six sequential4hour allocations (at most96training GPU-hours initially; unused segments cancelled). Four synchronous replicas each process4questions; summed gradients preserve globalbatch16 and the existing learning rates. Per-question deterministic dropout permits the real-model serial/distributed gradient and next-update comparison. CPUfour-rank tests cover gradient scaling, exact disk resume and concurrent cache publication. GPUvalidation is mandatory before formal runs.

`tools/train_distributed_interleave.py` trains the epoch6 adapter first, then an independently initialized connector with the epoch4 adapter, each on the exact same240k subset and fixed1kvalidation IDs. Vision/base/teacher remain frozen; existingLoRA trains after500connector-only updates. `tools/dispatch_interleave_allocation.py` uses any remaining allocation time for epoch4 only after epoch6 is terminal, then resumes the unfinished model in the next dependent allocation. Checkpoint, optimizer, per-rank RNG and stopping history are committed together. Cache extraction runs on allfourGPUs and only short disk transactions are locked. Already cached features remain valid across model checkpoints and rank counts.

Validate every500updates (8,000questions). Conservative stopping requires six complete post-warmup checks without cumulative CE improvement of at leastmax(0.001,0.1%) or accuracy gain of0.2percentage points; an improving recent median trend prevents stopping. Earliest possible plateau stop is4,000updates/64,000questions. Best checkpoint remains the minimum full-validation CE. This is an operational plateau criterion, not proof of optimization convergence. Earlystopped runs may leave part of the240k feature cache uncomputed; those features are computed only if latertraining needs them.

Historical evidence: `docs/relvlm_results_H1_H15.md` reports Qwen connector alignment63.3%@1k,63.4%@2k,63.0%@10k, while another joint recipe improves67.3%@2500 to68.9%@3000. Those histories contain accuracy, not validationCE, and use different models/alignment initialization. They motivate caution rather than an assumption that this InternVL run converges quickly. Newexperiment results are validation only; no fulltest job is included.
