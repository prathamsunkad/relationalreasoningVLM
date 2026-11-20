# LFM, SmolVLM and SpatialEvo integration

This integration extends the existing `pgmvlm` package and `lora/train_lora.py` / `lora/eval_vlm.py` commands. It does not add another LoRA trainer. The original Qwen/InternVL candidate-scoring behavior remains the default. The teammates' separate `relational/` slot-based implementation is unchanged.

## Choose the protocol explicitly

| Setting | Existing `candidate` | Added `answer_letter` |
|---|---|---|
| Input | Eight completed matrices per PGM question | One contact sheet with the incomplete matrix and A–H options; CLEVR uses complete candidate scenes |
| Supervision | Eight-way CE over Yes-minus-No candidate scores; optional BCE | Full-vocabulary causal CE on the answer and chat ending (`suffix`), or the single answer token (`answer`) |
| Evaluation | Argmax of eight candidate scores | Greedy text generation, strict answer parsing; malformed answers count as wrong |
| Applicable tasks | Existing PGM pathway | PGM, CLEVR-ART RMTS (A/B), CLEVR-ART identity (A–D) |

A suffix loss is averaged over answer **and ending** tokens. It is not numerically comparable to answer-only CE or candidate CE. Both suffix and answer losses are recorded for the LLM readout. Training accuracy is the first answer-token argmax; validation/test accuracy uses the generated response and strict parser. The diagnostic classifier uses eight-class CE and its records identify the classifier readout.

`configs/vlm/*.json` are flat settings with optional `extends`; CLI arguments override them. `_pgm_base.json` and `_4k_base.json` hold shared settings. They are portable rerun recipes, not claims that the refactored runner has already reproduced archived full-data scores. Historical orchestration used separate initialization seeds, staged continuations, selective validation and campaign-specific wall-time policies; those details remain relevant when interpreting historical results.

## Environment

Keep the existing `requirements.txt` environment for established teammate runs. `requirements-vlm.txt` records the historical CUDA environment for the added families: Torch 2.7.1, Transformers 5.1.0, PEFT 0.18.1. Create a user-owned environment rather than replacing someone else's installation. The architecture hooks target native LFM2-VL. Tiny-model checks were run under Transformers 5.1.0 / PEFT 0.18.1 and Transformers 5.17.0 / PEFT 0.21.2. The existing slot pipeline requires the newer environment: one of its tests imports a vision utility absent from 5.1.0.

The integration's local CPU test versions and completed checks are recorded in `integration_validation.json`. Those tests use tiny randomly initialized models and require no model/data downloads. A production H200 rerun with full checkpoints was not part of this code integration.

The model aliases `lfm25`, `smolvlm2`, and `spatialevo` resolve to immutable base revisions in `pgmvlm/registry.py`. Existing full model IDs remain supported. Run commands below from the repository root.

## Data: use the exact selected questions

For raw PGM `.npz` files, use the existing 240k list and 1k validation list. The 240k file records the original selection order. This is **20% of the official 1.2M training questions**, not 20% of whatever directory happens to be present. Do not combine `--train_list` with a different `--train_samples` or `--train_fraction`.

`data_manifests/vlm/pgm_train4096.txt` is the exact historical 4,096-question diagnostic subset. Every label was checked against the existing shared 240k list; all 1,000 validation IDs were matched. See `data_manifests/vlm/split_audit.json` for checksums and the precise verification boundary. The `_ids.txt` versions select canonical IDs from a contact-sheet JSONL manifest instead of filenames.

PGM data are reshaped from their stored bytes to `(16,160,160)`; transposing the declared `(160,160,16)` shape would scramble the panels. The direct-answer renderer and CLEVR preparation/label checks are transferred from the original implementation in `pgmvlm/manifests.py`.

For an existing portable contact-sheet export, pass `--train_manifest`, `--val_manifest` or `--manifest`; image paths must stay within the manifest directory. Image checksums are verified when present. For CLEVR, prepare each task separately:

```bash
python -m pgmvlm.manifests clevr-art --root "$CLEVR_RENDER_ROOT" \
  --task rmts --regime ood --split train --output "$PREPARED/rmts-train.jsonl"
```

The preparation command preserves official labels and native rendered scenes. It does not render Blender scenes or download data. Supply a separate training-domain validation manifest containing held-out training questions; **do not use the OOD test manifest for checkpoint selection**. Existing CLEVR export layouts may differ from the teammates' packed slot datasets; use the documented native OCRA render layout or an existing portable manifest.

## Train, resume, evaluate: the same entry points

```bash
# PGM LoRA, exact 240k subset and 1k validation; model aliases select pinned bases.
python lora/train_lora.py --config configs/vlm/lfm25_pgm_lora.json \
  --train_dir "$DATA/train_full/neutral" --val_dir "$DATA/val_full/neutral" \
  --out_dir "$RUNS/lfm25-pgm"

# Repeat the SAME command to resume. --epochs is the total target, not extra epochs.
# For SmolVLM2 or SpatialEvo use their corresponding *_pgm_lora.json recipes.

# Full PGM test; omit --adapter for zero-shot.
python lora/eval_vlm.py --protocol answer_letter --model_id lfm25 \
  --adapter "$RUNS/lfm25-pgm/best" --npz_dir "$DATA/test_full/neutral" \
  --out_dir "$RUNS/lfm25-pgm-test" --batch_size 1
python lora/eval_vlm.py --merge_dir "$RUNS/lfm25-pgm-test" --expected_n 200000

# CLEVR RMTS or identity: task/prompt/options come from each manifest.
python lora/train_lora.py --config configs/vlm/lfm25_clevr_lora.json \
  --train_manifest "$PREPARED/rmts-train.jsonl" --val_manifest "$PREPARED/rmts-validation.jsonl" \
  --out_dir "$RUNS/lfm25-rmts"
python lora/eval_vlm.py --protocol answer_letter --model_id lfm25 \
  --adapter "$RUNS/lfm25-rmts/best" --manifest "$PREPARED/rmts-test.jsonl" \
  --out_dir "$RUNS/lfm25-rmts-test"
```

`best/`, `last/`, and `final/` contain trainable weights, not another copy of the VLM. `state.pt` holds optimizer/RNG state. `contract.json` identifies the data order, exact ID lists, model and objective; resuming with a different contract fails. `metrics.jsonl`, `best.json`, and `args.json` explain training progress and selection. Checkpoints now commit weights and optimizer/RNG state together, retaining a verified previous generation for recovery. See [checkpoint import and recovery](CHECKPOINTS_AND_RECOVERY.md).

`--select_by accuracy_loss` chooses highest validation accuracy, breaking ties by lower validation loss; `--select_by loss` chooses lowest loss. `--early_stop_patience N` stops after N completed validation checks without the configured loss improvement. It is disabled by default. An epoch is one pass over the chosen subset; repeated epochs do not increase unique-data coverage. `position`, `python_epoch`, and `nested_first` data-order settings preserve the existing NumPy schedule, the diagnostic Python shuffle, and the original PGM first-pass ordering respectively.

Historical LoRA `adapter/` directories can be evaluated with the matching base/protocol or supplied through `--init_adapter` for a new training run. The offline converter imports old relational/plugin containers for evaluation or a new optimizer stage. Campaign-specific optimizer state remains **not interchangeable** with this trainer's resume format; see [supported formats and commands](CHECKPOINTS_AND_RECOVERY.md).

## Architectural experiments

| `--extension` | Insertion / connection | Training modes |
|---|---|---|
| `early` | Residual correction to vision patches before the projector | `module`, `lora_module` |
| `late` | Residual correction to projected visual tokens | Same |
| `joint` | Prefix correction after LFM block 2; supervised answer positions excluded | Same |
| `memory` | Original pooled symbolic memory read by language hidden states after block 2 | Same |
| `dual` | Separate visual/text relation streams combined in the visual correction | Same |
| `rca_plugin` | Symbol-valued RCA correction alongside the first native full-attention layer, at visual positions | Same |
| `rca_memory` | Unpooled RCA memory read by subsequent language positions | Same |

These are experimental architectures, including unsuccessful experiments. They do not reproduce Slot Abstractor's object slots. `module` freezes **every original VLM parameter**; `lora_module` trains the new module and language LoRA while keeping base weights frozen. There is no full-base end-to-end update hidden behind that name.

```bash
# Exact historical 4k questions, frozen base, small RCA contribution, validation stopping.
python lora/train_lora.py --config configs/vlm/lfm25_4k_rca_plugin.json \
  --train_dir "$DATA/train_full/neutral" --val_dir "$DATA/val_full/neutral" \
  --out_dir "$RUNS/plugin-4k"

# A subsequent joint stage is a NEW output directory/optimizer; load the learned module.
python lora/train_lora.py --config configs/vlm/lfm25_4k_rca_plugin.json \
  --train_mode lora_module --init_extension "$RUNS/plugin-4k/best" \
  --train_dir "$DATA/train_full/neutral" --val_dir "$DATA/val_full/neutral" \
  --out_dir "$RUNS/plugin-4k-joint"
```

For extension evaluation pass the **same extension settings** used in training (`--extension`, dimensions, readout, insertion, scale/cap), or a dedicated evaluation JSON containing only evaluator settings. A mismatch is rejected; the trainer's config also contains training-only flags, so do not pass it wholesale to the evaluator.

The depth/readout recipes compare 2 versus 6 layers through the frozen LLM, direct classifier heads, and a classifier-only control. Location recipes modify visual tokens after blocks 2/8/12, requiring downstream attention. Correction-cap recipes bound per-question RMS correction relative to native features. `--diagnostics_every N` records activation RMS, activation gradients and trainable-parameter gradients in `gradients.jsonl`; these are pre-clipping measurements. Frozen upstream tensors can have no gradient by design. A nonzero gradient is not evidence of useful learning.

The common diagnostic runner recomputes native features, unlike the old overnight diagnostic's cached-feature implementation. Individual recipes use their own initialization unless `--init_extension` is provided. Use `tools/init_matched_probes.py` to export the original matched depth/readout and insertion-point initializations; commands are in [the recovery guide](CHECKPOINTS_AND_RECOVERY.md). Use these distinctions when reproducing timing or attributing small differences.

## PACE and parallel execution

No jobs are submitted automatically. `env.sh` retains the teammates' paths/defaults. Point inputs at shared read-only data and outputs at your own storage; set `VLM_PYTHON` to your verified interpreter, and `VLM_USE_CONTAINER=1` if it runs inside the existing `CONTAINER`.

```bash
source env.sh
mkdir -p logs
export VLM_PYTHON=/path/to/your/environment/bin/python
sbatch slurm_portable/slurm/vlm_train.sbatch \
  --config configs/vlm/lfm25_pgm_lora.json \
  --train_dir "$DATA/train_full/neutral" --val_dir "$DATA/val_full/neutral" \
  --out_dir "$RUNS/lfm25-pgm" --time_limit_min 210 --check_lustre_quota

# Optional: four GPUs, same global batch 16. Scheduler policy/account must allow it.
sbatch --gres=gpu:h200:4 --time=04:00:00 slurm_portable/slurm/vlm_train.sbatch \
  --config configs/vlm/lfm25_pgm_lora.json \
  --train_dir "$DATA/train_full/neutral" --val_dir "$DATA/val_full/neutral" \
  --out_dir "$RUNS/lfm25-pgm-4gpu" --time_limit_min 210 --check_lustre_quota

# Four independent test shards, each with its own single-GPU allocation.
sbatch --array=0-3 slurm_portable/slurm/vlm_eval.sbatch \
  --protocol answer_letter --model_id lfm25 --adapter "$RUNS/lfm25-pgm/best" \
  --npz_dir "$DATA/test_full/neutral" --out_dir "$RUNS/lfm25-pgm-test"
```

Run the merge command after all evaluation shards finish. Re-running an unchanged shard skips finished IDs. Changed model, protocol, split or sharding settings are rejected. `--expected_n 200000` marks completion only with 200,000 unique error-free predictions and no duplicates.

Multi-GPU training uses explicit gradient averaging and disjoint global-batch slices. Only rank 0 writes shared outputs; per-rank RNG is checkpointed. Exact continuation currently requires the same GPU count. Use a new run with initialized weights if changing topology. Two-process CPU tests verify gradient/optimizer equivalence and exact training/checkpoint replay; NCCL/H200 throughput and full production runs remain unmeasured for this integration.

## Verification and provenance

```bash
python tests/test_pipeline.py                       # original teammate LoRA contracts/CLI
(cd relational && python tests/test_tiny.py)       # original slot-based pipeline
python -m unittest discover -s tests -v            # new protocols, architectures, replay, distributed checks
```

`docs/integration_source.json` records imported source hashes, including files that were not committed in the original working tree. `results/vlm/` indexes historical findings and explicitly distinguishes them from validation of this refactor. No images, pretrained weights, adapters, credentials, local CodeGraph index, or bulk prediction archives are included.

## Integration follow-up: checkpoint import, recovery, and ablations

See [CHECKPOINTS_AND_RECOVERY.md](CHECKPOINTS_AND_RECOVERY.md) for offline historical-weight conversion, transactional resume generations, rollback, and Slurm shutdown handling. Historical optimizer conversion is intentionally not claimed.

For a paired architectural bypass, rerun the same evaluation command with `--disable_extension` and a **different** output directory. The checkpoint, learned LoRA weights, inputs and all architecture settings stay identical; the added module is bypassed. This works for all five original LLM-readout probes and both RCA alternatives. Classifier-only/readout experiments have no equivalent VLM path to bypass and are rejected. Results record the bypass condition, and merging mixed conditions is rejected.

For the input-format audit, add `--pgm_presentation separate` to both training and evaluation of an answer-letter base/LoRA model. The original 16 panels are cropped losslessly from the canonical PGM contact sheet and supplied as separate images with row/column and option labels. The instruction and answer supervision remain unchanged. Composite remains the default. Separate input changes tokenization and compute; it is a distinct experimental condition, not a way to reproduce our composite-input scores. It is not enabled for CLEVR or architecture hooks whose multiple-image behavior has not been validated.

The follow-up validation record is [integration_followup_validation.json](integration_followup_validation.json): 123 primary CPU checks passed, plus targeted rechecks and historical-library compatibility. The previously queued GPU snapshot predates these follow-up changes.

## Input representation screen

The bounded [frozen-feature input pilot](INPUT_REPRESENTATION_PILOT.md) compares whole-sheet pooling, separate panels, spatial cells, and context normalization on the same 4,096 questions. It uses a direct classifier to isolate representation from LLM writeback; it does not claim a reproduced benchmark score.
