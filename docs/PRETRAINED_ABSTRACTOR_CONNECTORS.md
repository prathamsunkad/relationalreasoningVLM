# Pretrained Slot Abstractor → frozen PGM-finetuned LFM

## Current decision — 2026-11-23

The user shifted priority from learning fresh Abstractors through a frozen VLM
to learning connectors between two already-trained models. Start from the best
PGM-finetuned LFM2.5 checkpoint, including its existing LoRA weights; keep all of
those weights fixed. Keep the supplied Slot Attention encoder and Abstractor
fixed too. Initially train only the connector. Updating the language decoder
through LoRA is a separate, later experiment, not part of the first comparison.
The Slot Attention reconstruction decoder is not the language decoder and does
not need training for this experiment.

The previous campaign `7238324` is stopped: Early/Late/Joint completed; pending
Dual, Memory-P, RCA plugin and Memory-C (`7238324_3`–`7238324_6`) were cancelled
on request. Preserve every completed result and checkpoint. Retire the old
Early/pre-projector and Late/post-projector visual interventions from our future
experiments; teammates cover that direction. No new GPU jobs have been submitted.

Implementation is on `codex/pretrained-abstractor-connectors`. The old Case-A
trainer is not used. The first allocation is a shared correctness/throughput
pilot; larger training is not queued by this script.

## Verified components

| Component | Checkpoint/evidence | Role |
| --- | --- | --- |
| LFM2.5-VL-1.6B | Base revision `919fde3d022e3f90a4716006f993938ee8c2eb97`; adapter `checkpoint-101000-ecb566b9/adapter`, SHA256 `b7c369ff0ffbf996f6c38a6bc093e2c2201eaa48b262e4a0191d409478ed88c7` | Frozen VLM, including the trained LoRA |
| Slot encoder | `slot_ae_pgm20.pth.tar`, SHA256 `9146e233e130ff7b0544c031ad47a877f8b63105fa35f3717c640bd817e8b5ee` | Frozen panel encoder; 16 slots, width 32, three slot iterations |
| Slot Abstractor | `slot_abstractor_pgm_pgm20_1gpu_d24_lr8e-05_run0_best.pth.tar`, SHA256 `e1fa6cf42f172eecf4a340cc5e1306d7beaf8cba2c89eeb46c4f3b5a66c966b2` | Frozen 24-layer, 8-head reasoner; width 32, FFN 512 |

The Abstractor checkpoint safely loads on CPU with `weights_only=True`; its
metadata selects zero-based epoch 63 and records 62.8% validation accuracy.
The scorer contains 2,378,817 parameters; the complete slot autoencoder contains
169,506 parameters, including its reconstruction decoder. These metadata checks
are not an inference/evaluation reproduction.

The selected LFM previously reached 77.3% on 1,000 validation questions and
76.6865% on the full 200,000-question neutral test using direct answers. Reuse
these existing results and the supplied Abstractor's archived results. The user
explicitly ruled out repeating standalone evaluations, overlap analysis, or the
teammates' ensemble study. Do not compare test and validation numbers as matched
results, or transfer teammates' results to LFM without matching model/protocol
provenance. No result here establishes the paper's reported accuracy.

## Preserve each model's trained input interface

- **LFM:** use its historical 700×1016 contact sheet, exact direct-answer prompt,
  native processor settings, answer-letter/chat-ending CE and generation parser.
  Do not silently switch this checkpoint to Yes/No candidate scoring or the
  previous pilot's single-tile settings. That would change the baseline too.
- **Abstractor:** crop the same question into its original sixteen panels using
  the supplied `layout.json`; convert RGB to grayscale, scale to [-1,1], resize
  from 160×160 to 80×80 exactly as its data loader does. Preserve context
  normalization and row/column symbols. Pin resize/antialias behavior to the
  reference pipeline and check it numerically.
- Extract the Abstractor's **pre-pooling relational states**, not its answer or
  final scalar scores: eight candidate sequences, each 9 panels × 16 slots × 32
  features. The original scorer must still reproduce its output from the mean of
  these states and its unchanged linear head. Tag each memory sequence with its
  candidate identity A–H; never use the correct answer to build memory.
- Cache frozen representations once for all connector arms, in a small number
  of shards. Keep the full trained slot/Abstractor computation and its random
  slot initialization reproducible. Candidate permutation tests must move the
  corresponding panel noise/features along with each candidate. Skipping the
  reconstruction decoder is permitted only after feature/score equivalence is
  demonstrated against the unmodified forward path.

The teacher remains an image-based reasoner. Different connector locations change
the **destination representation space**, not the teacher's input space. Feeding
LFM hidden states into this image-trained Abstractor would be a different model.

## First comparison: three interfaces, not three depths

All arms read the same frozen, candidate-specific relational states. Average
only the 16 **post-Abstractor** slots within each panel, giving 8 candidates ×
9 panels × 32 features (72 memory vectors). This retains candidate/panel axes
and exactly preserves the original classifier's global mean. Candidate and panel
embeddings are trainable parts of each connector; no labels or teacher scores
are supplied as input.

| Arm | Interface | What changes |
| --- | --- | --- |
| `reader` | New 8-head cross-attention reader after LFM block 8 | Post-image language states query memory; zero-initialized output projection writes a residual, capped at 0.1× the receiving state's RMS. |
| `kv_memory` | Additional keys/values in native attention at block 8 | Native queries attend jointly to original context and the 72 memory vectors; a zero-initialized gate interpolates the native/joint attention difference, with the same RMS cap. |
| `tokens` | 72 projected embeddings after the final image patch and before instruction text | Native backbone reads extra tokens. Insertions are masked out of loss and removed from returned logits; positions are recomputed consistently. |

Reader and KV arms write only to language states after the image. Their initial
outputs exactly match native LFM. Token insertion changes sequence length and
attention even at small embedding magnitude, so its initial outputs need not
match; record the initial shift. Disabling **any** connector restores native
LFM exactly. Memory keys are projected into post-RoPE key space with candidate/
panel identities, not assigned fictitious text positions. KV fusion uses a joint
softmax, rather than a second independent cross-attention reader.

The reader uses width 256 and 8 heads; the KV map uses LFM's native 8 KV heads
and head dimension 64. Trainable parameter counts are saved per arm, not claimed
to be matched. All frozen models stay in evaluation mode; connector parameters
are FP32 and LFM runs in BF16. No LoRA/decoder or teacher updates.

## Pilot and validation

- Entry point: `tools/pretrained_connector_pilot.py`.
- Architecture: `pgmvlm/extensions/pretrained_connectors.py`.
- Reference loader: `pgmvlm/pretrained_relations.py`; imports external verified
  teacher source/weights and checks exact feature parity before skipping its
  unused reconstruction decoder. The only source portability adjustment is grid
  device placement. Original slot initialization remains stochastic but is seeded
  per question/panel and cached identically across arms.
- Config: `configs/vlm/pretrained_connectors_pilot.json`; Slurm:
  `slurm_portable/slurm/pretrained_connectors_pilot.sbatch`.
- One H200 × at most two hours; 64 train/32 validation questions from the original
  4,096/1,000 ID lists. AdamW 3e-4, weight decay .01, batch 4, 16 updates, two-update
  warmup then cosine decay to 3e-5, gradient clip 1. Pilot-only optimizer checks
  are discarded before the common training sequence. No full test evaluation.
- Native image processing and exact historical direct-answer prompt; suffix CE
  is the objective, answer-token CE is also reported. All three interfaces use
  full-prefix greedy decoding without KV caching, at most 16 tokens, preserving
  the same interface in training and inference. Cached generation is explicitly
  rejected rather than silently applying an untested path.
- Checks: frozen teacher/VLM/LoRA identity, reference feature and scorer parity,
  candidate permutation, causal masking, finite/deep connector gradients after
  opening the zero gate/projection, exact reload and next-update replay, native
  bypass and one-question native/manual greedy parity. `tests/test_pretrained_connectors.py`
  covers real tiny LFM modules in FP32 and BF16. This is implementation validation,
  not a repeated standalone benchmark.
- Outputs: checksummed `relation-cache.pt`, per-arm `checks.json`, learning curves,
  initial/final validation predictions, train-monitor predictions, `latest.pt`
  connector+optimizer+RNG state, config/backend/provenance and summaries. No frozen
  model copies. `replay.pt` is a discarded correctness-test checkpoint, not a
  selected model. Fresh output directories and quota/headroom checks are required.

The pilot reports the final common update, not a validation-selected accuracy.
Larger learning remains a separate launch decision using measured throughput:
use the original 4,096 train/1,000 validation IDs, select minimum complete
validation suffix loss, track answer loss and generated accuracy, and retain
resumable connector-only state. No automated resubmissions or full-test jobs.
Do not interpret incomplete arms or 32-question pilot accuracy as full results.

## Authorized 5% campaign (2026-11-23)

The user authorized one epoch on the same 60,000 questions for all three
connectors, with both pretrained models frozen, after every pilot arm passes.
This supersedes the previously proposed 4,096-question learning stage.
Original pilot 7266249 failed before training because the externally shared
reference code needed colorlog and einops. Pinned wheels are supplied in an
isolated runtime, without modifying the shared environment. Strict CPU reference
checkpoint loading passed; the repaired GPU pilot is 7271131. Model/connector
source and numerical pass criteria are unchanged.

`tools/prepare_connector_subset.py` verifies the existing 240k manifest and its
seeded SHA256 ordering, selects its first 60k rows, and preserves the historical
1k validation set. No images are copied. Training manifest SHA256:
`f0141b074e8abfba8c2a148dedb71aa115cca93857e3648da5eca568c1a071c2`.
Validation SHA256: `d3f685f7400d5bda50cb6077fd06e2b6054a295c0757aa64127f6e88937876c6`.

Entry points: `tools/train_pretrained_connectors.py` (shared feature cache or
single-connector training), `tools/launch_pretrained_connectors.py` (idempotent
conditional submission), `configs/vlm/pretrained_connectors_5pct.json`, and
`slurm_portable/slurm/pretrained_connectors_train.sbatch`. Dependency pins:
`requirements-pretrained-connectors.txt`. The existing pilot architecture/input/
loss code must match byte-for-byte. The full jobs never warm-start from pilot
connector weights.

Fresh connectors, batch 4, 15,000 optimizer updates (60,000 unique questions),
AdamW 3e-4, weight decay .01, clip 1, 150-update warmup then cosine decay to 3e-5.
No early stopping before the requested epoch. Validate on all 1,000 IDs every
3,750 updates (15k questions); save minimum complete validation suffix CE plus
final weights. Save transactional connector/optimizer/RNG checkpoints every 100
updates; resumed jobs continue the same shuffled pass. No baseline reevaluation,
LoRA updates, extra epochs or full-test evaluation.

The 61k frozen-teacher banks occupy about 562 MB (FP32), stored in 256-question
shards and reused by all three arms. Decoded images use a bounded cache. The
launcher waits for the shared cache to be complete before submitting training.
Pilot throughput sizes single-H200 allocations, capped at 16 hours per job;
hourly monitoring may resume clean allocation-limited cache/training work only
until the one epoch and complete final validation finish. Material failures stop
for inspection; pass thresholds are never relaxed. Submission intents/receipts
prevent duplicate jobs. Checkpoint retention touches only new run generations.

## What the experiment can establish

The hypothesis is that a pretrained relational representation supplies useful
information that LFM does not already exploit. Teammates' overlap/ensemble
results already motivate this direction; repeating that study is not required.
The immediate question is whether any of the three connectors improves the
chosen best LFM. Existing ensemble results provide context, with their original
model/protocol boundaries, rather than a new experiment in this round.

The connector's 4,096 examples are **additional connector training**, not the
total data used by the system. LFM has seen the fixed 240k subset over multiple
epochs; the supplied encoder/reasoner were also trained before sharing. Confirm
their exact subset hashes/provenance before making a matched-data or efficiency
claim. Preserve the completed fresh-Abstractor runs as historical evidence, not
as directly matched controls for this new frozen, finetuned starting point.
