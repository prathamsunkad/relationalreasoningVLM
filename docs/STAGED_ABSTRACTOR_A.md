# Case A: seven Abstractors with a frozen LFM

> **Status, 2026-11-23:** This campaign is stopped by user request. Early, Late and Joint completed; pending jobs `7238324_3`–`7238324_6` were cancelled. Preserve the results; do not relaunch. The new priority is [pretrained Abstractor connectors](PRETRAINED_ABSTRACTOR_CONNECTORS.md).

The staged experiment now has a separate implementation. Existing historical
probes and their checkpoint formats are unchanged. A trains randomly initialized
Abstractors/connectors through the original pretrained LFM2.5-VL-1.6B, pinned at
`919fde3d022e3f90a4716006f993938ee8c2eb97`. All original VLM weights are frozen,
including the vision encoder, native projector and language backbone. No LoRA,
branch pretraining, learned slot extractor, or interleaving sweep is included.
Cases B (branch pretraining/alignment), C and D (later LoRA) are not submitted.

## Seven designs

| Design | Relation source | Where the relation is used |
| --- | --- | --- |
| Early | Vision encoder patch features | Fuse into the same patches before the native projector |
| Late | Projected visual tokens | Fuse into the same visual tokens before the LLM |
| Joint | Contextualized visual and instruction states after block 2 | Joint Abstractor; fuse into visual and instruction states |
| Dual | Separate contextualized visual and instruction streams after block 2 | Separate Abstractors; cross-read/MLP combine; fuse into visual states |
| Memory-P | Projected visual tokens | Language queries read memory at attention block 2 |
| RCA plugin | Visual states entering attention block 2 | Fuse into that block's visual attention outputs |
| Memory-C | Same contextualized visual source as RCA plugin | Same language-memory reader and insertion as Memory-P |

Block numbers are zero-based; block 2 is LFM's first full-attention block. Earlier
blocks contain convolution. Memory-P and Memory-C share their reader design and
write point. They differ in the representation from which memory is constructed.
Memory-C and the plugin share their relation builder. Joint/Dual use instruction
states after the image; the PGM prompt is largely constant, so these do not test
broad language understanding. This scoring-only implementation rejects generated
suffixes, cached decoding, padded batches and unsupported model families.

## Input, grouping, objective

Use the shared `pgmvlm.data.PROMPT_TEXT` verbatim. Each question supplies eight
520x520 RGB completed 3x3 matrices: original 160px panels, 10px margins/gaps,
no answer labels. Rendering is independent of the answer. Native LFM processing
uses one whole-image tile (`do_image_splitting=False`), preserving a verifiable
mapping from native patch centres back to the matrix. Native resizing and image
normalization remain enabled. Candidate scores are FP32 `logit(Yes)-logit(No)`;
train with one eight-way CE and predict argmax. No binary per-candidate loss.

The Abstractor receives 36 addressed visual groups: nine panels x a 2x2 local
grid, mean-pooled within each group. These are fixed spatial groups, **not object
slots**. Margins are excluded; upstream features may already mix information
across panels. Group addresses carry forward while the feature vectors come from
each design's own insertion point. Joint/Dual also use 16 ordered prompt groups.
This removes the previous pooling across arbitrary sheet boundaries without
adding an unverified learned slot model. Slot-based grouping remains a future
choice, not an implemented or pretrained component in this Case A.

The common Abstractor has six layers, width 256 and eight heads. Each layer uses
content-derived Q/K, symbolic V, a residual and a feed-forward block. Symbols
start from learned panel/cell identities. Fusion projects relation states into
the destination space, matches each vector's RMS to the native vector, then uses
`(1-alpha)*native + alpha*relation`, with learned `alpha` initially 0.1. Thus a
50x/302x correction cannot arise from raw projection scale. Initial branch output
is random and nonzero, so all branch layers can receive gradients immediately.
For memory designs the projected relation is a language-query readout; for RCA
it is written into visual attention output. No additional sequence tokens.

## Small correctness pilot (not an accuracy benchmark)

One H200, at most two hours TOTAL for all seven designs; no dependent jobs. Use
the same 64 questions sampled deterministically from the historical 4,096-question
pool inside the fixed 240k PGM train subset, and the same first 32 official
validation questions. All IDs and source selection checksums are saved.
Each design receives 16 updates, batch 4, one pass over 64 unique questions.
AdamW 3e-4, weight decay .01, clip norm 1; two-update warmup followed by cosine
decay to 3e-5. Final checkpoint, no validation-based model selection. Report final
16-question train monitor and 32-question validation separately.

Checks: alpha=0 exactly restores native scores; candidate permutation equivariance;
finite nonzero gradients in the first/last relational layers; added weights update;
base weights stay bitwise unchanged; checkpoint predictions and the next optimizer
update replay exactly. Check/replay updates are discarded. Log loss, gradients,
fusion RMS/share, throughput, predictions, optimizer state and source identity.
A passed pilot means the software runs correctly, not that the architecture learns
or beats LoRA. Failed/time-limited arms must not trigger larger submissions.

## Entry points and state

- `pgmvlm/extensions/staged_abstractor.py`: modules and native hooks.
- `tools/staged_abstractor_pilot.py`: matched candidate scoring, checks and logging.
- `configs/vlm/lfm25_staged_a_pilot.json`: portable recipe; fill data paths.
- `slurm_portable/slurm/vlm_staged_a_pilot.sbatch`: bounded allocation wrapper.
- `tests/test_staged_abstractor*.py`: geometry, fusion, native hooks, frozen weights,
  reload and deterministic next-update replay.

Set `VLM_REPO`, `VLM_PYTHON`, `VLM_A_PLAN`, `VLM_A_OUTPUT`, the pinned runtime and
HF cache before submission. The submit receipt records the resolved plan, source
commit, bundle checksum and scheduler command. Outputs use per-design directories:
`spec.json`, `checks.json`, `learning-curve.jsonl`, `initial-validation.json`,
`final-training.json`, `final-validation.json`, `latest.pt` plus SHA256, `summary.json`.
Only module/optimizer states are saved, not copies of frozen base weights.

Interrupted pilots require explicit `--resume`; model/data/config/source identity
and checkpoint checksums must match. Completed arms are skipped. Optimizer,
update count, RNG and learning-rate schedule are restored. No automatic resume,
job submission, source mutation, or old-checkpoint deletion. Larger training
needs a separate reviewed data/order, validation-selection and budget plan.
