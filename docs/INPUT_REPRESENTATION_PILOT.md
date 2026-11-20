# Does our abstractor receive useful entities to compare?

This is a bounded input-representation diagnostic, not a new benchmark result.
All pretrained LFM2.5 weights are frozen, no LoRA is present, and the LLM is
bypassed by the **same eight-class readout used in the depth diagnostic**. This
isolates whether the abstractor can learn from better-organized vision features
before trying to communicate those features through a frozen language model.

## What the reference implementations do differently

The pinned upstream PGM implementation
([source](https://github.com/slotabstractor/slotabstractor/blob/2219512022157f8f8e5e357d8d29a44f6fbe6691/pgm/slot_abstractor_pgm.py))
encodes all 16 panels separately with a pretrained slot encoder. Its abstractor
compares object-feature vectors, uses object-position features plus matrix
row/column embeddings as symbols, normalizes features across the current
comparison sequence, and scores each nine-panel candidate matrix separately.
It does not feed a flattened, pooled whole-sheet token sequence into the head.

The teammates' `relational/` implementation identifies panel bounding boxes in
the vision feature map, samples a grid within each panel, applies learned slot
extraction (or its grid control), preserves panel/position information, and has
feature-standardization and pretraining stages. These are substantive differences
from our original `ordered_pool(..., 128)` over all native sheet tiles.

Relational attention has no universal rule that forbids patch features. The
hypothesis is that our pooling/normalization made content vectors too similar
or mixed the entities that should be compared. This pilot does **not** reproduce
upstream slots, candidate-wise scoring, Q/K initialization, or pretraining.
Failure would therefore not reject all benefits of their input representation.

## Matched arms

| Arm | Frozen features supplied to our two-layer abstractor |
|---|---|
| `sheet_flat` | Native whole-sheet image tokens, pooled into the original 128 consecutive bins. |
| `panels_bands` | Encode each of the 16 original panels separately; eight consecutive bins per panel. |
| `panels_grid` | Same independently encoded panels; a 2-by-4 spatial grid within each panel, still 128 tokens total. |
| `panels_grid_context` | Same grid, additionally normalize each feature channel across this question's 128 tokens. |
| `grid_context_classifier` | Same normalized grid and classifier, without relational layers; a readout control, not parameter matched. |

The first four share every initial trainable tensor, width 256, two layers,
four heads, 128 symbols, and the eight-way classifier. The control shares the
input projection and classifier initialization. Labels never influence image
cropping, feature extraction, pooling, or normalization. Panel order is eight
row-major context panels followed by options A-H; labels and the missing-cell
placeholder are excluded from the panel crops. No object annotations are used.

This decomposition separates whole-sheet versus separate-panel encoding,
one-dimensional versus spatial pooling, and context normalization. Separate
encoding also changes effective visual resolution and encoder cost; a gain in
that contrast alone cannot be attributed solely to explicit panel boundaries.
Fixed spatial cells are **not** learned object slots.

## Data, loss and budget

- Reuse the exact earlier 4,096 official-training questions and 1,000 official
  validation questions, with the same 512-question training monitor. Pin
  `selected-rows.json` by SHA256; verify every contact-sheet image's checksum.
- Ten epochs per arm: 2,560 updates, batch 16, 40,960 presentations, still only
  4,096 distinct training questions. Every arm receives the same next batch
  before advancing; stop all arms at a common update if the time budget ends.
- AdamW, learning rate 1e-4, 20-update warmup, weight decay 0.01, clipping 1.0;
  initialization seed 20261118 and data-order seed 20261112.
- Eight-class answer cross entropy; evaluate all 1,000 validation questions
  each epoch. Select lowest complete validation loss. No 200k test evaluation.
- One H200, four CPUs, 48 GiB RAM, at most two hours **including extraction**;
  reserve ten minutes for final evaluation/checkpoints. No automatic follow-up.

## Efficient extraction without changing the sheet control

For frozen token features x, LayerNorm is `norm(x)*gamma + beta`. Fixed mean
pooling commutes with its learned affine and the subsequent linear projection.
We therefore cache the pooled, non-affine normalized features in FP32 and train
the original affine/projection and abstractor afterward. CPU tests compare
logits, all gradients and the next optimizer update against the uncached original
classifier. A real-feature equivalence check also runs inside the allocation.

Three cache arrays use about 15 GiB of node-local disk and are removed on exit.
The script rejects RAM-backed/shared-storage caches, checks quota before saving,
hashes pretrained weights before and after extraction, and never writes to an
existing run. The specialized pilot saves best/latest heads but is **not** a
replacement for the shared trainer's long-run resume machinery.

## Outputs and interpretation

`features.json` records token cosine similarity, feature variation, native grid
shapes, extraction cost, and confirmation that base weights did not change.
`evaluation-curve.jsonl`, `learning-curve.jsonl`, and `summary.json` contain matched
training/validation curves, prediction histograms, and actual coverage. Small
validation prediction records identify every question. A final cyclic image
shuffle uses the same final head/targets to check image dependence.

Compare `panels_bands - sheet_flat`, `panels_grid - panels_bands`, and
`panels_grid_context - panels_grid`. Compare the best grid recipe against its
classifier-only control to assess whether the relational stack adds anything.
A useful screen is at least 5 percentage points and 0.05 lower validation CE at
matched exposure, with worse performance after image shuffling. This is a
practical decision threshold, not a statistical significance test or a claim
of convergence. One seed is insufficient for a final paper claim.

```bash
# Inside the allocated compute job; use the established CUDA environment.
python tools/input_representation_pilot.py \
  --plan /path/to/resolved-plan.json \
  --output /path/to/new/input-pilot \
  --deadline UNIX_DEADLINE_SECONDS
```

Start from `configs/vlm/lfm25_input_probe.json`; fill in the existing selected-row
file and image roots. The portable launcher is
`slurm_portable/slurm/vlm_input_probe.sbatch`. All code stays on `merged-yiran`.
