# Historical direct-answer results

These are **archived results produced before this code integration**, not new benchmark runs of the refactored trainer. The model weights, question images and full prediction archives are intentionally external. The existing teammate results elsewhere in this repository use their own documented scoring protocol.

| Model | CLEVR RMTS, 10k | CLEVR identity, 10k | Initial PGM adaptation |
|---|---:|---:|---:|
| lfm25 | 99.39% | 93.07% | 13.10% (full, 200,000 questions) |
| smolvlm2 | 89.05% | 68.54% | 12.47% (partial, 118,574 questions) |
| spatialevo | 100.00% | 95.79% | 13.32% (partial, 119,104 questions) |

The initial PGM adapters used only 9,000 sampled official training questions and a separate 1,000-question official validation set. CLEVR test scores are OOD scores on held-out visual properties. PGM neutral test scores are not OOD. Partial PGM results are not full 200k benchmarks; compare baseline predictions on matching IDs when evaluating improvements. `initial_lora_summary.json` contains source-summary hashes.

The historical training curves, run configurations, validation summaries, checkpoint-selection records, and completed diagnostic figures are now in [run_records/](run_records/README.md). Its index and SHA256 manifest identify the original files, deduplicate copies, and distinguish available snapshots from complete runs. Verify the bundle with `python3 results/vlm/run_records/verify.py`; weights and full prediction archives remain external.

The later PGM runs used the fixed 240,000-question pool for repeated epochs:

- [Four-epoch campaign](pgm_four_epochs.md).
- [Six-epoch campaign](pgm_six_epochs.md): LFM2.5 75.53%, SpatialEvo 61.87%, both full 200k tests.
- [LFM checkpoint selected at epoch 6.73](pgm_epoch673.md): 76.69%, full 200k test.

Those tables include published Slot Abstractor reference scores. They are not matched-compute reproductions. Checkpoint selection uses validation; test scores are reported separately.

The five original architectural probes, their frozen variants, and later plugin/memory experiments are included as experimental code, despite weak learning in the archived campaigns. Their presence is not an accuracy claim. See [the integration guide](../../docs/VLM_INTEGRATION.md#architectural-experiments) for the exact insertion sites and modes. The newer diagnostic recipes support capacity, classifier-readout, gradient and correction-size investigations; no new research runs were submitted during this migration.

## Current InternVL interleaving campaign

The [November 25 training snapshot](../internvl_interleave/2026-11-24-two-gpu/README.md)
contains the two-H200 epoch-6-then-epoch-4 candidate-scoring experiment: exact
plans, passed pilot checks, per-update training/validation logs, checkpoint
pointers and plots. It is a partial training snapshot with 63,648 questions
processed, not a completed epoch or benchmark test. The export preserves and
verifies remote SHA256; weights and feature caches remain external.
