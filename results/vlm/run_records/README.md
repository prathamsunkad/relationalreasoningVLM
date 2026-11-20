# Historical training and validation records

This is a compact export of Yiran's existing local campaign archive, captured on
2026-11-19. These records were produced by the original trainers, **not by a new
run of the integrated code**. The integration code freeze was `d9822b5`; this
record-only addition does not change the training implementation.

## Where to start

| Experiment | Records |
|---|---|
| CLEVR RMTS / identity and initial 9k-question PGM LoRA | `campaigns/lfm-lora-idrules-2026-10-07`, `campaigns/adaptation-controls-2026-10-07/lfm25/rmts`, `campaigns/adaptation-repair-2026-10-07` |
| Larger PGM coverage and one-epoch LFM | `campaigns/pgm-coverage-16gpuh-2026-10-08`, `campaigns/lfm-pgm-20pct-2026-10-10` |
| LoRA continuation through the four-epoch comparison | `campaigns/pgm-best-test-2026-10-27`, `progress-checks/2026-10-19-lora-continued`, `progress-checks/2026-10-22-target-revision` |
| Six-epoch and seven-epoch continuations | `campaigns/pgm-lora-six-epochs-2026-10-28`, `campaigns/pgm-six-epoch-test-2026-11-01`, `campaigns/lfm-pgm-seven-epochs-2026-11-02` |
| LFM checkpoint selected at epoch 6.73, full test | `campaigns/lfm-pgm-epoch673-test-2026-11-05` |
| Original module + LoRA probes | `campaigns/lfm-relational-probes-2026-10-11`; final dual summary is indexed under the historical `lfm-relational-frozen-2026-10-14/convergence` alias below |
| Original module-only probes | `campaigns/lfm-relational-frozen-2026-10-14/audit-2026-11-18` |
| Fresh one-epoch plugin + LoRA | `campaigns/lfm-abstractor-fresh-epoch1-2026-11-10/plugin_lora` |
| Tiny-batch and 4,096-question plugin experiments | `campaigns/lfm-abstractor-overfit-2026-11-16`, `campaigns/lfm-abstractor-plugin-first-2026-11-16`, `campaigns/lfm-abstractor-plugin-4k-2026-11-17` |
| Completed depth/readout and insertion diagnostics | `progress-checks/2026-11-19-abstractor-diagnostics` |
| Exact LoRA settings / prompts and diagnostic analyses / figures | `reports/lora-reproduction-2026-10-14`, `reports/abstractor-capacity-gradient-audit-2026-11-18`, `reports/abstractor-diagnostics-2026-11-19` |

Use [INDEX.csv](INDEX.csv) to locate an original record's exported artifact.
Identical records are stored once; multiple source paths can point to that same
artifact, so not every original directory has a physical copy of every file.
[MANIFEST.json](MANIFEST.json) records original and exported SHA256 hashes,
byte counts, gzip content hashes, and any omitted per-question arrays.

## Read and verify

From the repository root, with ordinary Python (no GPU, model download or Torch):

```bash
python3 results/vlm/run_records/verify.py
```

`learning-curve.jsonl.gz` files preserve the original JSONL bytes after
decompression. Read them with `gzip.open(path, 'rt')` and `json.loads(line)`, or
`gzip -cd <path>`. Each row retains its original step, exposure, loss, gradient,
validation or timing fields. Schemas differ by campaign; do not assume every
loss is answer-only CE or that every accuracy is generated test accuracy.

Configs, checkpoint pointers, summaries and scalar validation records are JSON.
Embedded prediction rows and repeated example-ID lists are removed from some
JSON documents, explicitly listed in `omitted_arrays`; aggregate metrics and
other configuration fields are retained. Dataset split lists already live in
`results/splits/` and `data_manifests/vlm/`. Full prediction archives, images,
weights, optimizer states, terminal logs and credentials are not included.

To recreate the export from the original archive into a **new** directory:

```bash
python3 results/vlm/run_records/export_records.py \
  --archive-root /path/to/original/benchmarks --output /path/to/new/export
python3 results/vlm/run_records/verify.py \
  --source-root /path/to/original/benchmarks
```

The exporter writes the data, index and manifest; the second command verifies
the committed bundle against the original archive, including each transformation.

## Interpretation and coverage

- These are available local snapshots, not a claim of complete per-update logs
  for every allocation. Some continuation segments have only configuration,
  summary and validation records; others also have the full learning curve.
  Partial progress snapshots are retained as such. Missing records are not
  reconstructed or invented.
- Snapshot and continuation streams can overlap. Use job IDs, steps, cumulative
  exposure, summary status and checkpoint pointers to join a trajectory; do not
  concatenate every JSONL file or count duplicated snapshots as new training.
- `lfm-relational-frozen-2026-10-14/convergence/` is an old, misleading folder
  name: those records contain **module + LoRA** comparisons. The actual
  module-only evidence is `audit-2026-11-18/`. The final dual module + LoRA
  summary appears in the former directory and supersedes its earlier partial
  snapshot. Check recorded policies/job IDs, not just parent-folder names.
- The original frozen-memory pilot failed before training; it has no measured
  learning result. The later RCA-memory comparison was cancelled. The cancelled
  fresh LoRA-only control is not the authoritative LoRA baseline. Do not treat
  absent summaries as zero accuracy or report incomplete runs as full tests.
- Some original paths and checkpoint directories recorded in JSON refer to
  external PACE storage. They provide provenance, not a promise that weights
  are included here or that those paths remain accessible to every teammate.
- Export checks compare with the existing local archive, not a fresh cluster
  retrieval. The bundle supports auditing historical findings; it does not
  establish end-to-end numerical reproduction by the refactored runner.
