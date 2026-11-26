# InternVL epoch-6 LoRA + Slot Abstractor: averaging control

**Interim validation comparison, not a benchmark-test result.** Job `7324410`
finished on November 25, 2026, 15:58:48–16:02:28 Eastern, on one H200
(3m40s, 0.0611 GPU-hours). The averaging job performs no training.

## Results on the same 1,000 validation questions

| Setup | Correct | Accuracy | Candidate CE |
|---|---:|---:|---:|
| Original InternVL3.5-1B epoch-6 LoRA | 616 | 61.6% | 1.04529482 |
| Pretrained Slot Abstractor | 617 | 61.7% | 1.15455155 |
| Equal-weight log-probability average (primary) | 667 | 66.7% | 0.87889355 |
| Equal-weight probability average (secondary) | 669 | 66.9% | 0.91391571 |
| Interleaving hybrid, selected update 3,500 | 668 | 66.8% | 1.00072729 |

Primary scores are `0.5*log_softmax(LoRA) + 0.5*log_softmax(Abstractor)`;
the selected answer is their argmax. For CE these scores are normalized again.
The secondary rule takes the arithmetic mean of probabilities. No weights or
temperatures were tuned. The two averaging rules must not be conflated.

The hybrid is 5.2 percentage points above the original LoRA, but just 0.1 point
above the primary average (one question): 32 puzzles are correct only for the
hybrid and 31 only for averaging; exact two-sided McNemar **p=1.0**. There is no
demonstrated hybrid advantage over averaging. The standalone models are both
correct on 477 puzzles, only LoRA on 139, only the Abstractor on 140, and neither
on 244. This shows complementary predictions, not a learned-connector benefit.

The branch and original LoRA used the project's 240k-question PGM training
subset (20% of the official training split). The hybrid checkpoint follows
3,500 additional updates / 56,000 question exposures, selected by minimum CE
on these same validation IDs. The first 500 updates train the connector only,
then connector plus language LoRA. Training is ongoing; original VLM base and
Abstractor weights stay frozen. This is equal unique training-data scope, not
equal compute or an equal-update LoRA control. It neither establishes a test
gain nor shows that the epoch-6 LoRA has converged.

## Protocol and provenance

Eight completed 520x520 matrices per question, native InternVL
`crop_to_patches=True`, native Yes-minus-No candidate scoring. The standalone
baseline has no connector or relation placeholders. Both standalone models
are frozen; the hybrid predictions are the pinned update-3,500 reference.
Exact IDs and labels agree for all 1,000 questions.

- Evaluation source: `fc9decfe582349f9ca6399fed47952b6cc335207`,
  [`tools/evaluate_internvl_average.py`](../../../tools/evaluate_internvl_average.py).
- Training source: `44ac9c74da42ed56d94d8798d24d2f8b0745a7a0`;
  [training records](../2026-11-24-two-gpu/README.md).
- Exact model/adapter/teacher/data hashes: [`records/plan.json`](records/plan.json).
- Versions, prompt and preprocessing: [`records/output/backend.json`](records/output/backend.json).
- Strict adapter, teacher/cache parity, teacher-head reconstruction and candidate
  permutation checks: [`records/output/checks.json`](records/output/checks.json).
- Standalone scores: `records/output/predictions.jsonl.gz`; hybrid scores:
  `records/hybrid-reference.json.gz`. These are predictions, not images or weights.
- Summary, job receipt, scheduler record, stdout/stderr and the original wrapper
  are included. `PROVENANCE.json` verifies exported bytes and, after lossless
  decompression, the original remote SHA256 values for all 12 retrieved artifacts.
  Models, datasets, caches and the runtime environment remain external.

Recompute the complete comparison locally, without a GPU or model weights:

```bash
python3 results/internvl_interleave/2026-11-25-average-control/verify.py
```

With the external assets and recorded runtime available, the evaluation entry is:

```bash
python tools/evaluate_internvl_average.py --plan /path/to/plan.json --output /path/to/new-output
```

The archived `run.sbatch` records the actual PACE/container command; its source
bundle, runtime archive and paths must be provisioned before reuse. It is not a
self-contained distribution of weights or data.

## Relationship to the other validation comparison

[`comparisons/val1000_update3500_paired.md`](../comparisons/val1000_update3500_paired.md)
uses separately produced Exp2 standalone scores (60.5% LoRA, 60.2% Abstractor,
66.3% equal-weight ensemble) and the same hybrid checkpoint. Those records are
preserved. This archive uses its own verified native-input H200 evaluation
(61.6%, 61.7%, 66.7%). Do not mix scores or paired statistics between the two
exports. The numerical differences have not been causally attributed; both
comparisons find no demonstrated hybrid advantage over averaging.
