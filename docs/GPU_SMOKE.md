# Real-model GPU smoke test

This is a bounded integration check, not a benchmark or a convergence run. It uses the public training and evaluation commands, cached pinned pretrained weights, and small disjoint manifests containing real PGM contact sheets.

The default six cases are language LoRA for LFM2.5, SmolVLM2 and SpatialEvo; frozen LFM plus RCA plugin; LFM LoRA plus the late relational module; and frozen LFM plus RCA memory. Other insertion sites remain covered by the CPU suite rather than this smoke test.

Each case:

1. Performs one optimizer update, saves weights/optimizer/RNG, and stops cleanly.
2. Reloads and continues to two updates; compares the final trainable tensors **exactly** with an uninterrupted two-update run using the same deterministic settings.
3. Confirms that parameters actually changed between updates, and that a frozen-module case contains no LoRA adapter.
4. Reloads the selected checkpoint in two separate evaluation processes. All questions must be scored, generated responses must match, and reported losses must agree within 1e-6.

Global batch is 2; each two-update trajectory presents 4 questions. The reference trajectory repeats those updates independently. Validation accuracy on these few questions is a software-check observation only.

Prepare 4–32 training and 2–8 validation rows in separate portable `train.jsonl` and `validation.jsonl` files, retaining their original IDs and image checksums. Copy the referenced PNGs beneath that directory; do not use symlinks that escape the manifest root. Use a fresh output directory and a verified environment with cached model revisions.

```bash
export VLM_REPO=/path/to/pinned/integration/checkout
export VLM_PYTHON=/path/to/verified/environment/bin/python
export HF_HOME=/path/to/existing/huggingface/cache
export VLM_SMOKE_DATA=/path/to/small/manifests
export VLM_SMOKE_OUTPUT=/path/to/new/smoke/output
# If PEFT/Accelerate are provided by an existing verified overlay, set PYTHONPATH too.
sbatch --output=/path/to/logs/smoke-%j.out --error=/path/to/logs/smoke-%j.err \
  slurm_portable/slurm/vlm_smoke.sbatch
```

The launcher requests one H200, 2 CPUs, 16 GiB host memory and at most 20 minutes. The runner stops after an 18-minute internal budget, requires visible CUDA inside a Slurm allocation, checks storage headroom before checkpoints, and never submits follow-up jobs. `--cases lfm25_lora` selects a narrower smoke test; otherwise all six cases run sequentially.

`summary.json` records the GPU, library versions, optional `VLM_SOURCE_COMMIT`, question IDs/hashes, exact commands, timings, per-case outcomes and any failure. Retrieve small summaries/logs for review; leave model/optimizer tensors on the cluster. A queued job or incomplete summary does not establish GPU validation. Keep the PR unmerged until the chosen checks have passed.
