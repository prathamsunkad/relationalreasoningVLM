# Frozen InternVL connector pilot

This adds the three pretrained-Abstractor interfaces to **InternVL3.5-1B-HF + the shared epoch-6 PGM LoRA**. The VLM, existing LoRA, and external Slot Abstractor remain frozen. Only a fresh connector updates. The existing LFM campaign and its frozen source are unchanged.

## Input and objective

Reuse `pgmvlm/data.py:CandidateCollator` and `pgmvlm/models.py:candidate_scores`: eight completed 520×520 matrices, the existing Yes/No prompt and native chat template, `crop_to_patches=True`, and eight-way cross-entropy over FP32 Yes-minus-No scores. No letter generation. The teacher's verified shared cache contains eight candidate × nine panel × 32-dimensional relational summaries per question. Each candidate sequence receives only its matching nine summaries; no candidate-index embedding, answer label, or teacher classifier score enters the connector. Candidate permutation must just permute the scores.

| Interface | Connection |
| --- | --- |
| Reader | Cross-attention from post-image language states to relations after language block 8 (zero-based), followed by a bounded residual. |
| KV memory | Project relations into extra keys/values inside block 8's native Qwen3 attention; apply a gated joint-minus-native correction, retaining native Q/K normalization, RoPE, and frozen LoRA. |
| Tokens | Project nine relation tokens into language width and insert after the final image token, before the instruction; preserve original score positions. |

Width 256, eight connector heads; reader output and KV gate start at zero. Their residual RMS is capped at 10% of the corresponding native state; inserted tokens are separately capped at native embedding RMS. Tokens do not have an identity initialization. Disable/remove restores the exact native path for every interface.

## Pilot, not an accuracy experiment

`configs/vlm/internvl_connectors_pilot.json` selects the first **8 training and 4 validation questions** from the existing 60k/1k manifests. Each arm performs a resettable three-update correctness audit, exact save/reinstall/reload/next-update replay, and eight fresh single-question updates (AdamW, lr 3e-4, weight decay .01, clip 1). It records candidate CE and scores before/after, connector gradient RMS, frozen-weight hashes, and candidate-permutation checks. The scores are too small-sample to establish efficacy.

Local verification:
```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -p test_internvl_connectors.py -v
```

GPU entry point (inside an allocation):
```bash
python tools/internvl_connector_pilot.py --plan /path/to/plan.json --output /path/to/new/pilot --seconds 1500
```

`slurm_portable/slurm/internvl_connectors_pilot.sbatch` runs one H200 for at most 30 minutes, using the teammate's verified packed runtime (torch 2.6.0+cu124, transformers 5.17.0, peft 0.21.2). It unpacks dependencies/source to node-local temporary storage to avoid scratch inode pressure. Submit only after the existing feature cache and package verification pass. The pilot never submits larger jobs. A future full training runner still needs a separate authorization and implementation; this entry point deliberately caps the dataset and update count.

The runtime plan, source archive/commit, SHA256 receipts, job receipt, backend, selected IDs, curves, validation predictions, and small connector checkpoints live under the campaign run directory. Weights/data stay outside Git.
