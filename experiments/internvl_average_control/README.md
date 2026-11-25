# InternVL epoch-6 averaging control

Inference only: original InternVL3.5-1B epoch-6 LoRA and the pretrained Slot
Abstractor used by the interleaving campaign. Both models remain frozen; there
is no connector and no optimizer. Evaluate the exact same 1,000 PGM neutral
validation IDs. These are validation results, not the 200,000-question test.

The primary rule follows `experiments/diag_overlap/analyze.py`: take the argmax
of the equal-weight mean of each model's eight log-softmax scores. Also record
the arithmetic mean of probabilities as a separately labeled secondary rule.
Neither weights nor temperatures are fitted. Log-softmax normalizes scores but
does not calibrate confidence; the two averaging rules need not agree.

The VLM uses the training campaign's native candidate images and processor
(`crop_to_patches=True`), Yes-minus-No scores, and FP32 LoRA parameters, without
relation placeholders. The Abstractor uses the verified shared feature cache
and its original classifier head. A fresh reference extraction must agree with
the first cached example before scoring. Checkpoint hashes, exact adapter
loading, candidate permutation and unchanged frozen weights are checked.

Save original scores and IDs in `output/predictions.jsonl`. `output/summary.json`
reports both individual accuracies, both averages, and a paired comparison with
a pinned copy of the hybrid's selected validation predictions. ID sets and
labels must match exactly; partial evaluations cannot produce a complete result.
The hybrid was selected on these validation IDs, so a small validation advantage
is not itself held-out evidence of improvement.

Entry points:

```bash
python -m unittest discover -s tests -p 'test_internvl_average.py' -v
python tools/evaluate_internvl_average.py --plan /path/to/plan.json --output /path/to/output
sbatch slurm_portable/slurm/internvl_average_control.sbatch
```

Submit the Slurm script from an immutable campaign directory containing
`source.tar.gz`, `plan.json`, `hybrid-reference.json`, their `.sha256` files and
pinned `wheels/`. The wrapper uses the existing PACE runtime bundle, one H200,
4 CPUs and 24 GiB for at most 30 minutes. It does not submit, cancel or modify
other jobs. The plan pins the input manifests, both checkpoints and the hybrid
reference snapshot. The full 60k training manifest is read only to locate the
validation entries in the existing cache; no training examples are evaluated.
