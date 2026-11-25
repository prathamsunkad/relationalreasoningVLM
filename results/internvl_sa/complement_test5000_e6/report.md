# Complementarity: Slot Abstractor vs fine-tuned VLM (5000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 61.0% [59.6%, 62.3%] |
| VLM alone | 60.3% [59.0%, 61.7%] |
| either right (oracle upper bound) | 75.0% [73.8%, 76.2%] |
| ensemble, weight chosen on the other half (w_SA = 0.40 / 0.35) | 67.0% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 2316 | 734 |
| SA wrong | 701 | 1249 |

SA accuracy on the puzzles the VLM gets wrong: 37.0% (n=1983). VLM accuracy on the puzzles the SA gets wrong: 35.9% (n=1950). Correlation of correctness (phi): 0.40.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
