# Complementarity: Slot Abstractor vs fine-tuned VLM (5000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 61.0% [59.6%, 62.3%] |
| VLM alone | 48.4% [47.0%, 49.8%] |
| either right (oracle upper bound) | 70.7% [69.4%, 71.9%] |
| ensemble, weight chosen on the other half (w_SA = 0.50 / 0.55) | 62.4% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 1935 | 1115 |
| SA wrong | 484 | 1466 |

SA accuracy on the puzzles the VLM gets wrong: 43.2% (n=2581). VLM accuracy on the puzzles the SA gets wrong: 24.8% (n=1950). Correlation of correctness (phi): 0.38.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
