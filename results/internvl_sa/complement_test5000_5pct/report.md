# Complementarity: Slot Abstractor vs fine-tuned VLM (5000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 61.0% [59.6%, 62.3%] |
| VLM alone | 38.3% [36.9%, 39.6%] |
| either right (oracle upper bound) | 68.5% [67.2%, 69.8%] |
| ensemble, weight chosen on the other half (w_SA = 0.70 / 0.65) | 61.2% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 1538 | 1512 |
| SA wrong | 375 | 1575 |

SA accuracy on the puzzles the VLM gets wrong: 49.0% (n=3087). VLM accuracy on the puzzles the SA gets wrong: 19.2% (n=1950). Correlation of correctness (phi): 0.31.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
