# Complementarity: Slot Abstractor vs fine-tuned VLM (5000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 61.0% [59.6%, 62.3%] |
| VLM alone | 40.1% [38.8%, 41.5%] |
| either right (oracle upper bound) | 68.7% [67.4%, 70.0%] |
| ensemble, weight chosen on the other half (w_SA = 0.65 / 0.60) | 61.2% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 1623 | 1427 |
| SA wrong | 384 | 1566 |

SA accuracy on the puzzles the VLM gets wrong: 47.7% (n=2993). VLM accuracy on the puzzles the SA gets wrong: 19.7% (n=1950). Correlation of correctness (phi): 0.33.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
