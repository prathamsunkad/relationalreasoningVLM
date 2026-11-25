# Complementarity: Slot Abstractor vs fine-tuned VLM (5000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 60.8% [59.5%, 62.2%] |
| VLM alone | 57.9% [56.5%, 59.2%] |
| either right (oracle upper bound) | 73.6% [72.4%, 74.8%] |
| ensemble, weight chosen on the other half (w_SA = 0.30 / 0.45) | 65.3% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 2255 | 787 |
| SA wrong | 639 | 1319 |

SA accuracy on the puzzles the VLM gets wrong: 37.4% (n=2106). VLM accuracy on the puzzles the SA gets wrong: 32.6% (n=1958). Correlation of correctness (phi): 0.41.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
