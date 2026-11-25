# Complementarity: Slot Abstractor vs fine-tuned VLM (1000 problems)

| | accuracy [95% CI] |
|---|---|
| SA alone | 60.2% [57.1%, 63.2%] |
| VLM alone | 60.5% [57.4%, 63.5%] |
| either right (oracle upper bound) | 74.5% [71.7%, 77.1%] |
| ensemble, weight chosen on the other half (w_SA = 0.50 / 0.60) | 65.9% |

| | VLM right | VLM wrong |
|---|---|---|
| SA right | 462 | 140 |
| SA wrong | 143 | 255 |

SA accuracy on the puzzles the VLM gets wrong: 35.4% (n=395). VLM accuracy on the puzzles the SA gets wrong: 35.9% (n=398). Correlation of correctness (phi): 0.41.

Reading: a large 'SA right, VLM wrong' cell and an ensemble clearly above max(SA, VLM) mean the SA knows things the VLM does not, which interleaving could pass on. If the ensemble is no better than the SA alone, the VLM adds nothing the SA lacks.
