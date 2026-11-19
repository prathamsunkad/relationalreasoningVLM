# PGM test comparison

Only rows explicitly marked complete cover all 200,000 questions.

| Model | Test questions | Complete | New test accuracy |
| --- | ---: | --- | ---: |
| lfm25 | 200,000 | True | 75.53% |
| spatialevo | 200,000 | True | 61.87% |

| Model | Earlier run | Matched questions | Earlier → new | Difference (pp) |
| --- | --- | ---: | ---: | ---: |
| lfm25 | Best checkpoint from four-epoch training | 200,000 | 69.60% → 75.53% | +5.94 |
| spatialevo | Best checkpoint from four-epoch training | 200,000 | 58.19% → 61.87% | +3.68 |

Slot Abstractor reports **71.25% with 20% training data** and **91.50% with all training data** ([paper, Table 5](https://arxiv.org/html/2403.03458v2)). These are published reference scores, not a compute- or epoch-matched reproduction. Our validation scores are not test scores.
