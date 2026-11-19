# PGM test comparison

Only rows explicitly marked complete cover all 200,000 questions.

| Model | Test questions | Complete | New test accuracy |
| --- | ---: | --- | ---: |
| lfm25 | 200,000 | True | 76.69% |

| Model | Earlier run | Matched questions | Earlier → new | Difference (pp) |
| --- | --- | ---: | ---: | ---: |
| lfm25 | Six-epoch checkpoint | 200,000 | 75.53% → 76.69% | +1.15 |

Slot Abstractor reports **71.25% with 20% training data** and **91.50% with all training data** ([paper, Table 5](https://arxiv.org/html/2403.03458v2)). These are published reference scores, not a compute- or epoch-matched reproduction. Our validation scores are not test scores.
