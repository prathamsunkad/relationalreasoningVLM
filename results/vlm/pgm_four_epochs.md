# PGM test comparison

Only rows explicitly marked complete cover all 200,000 questions.

| Model | Test questions | Complete | New test accuracy |
| --- | ---: | --- | ---: |
| lfm25 | 200,000 | True | 69.60% |
| smolvlm2 | 58,048 | False | 15.77% |
| spatialevo | 200,000 | True | 58.19% |

| Model | Earlier run | Matched questions | Earlier → new | Difference (pp) |
| --- | --- | ---: | ---: | ---: |
| lfm25 | Zero-shot | 200,000 | 12.37% → 69.60% | +57.22 |
| lfm25 | Earlier coverage LoRA | 200,000 | 30.89% → 69.60% | +38.70 |
| lfm25 | 20% pool, one epoch LoRA | 200,000 | 43.77% → 69.60% | +25.83 |
| smolvlm2 | Zero-shot | 58,048 | 12.53% → 15.77% | +3.24 |
| smolvlm2 | Earlier coverage LoRA | 58,048 | 12.76% → 15.77% | +3.01 |
| spatialevo | Zero-shot | 173,554 | 12.97% → 58.13% | +45.17 |
| spatialevo | Earlier coverage LoRA | 200,000 | 30.03% → 58.19% | +28.16 |

Slot Abstractor reports **71.25% with 20% training data** and **91.50% with all training data** ([paper, Table 5](https://arxiv.org/html/2403.03458v2)). These are published reference scores, not a compute- or epoch-matched reproduction. Our validation scores are not test scores.
