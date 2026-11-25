# Overlap check: LoRA alone vs the branch alone

5000 problems (the shared validation set), the same for every model. Chance is 12.5%.

## 1. Accuracy of each model

| Model | Checkpoint step | Accuracy |
|---|---|---|
| `br636k` | 636000 | 62.0% |
| `lora20e1` | 3000 | 55.2% |
| `lora20e2` | 3000 | 68.1% |
| `lora20e4` | 3000 | 81.2% |
| `lora5p0` | 14000 | 52.8% |

## 2. Pairs

Combined = w x LoRA log-probabilities + (1 - w) x branch log-probabilities, argmax over the 8 candidates. 'w = 0.5' is untuned. 'CV' picks w on a random half and scores the other half (100 splits); its gain is against LoRA alone on the same halves. McNemar compares the w = 0.5 combination with LoRA alone, problem by problem (p < 0.05: unlikely to be noise).

| LoRA | Branch | Both right | Only LoRA | Only branch (if independent) | Neither | Either right | Combined w=0.5 | Gain | McNemar p | CV gain (w picked) |
|---|---|---|---|---|---|---|---|---|---|---|
| `lora20e1` | `br636k` | 2228 | 533 | **874** (1389) | 1365 | 72.7% | 63.9% | +8.7 | 7.31e-47 | +8.6 (w~0.40) |
| `lora20e2` | `br636k` | 2574 | 829 | **528** (991) | 1069 | 78.6% | 70.4% | +2.3 | 5.18e-06 | +2.7 (w~0.65) |
| `lora20e4` | `br636k` | 2864 | 1196 | **238** (583) | 702 | 86.0% | 81.1% | -0.1 | 0.835 | +0.1 (w~0.85) |
| `lora5p0` | `br636k` | 2123 | 515 | **979** (1465) | 1383 | 72.3% | 63.0% | +10.2 | 3.44e-64 | +10.5 (w~0.25) |

## 3. By rule (`lora20e1` vs `br636k`, combined at w = 0.5)

A problem has 1-4 rules and counts once in each group it has. Small groups are noisy: with n = 100, a difference under ~10 points can be chance.


**Relation**

| Relation | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| consistent_union | 1453 | 56.0% | 63.7% | 64.9% | +7.8 |
| OR | 1275 | 61.4% | 66.5% | 68.6% | +5.1 |
| XOR | 1271 | 44.2% | 44.0% | 47.9% | -0.2 |
| AND | 1250 | 53.0% | 60.5% | 61.6% | +7.4 |
| progression | 1124 | 54.4% | 67.1% | 68.2% | +12.7 |

**Attribute**

| Attribute | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| color | 2243 | 50.7% | 56.4% | 59.6% | +5.7 |
| type | 2207 | 73.4% | 77.0% | 79.7% | +3.6 |
| size | 956 | 25.1% | 31.4% | 28.9% | +6.3 |
| position | 854 | 49.3% | 52.5% | 55.0% | +3.2 |
| number | 539 | 39.0% | 70.9% | 68.8% | +31.9 |

**Object**

| Object | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| shape | 3283 | 46.6% | 53.5% | 55.8% | +6.9 |
| line | 2979 | 62.7% | 69.7% | 70.9% | +7.0 |

**Number of rules in the problem**

| Rules | n | LoRA | Branch | Combined |
|---|---|---|---|---|
| 1 | 3426 | 58.3% | 65.0% | 66.9% |
| 2 | 1355 | 48.8% | 56.1% | 58.3% |
| 3 | 213 | 45.5% | 53.1% | 50.7% |
| 4 | 6 | 66.7% | 33.3% | 66.7% |
