# Overlap check: LoRA alone vs the branch alone

1000 problems (the shared validation set), the same for every model. Chance is 12.5%.

## 1. Accuracy of each model

| Model | Checkpoint step | Accuracy |
|---|---|---|
| `br10` | 145000 | 37.1% |
| `br50_204k` | 204000 | 40.3% |
| `br50_210k` | 210000 | 42.1% |
| `brset` | 238000 | 44.8% |
| `lora100` | 60000 | 81.9% |
| `lora5` | 14000 | 51.9% |

## 2. Pairs

Combined = w x LoRA log-probabilities + (1 - w) x branch log-probabilities, argmax over the 8 candidates. 'w = 0.5' is untuned. 'CV' picks w on a random half and scores the other half (100 splits); its gain is against LoRA alone on the same halves. McNemar compares the w = 0.5 combination with LoRA alone, problem by problem (p < 0.05: unlikely to be noise).

| LoRA | Branch | Both right | Only LoRA | Only branch (if independent) | Neither | Either right | Combined w=0.5 | Gain | McNemar p | CV gain (w picked) |
|---|---|---|---|---|---|---|---|---|---|---|
| `lora100` | `br10` | 339 | 480 | **32** (67) | 149 | 85.1% | 81.4% | -0.5 | 0.568 | -0.5 (w~0.85) |
| `lora100` | `br50_204k` | 378 | 441 | **25** (73) | 156 | 84.4% | 80.6% | -1.3 | 0.124 | -0.5 (w~0.90) |
| `lora100` | `br50_210k` | 386 | 433 | **35** (76) | 146 | 85.4% | 81.0% | -0.9 | 0.289 | -0.7 (w~0.95) |
| `lora100` | `brset` | 405 | 414 | **43** (81) | 138 | 86.2% | 81.1% | -0.8 | 0.41 | -0.9 (w~0.90) |
| `lora5` | `br10` | 275 | 244 | **96** (178) | 385 | 61.5% | 54.1% | +2.2 | 0.0371 | +2.6 (w~0.55) |
| `lora5` | `br50_204k` | 302 | 217 | **101** (194) | 380 | 62.0% | 54.7% | +2.8 | 0.0104 | +2.1 (w~0.50) |
| `lora5` | `br50_210k` | 310 | 209 | **111** (203) | 370 | 63.0% | 55.3% | +3.4 | 0.00337 | +3.0 (w~0.55) |
| `lora5` | `brset` | 317 | 202 | **131** (215) | 350 | 65.0% | 55.4% | +3.5 | 0.00205 | +3.6 (w~0.55) |

## 3. By rule (`lora5` vs `brset`, combined at w = 0.5)

A problem has 1-4 rules and counts once in each group it has. Small groups are noisy: with n = 100, a difference under ~10 points can be chance.


**Relation**

| Relation | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| consistent_union | 306 | 50.7% | 41.5% | 55.6% | -9.2 |
| AND | 270 | 55.6% | 45.9% | 55.6% | -9.6 |
| XOR | 242 | 34.3% | 30.2% | 36.4% | -4.1 |
| OR | 236 | 58.5% | 56.4% | 64.4% | -2.1 |
| progression | 224 | 48.7% | 42.9% | 56.7% | -5.8 |

**Attribute**

| Attribute | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| type | 464 | 66.6% | 59.5% | 70.3% | -7.1 |
| color | 433 | 46.7% | 31.2% | 48.0% | -15.5 |
| size | 199 | 23.6% | 26.6% | 27.6% | +3.0 |
| position | 157 | 47.8% | 41.4% | 52.2% | -6.4 |
| number | 104 | 31.7% | 47.1% | 48.1% | +15.4 |

**Object**

| Object | n | LoRA | Branch | Combined | Branch - LoRA |
|---|---|---|---|---|---|
| shape | 653 | 41.0% | 37.1% | 45.5% | -4.0 |
| line | 594 | 60.8% | 49.8% | 63.3% | -10.9 |

**Number of rules in the problem**

| Rules | n | LoRA | Branch | Combined |
|---|---|---|---|---|
| 1 | 689 | 56.3% | 48.6% | 58.8% |
| 2 | 266 | 43.2% | 36.5% | 49.2% |
| 3 | 44 | 36.4% | 34.1% | 40.9% |
| 4 | 1 | 0.0% | 100.0% | 0.0% |
