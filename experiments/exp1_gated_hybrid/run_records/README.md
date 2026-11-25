# Exp1 run records

The logs of every Exp1 run (`train_log.jsonl`, and `summary.json` when the run finished), copied from `ckpt/` by `collect_run_records.py`. Checkpoints are not included.

`spread` = mean within-puzzle spread of the 8 candidate scores (0 = every candidate scored alike, i.e. collapse). Spread at end = its median over the last 10% of logged steps; collapsed = below 0.01. Runs without a spread column predate its logging. Held-out = the last 10,000 rows of `train.jsonl`, never trained on.

| run | state | step / total | best held-out loss (step) | acc at best | spread at end | collapsed |
|---|---|---|---|---|---|---|
| `ablation/cnn` | finished | 7500 / ? | 2.0794 (1000) | 11.9% |  | not logged |
| `ablation/resnet` | finished | 7500 / ? | 2.0783 (3000) | 13.0% |  | not logged |
| `ablation/vit_L0` | finished | 7500 / ? | 2.0786 (7000) | 13.1% |  | not logged |
| `ablation/vit_L12` | finished | 7500 / ? | 2.0773 (3000) | 13.1% |  | not logged |
| `ablation/vit_L18` | finished | 7500 / ? | 2.0565 (7000) | 15.8% |  | not logged |
| `ablation/vit_L6` | finished | 7500 / ? | 2.0535 (7500) | 16.2% |  | not logged |
| `ablation/vit_last` | finished | 7500 / ? | 2.0698 (7500) | 13.8% |  | not logged |
| `collapse/cnn_fix` | finished | 6000 / ? | 2.0709 (6000) | 14.8% | 0.095 | no |
| `collapse/cnn_fix_lr1` | finished | 6000 / ? | 2.0321 (6000) | 18.7% | 0.284 | no |
| `collapse/cnn_fix_lr3e5` | finished | 6000 / ? | 2.0696 (6000) | 15.4% | 0.153 | no |
| `collapse/cnn_fix_lr5e5` | finished | 6000 / ? | 2.0445 (6000) | 17.6% | 0.188 | no |
| `collapse/cnn_fix_r01` | finished | 6000 / ? | 2.0794 (4000) | 12.8% | 0.000 | **yes** |
| `collapse/cnn_old` | finished | 6000 / ? | 2.0794 (2000) | 12.8% | 0.000 | **yes** |
| `overfit_L6` | finished | 3200 / ? | 2.0663 (1000) | 18.0% |  | not logged |
| `pipeline/lorafirst_cnn_abs24_seed0/base` | partial | 18850 / 43125 | 0.5883 (18000) | 77.8% |  | not logged |
| `pipeline/lorafirst_cnn_abs24_seed0/s0A` | finished | 230000 / 230000 | 1.5282 (170000) | 39.7% | 4.579 | no |
| `pipeline/lorafirst_cnn_abs24_seed0/s0B` | finished | 232500 / 232500 | 1.5409 (172500) | 38.1% | 3.830 | no |
| `sh/seed0/cnn` | finished | 20000 / ? | 2.0794 (2500) | 12.8% | 0.000 | **yes** |
| `sh/seed0/resnet` | finished | 20000 / ? | 2.0788 (2500) | 13.5% | 0.000 | **yes** |
| `sh/seed0/vit_L0` | partial | 8150 / ? | 2.0793 (5000) | 12.7% | 0.000 | **yes** |
| `sh/seed0/vit_L12` | partial | 7950 / ? | 2.0704 (5000) | 14.2% | 0.094 | no |
| `sh/seed0/vit_L18` | finished | 20000 / ? | 2.0667 (17500) | 15.6% | 0.124 | no |
| `sh/seed0/vit_L6` | finished | 20000 / ? | 2.0777 (2500) | 13.8% | 0.000 | **yes** |
| `sh/seed0/vit_last` | finished | 20000 / ? | 2.0695 (2500) | 15.6% | 0.000 | **yes** |
| `sh_v2/seed0/cnn` | finished | 60000 / ? | 1.8976 (55000) | 23.2% | 0.521 | no |
| `sh_v2/seed0/resnet` | finished | 60000 / ? | 1.9248 (60000) | 21.4% | 0.446 | no |
| `sh_v2/seed0/vit_L0` | finished | 20000 / ? | 1.9893 (20000) | 20.4% | 0.337 | no |
| `sh_v2/seed0/vit_L12` | finished | 60000 / ? | 1.8854 (60000) | 24.1% | 0.559 | no |
| `sh_v2/seed0/vit_L18` | finished | 60000 / 60000 | 1.8913 (60000) | 21.9% | 0.524 | no |
| `sh_v2/seed0/vit_L6` | finished | 60000 / 60000 | 1.8487 (55000) | 25.9% | 0.636 | no |
| `sh_v2/seed0/vit_last` | finished | 60000 / 60000 | 1.8328 (60000) | 27.2% | 0.657 | no |
| `sh_v2_abs12/seed0/cnn` | finished | 20000 / ? | 1.9428 (17500) | 21.1% | 0.407 | no |
| `sh_v2_abs24/seed0/cnn` | finished | 20000 / ? | 1.9171 (20000) | 23.4% | 0.420 | no |
| `sh_v2_abs24/seed0/vit_L12` | finished | 20000 / ? | 1.9250 (15000) | 20.6% | 0.430 | no |
| `sh_v2_abs6/seed0/cnn` | finished | 20000 / ? | 1.9560 (20000) | 20.8% | 0.428 | no |
