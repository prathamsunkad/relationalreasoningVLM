# H15 on InternVL: Yiran's hybrid (epoch-6 LoRA + Slot Abstractor, update 3500) vs its parts, 1,000 validation puzzles

Interim (the run continues to 15,000 updates or the plateau rule). Hybrid scores: Yiran's `average-control/hybrid-reference.json`
(update 3500, minimum validation CE). LoRA alone (`ivl1b_pc_recipe_e1/final_e6`, the same adapter) and Slot Abstractor alone:
`results/internvl_sa/complement_val1000_e6/` (Exp2 `complement`). Same 1,000 puzzle IDs; exact McNemar on the paired outcomes.

| Hybrid vs | Hybrid | Other | Difference (pts) | Only hybrid right | Only other right | p |
|---|---|---|---|---|---|---|
| LoRA alone (epoch 6) | 66.8% | 60.5% | +6.3 | 117 | 54 | 1.6e-06 |
| Slot Abstractor alone | 66.8% | 60.2% | +6.6 | 128 | 62 | 1.9e-06 |
| Ensemble, equal weight (log-softmax) | 66.8% | 66.3% | +0.5 | 51 | 46 | 0.68 |
| Ensemble, cross-fitted weight [0.45, 0.5] | 66.8% | 66.2% | +0.6 | 52 | 46 | 0.61 |

Not yet available: equal-steps control (running), branch-off and token ablations, test5000 scores (scoring of the hybrid checkpoint is being validated).
