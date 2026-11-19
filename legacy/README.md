# legacy/: the original code that produced the LoRA baseline runs

`vlm_lora_pc/` is the exact code (`~/scratch/vlm_lora_pc` on PACE) that trained and test-scored every LoRA run in
`results/lora/` (`qwen_pc_recipe_e1`, `ivl1b_pc_recipe_e1`, the 5% and 1% runs, the 500-puzzle overfit runs). It
was later reorganised into `pgmvlm/` + `lora/` at the repo root, which is the version to use and extend.

Check that the two agree: a re-score made on Nov 4, when the code was moved into this repo, scored the Qwen epoch-4
adapter on 2,000 test puzzles (`results/lora/test_200k/qwen_pc_recipe_e1_neutral_repo_check`) and gave the same prediction and the same 8 scores
(to 4 decimals) on all 2,000 as the original run. Training with the repo version was not re-run end to end.
