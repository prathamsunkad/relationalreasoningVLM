# AI coding-agent usage

## Summary

We brainstormed the research ideas, hypotheses and experiment designs ourselves as a team, then prompted coding agents
(**Claude Code** and **OpenAI Codex**) for the implementation part: writing and refactoring code, Slurm job scripts,
data-preparation tools, analysis scripts and documentation, and submitting and monitoring jobs on PACE ICE.
The agents did not choose the research questions. Every result reported in the repo comes from jobs run on PACE,
and we checked agent-written code and numbers as described below before relying on them.

## Tools

| Tool | Used by | Purpose |
|---|---|---|
| Claude Code (Anthropic; Claude Opus 5.5, Claude Fable 5.2) | Pratham Sunkad, Kun-Lin Hsieh, Sai Pranav Komaragiri | Implementation, refactoring, Slurm jobs and job chains, evaluation and analysis scripts, documentation, commits |
| OpenAI Codex | Yiran Luo | Integrating the two codebases into shared training/evaluation infrastructure, porting modules, tests, documentation |
| Local CodeGraph index | Yiran Luo | Code navigation during the Codex integration (not committed) |

## What each agent materially affected

### Claude Code: main branch (Pratham Sunkad)

Ten of the eleven commits on `main` carry a `Co-Authored-By: Claude Opus 5.5` trailer (all except `201b1a7`).

| Area | Files / experiments |
|---|---|
| Repository assembly | `8bf3101`: import of the existing code, unchanged; `87fa3d4`: `pgmvlm/` package, import and path fixes, `env.sh`, pinned `requirements.txt`, `results/zeroshot/zeroshot_summary.csv` |
| LoRA baseline results and docs | `README.md`, `docs/experiments.md`, curves and test results in `results/lora/` (`50ff1c1`, `9c8a659`) |
| Adapter-sharing bundle | `share/` (`make_bundle.sh`, `run.sh`, `setup`/`check`/`eval_test`/`record_expected.sbatch`, `check.py`, `BUNDLE_README.md`) (`650cefc`) |
| Relational branch | `relational/` code and jobs, `experiments/` (H7–H13 launchers and chains in `experiments/auto/`, scoring jobs and analyzers for A1, A2 and the branch-off check), `results/relational/`, `00_START_HERE/` guides (`ab4fd0c`) |
| Data and checkpoints | `00_START_HERE/DATA.md`, `CONDITIONS_AND_ASSUMPTIONS.md`, `CHECKPOINTS.md`, exact splits in `results/splits/`, `data_manifests/`, `slurm_portable/` path-free copies (`3a008a3`, `08de3ca`) |
| A3 relation-token ablation | `experiments/diag_ablate/` and its results (`e256309`) |
| Running experiments | Jobs were submitted and chained by the agent on PACE, e.g. `experiments/auto/auto_stage2.py` and the job log `results/relational/job_log/night_queue_2026-11-08.txt`, which records which jobs Claude submitted by hand |

### Codex: `merged-yiran` branch (Yiran Luo)

From `docs/VLM_AGENT_USAGE.md` on that branch: Codex inspected both local repositories, refactored the shared
training/evaluation interfaces, transferred the LFM architectural modules and diagnostic kernels, prepared portable
recipes and documentation, and ran offline regression checks. Yiran approved the scope and asked for shared
infrastructure rather than a duplicate pipeline.

| Area | Files / experiments |
|---|---|
| Shared LoRA training/evaluation for candidate-scoring and answer-letter protocols | commits `f96743a`–`aafaa93`; `tests/` (protocol, distributed, checkpoint-recovery, resume and native-family tests) |
| LFM probes, RCA plugin, relational memory modules | `08a296a` |
| Portable recipes, subset lists, docs, historical-result provenance | `5ccc573`, `ba49c64`, `f4798db`; `docs/integration_source.json` |
| Checkpoint import and matched initialization | `1623f3e`, `667feb9`; `tools/convert_vlm_checkpoint.py`, `tools/init_matched_probes.py` |
| Ablations and input-representation pilot | `5466e6a`, `8ed2123`, `d9822b5`; `tools/input_representation_pilot.py` |
| GPU smoke check | `0d30c4d`; `tools/gpu_smoke.py` |

Results under `results/vlm/` predate this integration and are not presented as newly reproduced scores. No new PACE
jobs, data uploads or weight publications were made as part of it.

### Claude Code: `kunlin` branch (Kun-Lin Hsieh)

The I-RAVEN generalization test of the PGM-trained `lfm25-pgm-best` adapter (`iraven_eval/`, commit `9981420`).
Kun-Lin set the goal: evaluate the PGM-trained adapters on the I-RAVEN test set. Claude Code then:

- generated I-RAVEN with the authors' unmodified generator (`jobs/gen_iraven.sbatch`, `scripts/pack_test.py`);
- set up the pinned environment (`jobs/setup_env.sbatch`, `scripts/download_base.py`);
- rendered the puzzles as PGM-style contact sheets (`scripts/build_sheets.py`, `template/`);
- wrote the evaluation (`jobs/eval.sbatch`, `scripts/eval_adapter.py`);
- submitted and monitored the Slurm jobs, and wrote `iraven_eval/README.md`, the results summaries and the commit.

### Claude Code: `pranav` branch (Sai Pranav Komaragiri)

The slot abstractor code was just cloned from the paper's repo and run on reduced pretraining data to match to the lower data. All the probes, experiments and the encoder feature extraction ablation were our ideas that were described to Claude to create bulk of the code.

## How the output was validated

| Work | Validation |
|---|---|
| Baseline and relational code | CPU tests on tiny random models (`tests/test_pipeline.py`, `relational/tests/test_tiny.py`), run through Slurm (`slurm_portable/.../tests.sbatch`) |
| Repository refactor | Code imported unchanged first (`8bf3101`), then adapted in a separate commit, so the changes can be reviewed on their own; the adapted repo code re-scored the Qwen LoRA adapter on 2,000 test puzzles and gave the same prediction and the same 8 scores as the original code (`results/lora/test_200k/qwen_pc_recipe_e1_neutral_repo_check/`, described in `legacy/README.md`) |
| Adapter bundle | `sha256sum -c SHA256SUMS`; `check.sbatch` scores 10 sample problems against recorded reference scores and must end with "ALL CHECKS PASSED" |
| Analyses (A1, A2) | Reports rebuilt from the committed per-puzzle records; the rebuilt A2 report matches the committed one line for line (checked Nov 19) |
| Statistical claims | McNemar paired tests and Wilson intervals; GPU-type noise (~0.5% of answers flip across GPU types) is documented in `CONDITIONS_AND_ASSUMPTIONS.md` and taken into account when comparing runs |
| Codex integration | `docs/integration_validation.json`: 22 original LoRA pipeline tests, 34 original slot-pipeline tests and 46 integration tests passed on CPU with tiny models, with log hashes. Items not verified (full pretrained-model accuracy through the refactored runner, production CUDA stack, NCCL/H200 throughput, full 240k image parity) are listed there explicitly |
| I-RAVEN evaluation | The generator's own solver check scored 1.0 on all 7 configurations; 2,000 test puzzles per configuration with balanced answer positions. The contact-sheet renderer reproduces real PGM sheets pixel for pixel. The adapter is loaded through the bundle's own loader, which checks its sha256. Pipeline check: 77.2% on the first 1,000 PGM test questions vs the published 75.5% for this adapter. Rendered sheets were inspected by eye, and 0 of 15,000 outputs were invalid |
| Human review | Each team member reviewed their agent's output and decided which results to report. *Each member: confirm or adjust this row for your own work.* |
