# Conditional larger Case-A campaign

> **Status, 2026-11-23:** This campaign is stopped by user request. Early, Late and Joint completed; pending jobs `7238324_3`–`7238324_6` were cancelled. Preserve the results; do not relaunch. The new priority is [pretrained Abstractor connectors](PRETRAINED_ABSTRACTOR_CONNECTORS.md).

User-approved scope: the historical 4,096 questions, up to 30 epochs or four
H200-hours per design (28 GPU-hours total), after all seven pilot arms pass.
No larger jobs are submitted by the trainer itself.

Current pilot: job 7237733 (the isolated retry after a scalar checksum fix),
`lfm-abstractor-staged-a-2026-11-22/retry-01`. All seven designs must pass; accuracy
is not a gate for this 64-question correctness test. Failed numerical/checkpoint
checks must not be weakened to release the larger campaign. The current chat may
perform at most two further implementation-repair retries within the remaining
two-hour pilot GPU-use budget. No scheduled background automation is created.

The launch gate requires all seven statuses passed, all 16 updates and 32
validation questions completed, unchanged frozen base weights, deep gradients,
candidate permutation equivariance, exact zero-share bypass, exact checkpoint
reload and next-update replay. Every larger job independently rechecks the gate.
It also hashes architecture, native-model, image/protocol and scoring source files
against the passing pilot. A changed mechanism requires another pilot.

The selected campaign and the unselected future option are:

| Option | Train pool | Epoch limit | Allocation cap per design | Seven-design cap |
| --- | ---: | ---: | ---: | ---: |
| **Approved: smaller learning campaign** | Historical 4,096 | 30 | 1 H200 x 4h | 28 GPU-hours |
| Not authorized: fixed 20% campaign | Historical 240,000 | 1 | 1 H200 x 16h | 112 GPU-hours |

A cap may stop training before the requested epochs. Report actual unique
questions, presentations, optimizer updates and coverage; pool size is not
coverage. Both options use the same 1,000 official-validation IDs, no test data,
original frozen LFM, fresh added modules, unchanged candidate-scoring protocol,
and no LoRA. Do not continue from the tiny pilot's trained module weights.

The shared recipe is AdamW 3e-4, weight decay .01, global batch16, gradient clip1,
20-update warmup and cosine decay to .1 of the initial LR over 1,024 updates
(four 4k epochs), then hold that minimum. The pilot measured about 1.5 questions
per second, making about 3–4 epochs plausible within four hours including
validation; a 30-epoch decay would barely change the LR within this allocation. The 4k option validates each epoch and can stop after at least
three epochs plus five checks without a cumulative validation-CE improvement of
.001. Save the minimum complete validation CE even if the gain is smaller than
.001; the threshold affects stopping only. The 240k/one-pass option disables
plateau stopping and validates every 1,000 updates (16k questions). Both have
initial/final validation and measured time reserves; incomplete validation never
selects a checkpoint. All runs stop at their allocation cap with resumable state.
These stopping rules do not prove global convergence.

`tools/train_staged_abstractor.py` uses checksummed input images, a bounded decoded
image cache, deterministic per-epoch ordering, source/data contracts and atomic
module/optimizer/RNG/progress checkpoints. Save only `best.pt` and `latest.pt`,
not frozen VLM copies. `--resume` is explicit and rejects changed identity or
corrupt checkpoints. No resubmission, resource increase, cancellation or test job
is performed automatically by the trainer.

Entry points: `configs/vlm/lfm25_staged_a_training.json`,
`slurm_portable/slurm/vlm_staged_a_train.sbatch`,
`tools/train_staged_abstractor.py`; checks:
`tests/test_staged_training_gate.py`. Submission must record the final resolved
config, passing-pilot receipt, source bundle hashes and scheduler job IDs, and
must refuse duplicate submission. Jobs use the experiment branch; main is not
modified. Historical direct-answer LoRA results are not a matched control for
this candidate-scoring pipeline.
