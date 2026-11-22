# Direct answer versus candidate scoring with a frozen LFM

This is the missing full-VLM protocol comparison. The earlier input probe only
compared frozen visual representations with a direct classifier; it did not
compare native letter generation against Yes/No scoring.

Two fresh, identical late/post-projector Abstractors share one frozen pretrained
LFM2.5: two layers, width 256, four heads, 128 pooled symbols, zero output
initialization. There is no LoRA, pretrained task checkpoint, correction cap,
learned slot encoder, or additional classifier.

| Arm | Input and prompt | Training objective | Final prediction |
| --- | --- | --- | --- |
| `answer_letter` | Original 700x1016 contact sheet; exact `PGM_PROMPT` in `pgmvlm/manifests.py` | Full-vocabulary causal CE on the correct letter plus native ending tokens; prompt masked | Native greedy generation, max 16 tokens; invalid output is incorrect |
| `candidate` | Eight 520x520 completed matrices, original 160px panels and 10px white gaps; exact `PROMPT_TEXT` in `pgmvlm/data.py` | Shared module and model produce FP32 logit(Yes)-logit(No); one CE across eight scores | Highest-scoring candidate |

The second arm reconstructs panels losslessly from the checksummed original
contact sheet. Candidate rendering does not depend on the correct answer.
Native image processors and chat templates are preserved. Each candidate is
processed serially, retaining native LFM spatial metadata and avoiding padded
multi-image hooks. Gradients from all eight scores jointly update the module.

Reuse the exact earlier 4,096 train questions and 1,000 validation questions.
AdamW: lr 1e-4, weight decay .01, clip 1, 20-update warmup, global batch 16.
Both arms receive the identical next batch before either advances again, rotating
which runs first. Initialization seed is 20261118; data-order seed 20261112.
They share exposure, not compute: scoring needs eight forwards per question.

One H200 for at most two hours, including correctness checks and evaluation.
Stop at ten epochs or the wall-time cutoff, whichever occurs first; completing
a full epoch is not guaranteed. Initial 64-question validation timing determines
a conservative final-evaluation reserve (at least 20 minutes). Screens use the
same 64 validation IDs every 64 updates and are monitoring only. The final common
update is evaluated on all 1,000 validation questions with native predictions.
The final train monitor uses at most 128 actually exposed IDs from the historical
512-question monitor. Partial evaluations/epochs are explicitly labeled. No test
questions are read, and no follow-up jobs are submitted automatically.

Correctness checks include exact zero-module native predictions, gradients into
the new module after opening zero writeback, candidate-order equivariance, and
unchanged frozen VLM weights. CPU tests verify pixels, ordering, scoring precision,
and joint CE gradients. Best/latest experiments elsewhere are untouched; this
pilot writes only its two latest module/optimizer checkpoints and small logs.

Interpretation: a scoring advantage supports the protocol as a contributor to
failure. It does not separate composition, prompt, objective, or extra compute.
Failure of both within this short budget is not proof of architectural inability.
Loss values have different meanings; compare accuracy at matching exposure, and
report answer-token loss separately from direct-answer suffix loss.

Entry points: `tools/protocol_comparison_pilot.py`,
`configs/vlm/lfm25_protocol_probe.json`, and
`slurm_portable/slurm/vlm_protocol_probe.sbatch`. Fill in dataset/selection paths
in the JSON. Set `VLM_REPO`, `VLM_PYTHON`, `VLM_PROTOCOL_PLAN`, and
`VLM_PROTOCOL_OUTPUT` before submitting the wrapper inside the authorized budget.
Source/plan hashes and the exact scheduler command are saved in campaign receipts.
