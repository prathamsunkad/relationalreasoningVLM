# Historical checkpoints and long-run recovery

Use these commands from the `merged-yiran` checkout with the verified VLM Python environment. Data and weights remain external. Nothing here submits another job automatically.

## Import our historical checkpoints

The converter runs offline on CPU, without loading the pretrained VLM. It supports:

- Standard `adapter/adapter_model.safetensors` or a standalone PEFT adapter directory, for all three model families.
- Original early/late/joint/memory/dual checkpoints containing `probe.safetensors`, with or without `adapter/`.
- RCA plugin/memory snapshots containing `weights.pt` and the original `manifest.json`.
- Depth/readout and insertion-point diagnostic `best.pt`/`latest.pt` files containing an `arm` state dictionary.

It verifies available checksums, recorded architecture settings and tensor keys/shapes. A mismatched correction scale is rejected when it is recorded in the source metadata. No source files are changed, and an existing destination is never overwritten. `conversion.json` records the source hashes and settings used.

Original joint late probe:

```bash
python tools/convert_vlm_checkpoint.py \
  --source /path/to/old/late/train --pointer best \
  --config configs/vlm/lfm25_late_lora_module.json \
  --output /path/to/converted/late
```

A pointer root must contain `best.json` or `latest.json` naming its generation. Alternatively, pass the checkpoint generation directory directly. Use the *actual* source architecture, depth, dimensions and correction scale; a matching state dictionary does not establish that every experimental setting matches.

A small-contribution RCA plugin snapshot:

```bash
python tools/convert_vlm_checkpoint.py \
  --source /path/to/plugin/checkpoint-UUID \
  --extension rca_plugin --correction_scale 0.01 \
  --adapter-config /path/to/original/adapter_config.json \
  --output /path/to/converted/plugin
```

For packed LoRA tensors, the original `adapter_config.json` is used when supplied; otherwise the converter requires the original recorded `plan.lora` settings and derives target-module names from the tensors. It never guesses rank, alpha or dropout. The defaults `text_dim=2048` and `vision_dim=1152` describe our native LFM2.5; the flags permit offline tiny-model tests.

Evaluate a converted joint checkpoint using its matching settings:

```bash
python lora/eval_vlm.py --model_id lfm25 --protocol answer_letter \
  --extension late --adapter /path/to/converted/late \
  --npz_dir /path/to/pgm/test --out_dir /path/to/new/evaluation
```

To start a **new optimizer stage** from its weights:

```bash
python lora/train_lora.py --config configs/vlm/lfm25_late_lora_module.json \
  --init_adapter /path/to/converted/late --init_extension /path/to/converted/late \
  --train_dir /path/to/pgm/train --val_dir /path/to/pgm/val \
  --out_dir /path/to/new/stage
```

For pure module checkpoints, supply only `--init_extension`; for plain LoRA, supply only `--init_adapter`. A packed plugin checkpoint may include frozen LoRA tensors too; supplying both preserves those inherited tensors. Use `--train_mode module` to keep them frozen, or `lora_module` to update them.

**Conversion imports weights, not historical optimizer/RNG state or training progress.** It permits evaluation and a new fine-tuning stage. Exact continuation of an old campaign still requires that campaign's original trainer. Do not label a new stage as an exact resume, or reset cumulative data exposure in the experiment report.

## Resume a run made by the shared trainer

Repeat the original training command with the same output directory. `--epochs` is the total requested epoch count, not additional epochs. Exact continuation requires the same rank count, data order, batch configuration, objective and model settings.

Each save writes an immutable generation under `.checkpoints/`: model/adapter files, optimizer/RNG state, selection information and checksums. Only after the complete generation is flushed is its pointer replaced atomically. `last/`, `best/`, `final/` and `state.pt` remain convenient paths for existing commands; pointers inside `.checkpoints/` are authoritative.

If the latest resume generation is corrupt or incomplete, startup verifies and uses the previous committed generation. Metrics beyond its step are removed and best-checkpoint selection is restored to that generation's state. Training repeats only the work after that committed step. Two recent resume generations and their selected best checkpoints are retained; unrelated and historical files are never candidates for this cleanup. A run lock prevents two allocations writing the same output concurrently. Evaluation and initialization hold a read lease so retention cannot delete a checkpoint while it is in use.

The pre-transaction shared trainer's `last/` plus `state.pt` format is also accepted, with its existing model/optimizer hash check. The first new committed save preserves those original files as `legacy-*` entries. Older logs are rewound to the restored step.

## Slurm stop behavior

The portable training launcher requests a warning three minutes before the allocation ends. A shared stop marker reaches all workers without killing the distributed launcher. The trainer also handles SIGUSR1/SIGTERM/SIGINT directly and uses the allocation's absolute end time, including loading and validation, with a default two-minute checkpoint reserve.

Stops occur at optimizer-update boundaries. Validation checks for stop requests between batches; a partial validation is never eligible for selecting `best` or declaring a plateau. Training randomness is preserved around validation, so skipping validation at a timeout does not change the next training update. SIGKILL or hardware failure can still interrupt a write; the earlier committed generation remains recoverable.

These safeguards do not increase the wall-time request, automatically requeue, or launch follow-up allocations.

## Matched initialization for diagnostic controls

For depth/readout or insertion-point comparisons, export the common initial tensors once:

```bash
python tools/init_matched_probes.py --output /path/to/initial-probes --seed 20261118
python lora/train_lora.py --config configs/vlm/lfm25_depth2_llm.json \
  --init_extension /path/to/initial-probes/llm_d2 \
  --train_dir /path/to/pgm/train --val_dir /path/to/pgm/val --out_dir /path/to/new/depth2
```

Use `llm_d6`, `classifier_d2`, `classifier_d6`, `classifier_only`, or `location_2`/`location_8`/`location_12` with the corresponding recipe for the other arms. Common layers, classifier heads and insertion-point tensors are identical; the six-layer model has four additional freshly initialized layers. This invokes the original matched-initialization helper. The shared trainer still computes native features during each forward, so its throughput should not be equated with the old cached-feature diagnostic.
