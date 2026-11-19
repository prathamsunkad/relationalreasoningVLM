# Shared VLM recipes

All these recipes use the same `lora/train_lora.py` engine. Inspect inherited settings with:

```bash
python tools/show_vlm_recipe.py configs/vlm/lfm25_4k_rca_plugin.json
```

- `*_pgm_lora`: three model families, fixed 240k/1k splits, one epoch by default.
- `*_clevr_lora`: provide separate training/validation manifests for RMTS or identity.
- `lfm25_{early,late,joint,memory,dual}_{module,lora_module}`: original five modifications.
- `lfm25_rca_{plugin,memory}_{module,lora_module}`: the newer alternatives.
- `lfm25_4k_*`: exact 4,096-question subset, up to 30 epochs, validation stopping.
- `lfm25_depth*`, `lfm25_classifier_only`: depth and readout diagnostics.
- `lfm25_correction_cap_*`: relative correction bounds.
- `lfm25_location_*`: visual-token correction after a selected decoder block.

Data and output directories are supplied on the command line. These are executable research recipes, including unsuccessful architectures. They do not imply the historical campaigns used identical initialization or stopping policies; see the detailed [integration guide](../../docs/VLM_INTEGRATION.md). `module` freezes the VLM completely; `lora_module` updates LoRA and new module weights, with the original VLM weights still frozen.
