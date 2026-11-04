"""Backbone adapters: where the relational branch attaches to Qwen3-VL and InternVL3.5-HF (transformers 5.17.0).

Everything attaches through forward hooks, so the native VLM path runs exactly as in the LoRA baseline:
  vision  (forward hook)     : pre-merge ViT patch tokens of every image  -> slot attention (relational branch input)
  merged  (forward hook)     : the visual tokens the LLM receives          -> lambda-fusion with relational features
  lm      (forward pre-hook) : inputs_embeds entering the language model   -> relation tokens written at <|rel|> slots

Verified against the transformers 5.17.0 source:
  Qwen3-VL   model.model.visual(pixel_values, grid_thw) -> .last_hidden_state = pre-merger tokens (sum h*w, 1024) in
             merge-block order (block_row, block_col, in_row, in_col); .pooler_output = merged tokens (sum h*w/4, 2048),
             one per 2x2 block, raster over blocks. DeepStack features (ViT layers 5/11/17) are left untouched.
  InternVL   model.model.vision_tower(pixel_values) -> .last_hidden_state (n_tiles, 1 + 32*32, 1024), CLS first
             (vision_feature_layer -1, select_strategy "default" drops it); pixel_shuffle(0.5) + multi_modal_projector
             -> (n_tiles, 16*16, 1024), raster order. We require ONE tile per image (crop_to_patches=0).
"""
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
MODEL_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")
REL_TOKEN = "<|rel|>"


def detect_family(model_id):
    mt = AutoConfig.from_pretrained(model_id).model_type
    if mt == "qwen3_vl":
        return "qwen3vl"
    if mt == "internvl":
        return "internvl"
    raise SystemExit(f"model_type {mt!r} is not supported (use Qwen/Qwen3-VL-2B-Instruct or OpenGVLab/InternVL3_5-1B-HF)")


def load_vlm(model_id, dtype="bf16", attn="sdpa", device=None, family=None):
    family = family or detect_family(model_id)
    kw = {} if attn == "default" else {"attn_implementation": attn}
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=DTYPES[dtype], **kw)
    processor = AutoProcessor.from_pretrained(model_id)
    if device is not None:
        model = model.to(device)
    return model, processor, family


def window_to_raster(tokens, h, w, m):
    """(h*w, C) Qwen merge-block order -> (h, w, C) raster."""
    C = tokens.shape[-1]
    return tokens.reshape(h // m, w // m, m, m, C).permute(0, 2, 1, 3, 4).reshape(h, w, C)


class Backbone:
    """Family-specific glue. `model` is the HF *ForConditionalGeneration; it stays frozen except for optional LoRA."""

    def __init__(self, model, processor, family):
        self.model, self.processor, self.family = model, processor, family
        self.core = model.model
        self.lm = self.core.language_model
        self.d_llm = int(model.config.text_config.hidden_size)
        self.vit_dim = int(model.config.vision_config.hidden_size)
        if family == "qwen3vl":
            self.vision = self.core.visual
            self.merged_module = self.core.visual          # same output object carries the merged tokens
            self.merge = int(model.config.vision_config.spatial_merge_size)
        elif family == "internvl":
            self.vision = self.core.vision_tower
            self.merged_module = self.core.multi_modal_projector
            self.down = float(model.config.downsample_ratio)
            ps = model.config.vision_config.patch_size
            self.patch = int(ps[0] if isinstance(ps, (list, tuple)) else ps)
        else:
            raise ValueError(family)
        tok = processor.tokenizer
        tok.padding_side = "left"
        if REL_TOKEN not in tok.get_vocab():
            tok.add_tokens([REL_TOKEN], special_tokens=True)
        self.rel_id = tok.convert_tokens_to_ids(REL_TOKEN)
        n_emb = model.get_input_embeddings().weight.shape[0]
        if self.rel_id >= n_emb:   # real checkpoints have spare (padded) rows; tiny test models do not
            model.resize_token_embeddings(len(tok))
        self.image_token_id = int(model.config.image_token_id)

    # ------------------------------------------------------------------ inputs
    def model_inputs(self, batch, device, dtype):
        out = {}
        for k in MODEL_KEYS:
            if k in batch and batch[k] is not None:
                v = batch[k].to(device, non_blocking=True)
                out[k] = v.to(dtype) if k == "pixel_values" else v
        return out

    def processor_kwargs(self):
        # InternVL's processor tiles images unless told otherwise; the branch needs one 448px tile per image.
        return {"crop_to_patches": False} if self.family == "internvl" else {}

    def grids(self, inputs, n_images):
        """Per image: (patch grid (h, w), merged grid (hm, wm))."""
        if self.family == "qwen3vl":
            g = inputs["image_grid_thw"].tolist()
            assert len(g) == n_images, f"{len(g)} grids for {n_images} images"
            return [((h, w), (h // self.merge, w // self.merge)) for _, h, w in g]
        n_tiles, _, H, W = inputs["pixel_values"].shape
        assert n_tiles == n_images, (f"InternVL produced {n_tiles} tiles for {n_images} images: the relational branch "
                                     f"needs one tile per image (crop_to_patches=False)")
        h, w = H // self.patch, W // self.patch
        return [((h, w), (int(h * self.down), int(w * self.down)))] * n_images

    def run_vision(self, inputs):
        """Vision tower only (fires the vision hook). Used by Stage 0 and the LLM-free aux objective."""
        pv = inputs["pixel_values"]
        if self.family == "qwen3vl":
            return self.vision(pv.to(self.vision.dtype), grid_thw=inputs["image_grid_thw"], return_dict=True)
        return self.vision(pixel_values=pv.to(self.core.dtype), return_dict=True)

    # ------------------------------------------------------------------ hook payloads
    def feature_maps(self, vis_out, grids):
        """Vision hook output -> list of (h, w, C) raster patch maps, one per image."""
        x = vis_out.last_hidden_state
        if self.family == "qwen3vl":
            sizes = [h * w for (h, w), _ in grids]
            return [window_to_raster(t, h, w, self.merge) for t, ((h, w), _) in zip(torch.split(x, sizes), grids)]
        maps = []
        for i, ((h, w), _) in enumerate(grids):
            t = x[i]
            if t.shape[0] == h * w + 1:
                t = t[1:]                                  # CLS
            maps.append(t.reshape(h, w, -1))
        return maps

    def merged_tokens(self, out, grids):
        """Merged-token hook output -> list of (hm*wm, d_llm) per image."""
        if self.family == "qwen3vl":
            return list(torch.split(out.pooler_output, [hm * wm for _, (hm, wm) in grids]))
        return list(out.unbind(0))

    def replace_merged(self, out, new_list):
        if self.family == "qwen3vl":
            out.pooler_output = torch.cat(new_list, 0)
            return out
        return torch.stack(new_list, 0)
