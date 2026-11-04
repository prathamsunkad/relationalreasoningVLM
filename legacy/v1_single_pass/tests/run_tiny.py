"""Test-only launcher: runs train_lora.py / eval_vlm.py with the tiny random models patched in for the HF loaders.
usage: run_tiny.py --model_id tiny_<family> [common flags] train|eval [script flags]"""
import os
import runpy
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
import torch

import model_utils
import tiny_models as T

FAM = os.environ["TINY_FAMILY"]


def fake_load_base(model_id, dtype="bf16", attn="sdpa", device=None, family=None):
    model, proc = (T.tiny_qwen3vl if FAM == "qwen3vl" else T.tiny_internvl)(seed=0)
    proc.tokenizer.padding_side = "left"
    return (model.to(device) if device is not None else model), proc, FAM


model_utils.load_base = fake_load_base
argv = sys.argv[1:]
# split: everything before the sub-command are shared flags
cmd_i = next(i for i, a in enumerate(argv) if a in ("train", "eval", "preflight"))
shared, cmd, rest = argv[:cmd_i], argv[cmd_i], argv[cmd_i + 1:]
sys.argv = [cmd] + rest + shared
for mod in ("train_lora", "eval_vlm", "preflight"):
    sys.modules.pop(mod, None)
script = {"train": "train_lora.py", "eval": "eval_vlm.py", "preflight": "tools/preflight.py"}[cmd]
# the scripts did `from model_utils import load_base`, so import them *after* patching
runpy.run_path(os.path.join(ROOT, script), run_name="__main__")
