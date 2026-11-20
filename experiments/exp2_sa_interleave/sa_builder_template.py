"""TEMPLATE for a Slot Abstractor other than ours. For our run (slot_abstractor/ in this repo) use exp2/sa_builder.py.
Connects Exp2 to a pretrained Slot Abstractor. Copy it, fill in the three TODOs, then check it:

    cp exp2/sa_builder_template.py exp2/my_sa_builder.py   # edit this copy (exp2/sa_builder.py is ours)
    python exp2/exp2_sa_interleave.py inspect --list_modules --sa_builder exp2/my_sa_builder.py --sa_ckpt $SA_CKPT
    python exp2/exp2_sa_interleave.py inspect --sa_builder exp2/my_sa_builder.py --sa_ckpt $SA_CKPT --n 300

The second command scores 300 validation problems with the SA alone. It must reproduce the SA's own validation
accuracy. If it is near 12.5%, the preprocessing below does not match what
the SA was trained on (size, [-1, 1] scaling, black/white, panel order).

Contract (see sa_interface.py):  build(ckpt, device) -> model ;  encode(model, panels uint8 (N, 9, 160, 160)) ->
{"slots": (N, 9, K, D), "rel": (N, 9, K, D) or (N, R, D), "score": (N,)}
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_interface import Capture, paper_preprocess  # noqa: E402

# TODO 1: the folder with your Slot Abstractor code (the one with run_pgm_jsonl.py / slot_abstractor_pgm.py)
SA_CODE_DIR = os.environ.get("SA_CODE_DIR", "/path/to/slotabstractor/pgm")
sys.path.insert(0, SA_CODE_DIR)

# TODO 2: the submodules whose outputs are the slots and the relation embeddings
# (names from `inspect --list_modules`). Slot attention output: per panel K slots of size D.
# Abstractor output: the relation embeddings before mean pooling and the final linear scorer.
SLOT_MODULE = "slot_attention"
REL_MODULE = "abstractor"


def build(ckpt_path, device):
    # TODO 3: construct the model exactly as your training script does and load the weights, e.g.
    #   from slot_abstractor_pgm import SlotAbstractor
    #   model = SlotAbstractor(**hparams)
    #   sd = torch.load(ckpt_path, map_location="cpu"); model.load_state_dict(sd.get("model", sd))
    raise NotImplementedError("fill in build() in your copy of sa_builder_template.py")
    return model.to(device).eval()


@torch.no_grad()
def encode(model, panels):
    n = panels.shape[0]
    x = paper_preprocess(panels, size=80)          # (N, 9, 1, 80, 80) in [-1, 1]; change if your SA used other input
    with Capture(model, {"slots": SLOT_MODULE, "rel": REL_MODULE}) as cap:
        score = model(x)                           # adapt to your forward signature; must give one score per matrix
    slots, rel = cap["slots"], cap["rel"]
    # Reshape to the contract. Typical layouts:
    #   slot attention run on all N*9 panels -> (N*9, K, D)      -> view(n, 9, K, D)
    #   Abstractor over the 9*K slots        -> (N, 9*K, D)      -> view(n, 9, K, D)
    K = slots.shape[-2]
    slots = slots.reshape(n, 9, K, slots.shape[-1])
    rel = rel.reshape(n, 9, K, rel.shape[-1]) if rel.numel() == n * 9 * K * rel.shape[-1] else rel.reshape(n, -1, rel.shape[-1])
    return {"slots": slots, "rel": rel, "score": score.reshape(n)}
