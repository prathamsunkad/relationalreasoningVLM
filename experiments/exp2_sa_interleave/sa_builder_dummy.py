"""A random-weight stand-in with the Slot Abstractor's shapes (for the CPU tests only; it knows nothing).
CNN per panel -> K slots (learned queries, one attention step) -> transformer over the 9*K slots -> mean -> score."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sa_interface import Capture, paper_preprocess  # noqa: E402

K, D = 4, 16


class DummySA(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, D, 5, 4, 2), nn.ReLU(), nn.Conv2d(D, D, 5, 4, 2))
        self.q = nn.Parameter(torch.randn(K, D))
        self.slot_attention = nn.Identity()
        self.abstractor = nn.TransformerEncoder(nn.TransformerEncoderLayer(D, 2, 32, batch_first=True), 1)
        self.head = nn.Linear(D, 1)

    def forward(self, x):                                   # (N, 9, 1, 80, 80)
        n = x.shape[0]
        f = self.cnn(x.flatten(0, 1)).flatten(2).transpose(1, 2)               # (N*9, 25, D)
        a = torch.softmax(self.q @ f.transpose(1, 2), -1)                       # (N*9, K, 25)
        z = self.slot_attention(a @ f)                                          # (N*9, K, D)
        s = self.abstractor(z.reshape(n, 9 * K, D))                             # (N, 9K, D)
        return self.head(s.mean(1)).squeeze(-1)


def build(ckpt_path, device):
    torch.manual_seed(0)
    m = DummySA()
    if ckpt_path and os.path.exists(ckpt_path):
        m.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return m.to(device).eval()


@torch.no_grad()
def encode(model, panels):
    n = panels.shape[0]
    with Capture(model, {"slots": "slot_attention", "rel": "abstractor"}) as cap:
        score = model(paper_preprocess(panels))
    return {"slots": cap["slots"].reshape(n, 9, K, D), "rel": cap["rel"].reshape(n, 9, K, D), "score": score}
