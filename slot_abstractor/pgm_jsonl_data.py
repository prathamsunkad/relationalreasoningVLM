"""
PGM dataset for Slot Abstractor that reads the shared JSONL + contact-sheet package
(pgm-neutral-train20pct-seed20261007-v1) instead of the original 16-panel NPZ files.

Each sheet is cropped with layout.json (see infer_layout.py) into the 8 context panels
(row-major, skipping the missing cell) followed by the 8 choices A-H, each resized to
160x160 like an original PGM panel. The per-panel transform is the same one used in
train_slot_abstractor_pgm.py: ToTensor -> 2x-1 -> Resize(img_size).
Returns (16, 1, img_size, img_size) in [-1, 1] and the answer index, exactly like the
original `dataset` class.
"""
import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

LETTERS = "ABCDEFGH"


def read_rows(path, n=0, offset=0):
    rows, seen = [], 0
    with open(path) as fh:
        for line in fh:
            if line.strip():
                seen += 1
                if seen <= offset:
                    continue
                rows.append(json.loads(line))
                if n and len(rows) >= n:
                    break
    return rows


def image_path(row, key):
    p = row[key]
    return p[0] if isinstance(p, list) else p


def load_layout(path, sheet_wh):
    """Pixel boxes (x0, y0, x1, y1): 8 context boxes then 8 choice boxes."""
    j = json.load(open(path))
    w, h = sheet_wh
    matrix, choices = j["matrix"], j["choices"]
    if len(matrix) != 9 or len(choices) != 8:
        raise ValueError("layout needs 9 matrix boxes and 8 choice boxes")
    missing = int(j.get("missing", 8))
    boxes = [b for i, b in enumerate(matrix) if i != missing] + list(choices)
    if max(max(b) for b in boxes) <= 1.0:                      # normalised -> pixels
        boxes = [[b[0] * w, b[1] * h, b[2] * w, b[3] * h] for b in boxes]
    if "sheet_size" in j and tuple(j["sheet_size"]) != (w, h):
        raise ValueError(f"layout made for {j['sheet_size']} but sheets are {w}x{h}")
    return [tuple(int(round(v)) for v in b) for b in boxes]


class JsonlPGM(Dataset):
    def __init__(self, root, split_file, layout, img_size, n=0, panel_px=160,
                 image_key="images", answer_key="answer", return_index=False, offset=0):
        self.root = root
        rows = read_rows(os.path.join(root, split_file), n, offset)
        self.paths = [image_path(r, image_key) for r in rows]
        self.targets = [LETTERS.index(str(r[answer_key]).strip()[0]) for r in rows]
        self.ids = [str(r.get("id", offset + i)) for i, r in enumerate(rows)]
        self.sheet_wh = Image.open(os.path.join(root, self.paths[0])).size
        self.boxes = load_layout(layout, self.sheet_wh)
        self.panel_px = panel_px
        self.return_index = return_index
        self.transforms = transforms.Compose([                 # as in train_slot_abstractor_pgm.py
            transforms.ToTensor(),
            transforms.Lambda(lambda X: 2 * X - 1.0),
            transforms.Resize((img_size, img_size)),
        ])

    def __len__(self):
        return len(self.paths)

    def panels(self, idx):
        sheet = Image.open(os.path.join(self.root, self.paths[idx])).convert("RGB").convert("L")
        if sheet.size != self.sheet_wh:
            raise ValueError(f"{self.paths[idx]} is {sheet.size}, expected {self.sheet_wh}")
        return [sheet.crop(b).resize((self.panel_px, self.panel_px), Image.BICUBIC) for b in self.boxes]

    def __getitem__(self, idx):
        img = torch.stack([self.transforms(p) for p in self.panels(idx)], dim=0)
        if self.return_index:
            return img, self.targets[idx], idx
        return img, self.targets[idx]


if __name__ == "__main__":
    # Check: python pgm_jsonl_data.py --root $DATA --layout layout.json  -> pgm_check.png
    import argparse
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layout", required=True)
    ap.add_argument("--split_file", default="validation.jsonl")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="pgm_check.png")
    a = ap.parse_args()
    ds = JsonlPGM(a.root, a.split_file, a.layout, 80, n=a.index + 1)
    img, t = ds[a.index]
    print(f"{ds.paths[a.index]}: sheet {ds.sheet_wh}, tensor {tuple(img.shape)}, "
          f"range [{img.min():.2f}, {img.max():.2f}], answer {LETTERS[t]}")
    x = ((img[:, 0].numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)       # (16, 80, 80)
    gap = np.full((80, 4), 128, np.uint8)
    row = lambda ps: np.concatenate(sum([[p, gap] for p in ps], [])[:-1], 1)
    ctx = [row(list(x[0:3])), row(list(x[3:6])), row(list(x[6:8]) + [np.full((80, 80), 128, np.uint8)])]
    ctx = [np.pad(c, ((0, 4), (0, row(list(x[8:12])).shape[1] - c.shape[1])), constant_values=128) for c in ctx]
    sheet = np.concatenate(ctx + [np.full((12, ctx[0].shape[1]), 255, np.uint8),
                                  row(list(x[8:12])), gap[:4, :1].repeat(ctx[0].shape[1], 1), row(list(x[12:16]))])
    Image.fromarray(sheet).save(a.out)
    print(f"saved {a.out}: 3x3 context (gray = missing) above choices A-D / E-H, as the model sees them (80x80)")
