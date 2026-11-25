"""
Nov 25: drop-in replacement for pgm_jsonl_data.JsonlPGM that reads the original 16-panel PGM .npz files listed in a
text file, so the Slot Abstractor can be trained on our exact 5% puzzle list (results/splits/pgm_neutral_train_5pct_60k).

Same tensors as the original Slot Abstractor loader (train_slot_abstractor_pgm.py `dataset`): image.reshape(16, 160, 160)
-> uint8 -> ToTensor -> 2x-1 -> Resize(img_size), and the same as JsonlPGM on the contact sheets. The resize is
antialiased explicitly, as torchvision >= 0.17 does by default (the env the shared 20% Slot Abstractor was trained in;
Exp2's sa_builder uses SA_ANTIALIAS=1), so the container's older torchvision gives the same panels.

Signature matches JsonlPGM(root, split_file, layout, img_size, n=0, ...): root = the pgm_data folder, split_file = a
text file with one path per line relative to root (e.g. train_full/neutral/PGM_neutral_train_1000007.npz); layout is
ignored. Returns (16, 1, img_size, img_size) in [-1, 1] and the answer index; ids = file names.
"""
import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class NpzPGM(Dataset):
    def __init__(self, root, split_file, layout=None, img_size=80, n=0, panel_px=160, image_key="images",
                 answer_key="answer", return_index=False, offset=0):
        self.root = root
        names = [l.strip() for l in open(split_file) if l.strip() and not l.startswith("#")]
        names = names[offset:]
        if n:
            names = names[:n]
        self.paths = names
        self.ids = [os.path.basename(p) for p in names]
        self.return_index = return_index
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda X: 2 * X - 1.0),
            transforms.Resize((img_size, img_size), antialias=True),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.root, self.paths[idx]))
        image = data["image"].reshape(16, 160, 160)
        target = int(data["target"])
        img = torch.stack([self.transforms(Image.fromarray(image[i].astype(np.uint8))) for i in range(16)], dim=0)
        if self.return_index:
            return img, target, idx
        return img, target
