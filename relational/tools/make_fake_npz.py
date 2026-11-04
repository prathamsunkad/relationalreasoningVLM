"""Random PGM-shaped problems (real file layout) for smoke tests:  python tools/make_fake_npz.py OUT_DIR [N]"""
import os
import sys

import numpy as np

out, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 32
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(0)
for i in range(n):
    img = np.full((16, 160, 160), 255, np.uint8)
    for p in range(16):                                  # a few dark squares per panel
        for _ in range(rng.integers(1, 5)):
            y, x, s = rng.integers(10, 120), rng.integers(10, 120), rng.integers(10, 30)
            img[p, y:y + s, x:x + s] = rng.integers(0, 200)
    t = int(rng.integers(0, 8))
    img[8 + t] = img[2]
    np.savez(os.path.join(out, f"fake{i:04d}.npz"), image=img.reshape(160, 160, 16), target=np.int64(t))
