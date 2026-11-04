"""SVRT (Fleuret et al. 2011; 23 binary problems) -> packed jsonl/zip, generated with Fleuret's own C++ generator.

The generator (https://fleuret.org/git/pysvrt, GPL-3) is cloned into <out>/_src and built as libsvrt.so; we call
svrt_generate_vignettes() through ctypes (pysvrt's own Python binding needs the long-gone torch.utils.ffi).
128x128 grayscale vignettes, classes 0/1 balanced. OCRA / Slot Abstractors trained one model per problem with 500 or
1000 examples; here every problem is one source of binary Yes/No examples, the problem number is in the prompt (SVRT
rules are hidden by design; the model has to learn each problem's rule from its examples). Panel = the whole image.

  python datasets/make_svrt.py [--train 1000 --val 200 --test 2000]
"""
import argparse
import ctypes
import os
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT, Packer

# Standard grouping of the 23 problems (Kim et al. 2018; Vaishnav et al. 2022): same-different vs spatial-relation
SAME_DIFF = {1, 5, 6, 7, 13, 16, 17, 19, 20, 21, 22}
PROMPT = ("This image is from SVRT problem {p}. Images of this problem belong to one of two classes, defined by a hidden "
          "rule about the shapes in the image. Is this image in class 1? Answer Yes or No.")


class VignetteSet(ctypes.Structure):
    _fields_ = [("n_problem", ctypes.c_int), ("nb_vignettes", ctypes.c_int), ("width", ctypes.c_int),
                ("height", ctypes.c_int), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def build_lib(src):
    if not os.path.isdir(src):
        subprocess.check_call(["git", "clone", "-q", "https://fleuret.org/git/pysvrt", src])
    lib = os.path.join(src, "libsvrt.so")
    if not os.path.exists(lib):
        subprocess.check_call(["make", "-C", src, "libsvrt.so", "LDFLAGS=-static-libstdc++ -static-libgcc"])
    so = ctypes.CDLL(lib)
    so.svrt_generate_vignettes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_long),
                                          ctypes.POINTER(VignetteSet)]
    return so


def generate(so, problem, labels):
    lab = np.ascontiguousarray(labels, dtype=np.int64)
    vs = VignetteSet()
    so.svrt_generate_vignettes(problem, len(lab), lab.ctypes.data_as(ctypes.POINTER(ctypes.c_long)), ctypes.byref(vs))
    n, h, w = vs.nb_vignettes, vs.height, vs.width
    arr = np.ctypeslib.as_array(vs.data, shape=(n * h * w,)).reshape(n, h, w).copy()
    ctypes.CDLL(None).free(ctypes.cast(vs.data, ctypes.c_void_p))
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "svrt"))
    ap.add_argument("--train", type=int, default=1000, help="per problem (the paper used 500 or 1000)")
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--test", type=int, default=2000, help="per problem (OCRA used 40,000)")
    ap.add_argument("--scale", type=int, default=2, help="upsample 128 px vignettes (nearest) for more ViT patches")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    so = build_lib(os.path.join(args.out, "_src"))
    libc = ctypes.CDLL(None)
    for split, n, seed_off in (("train", args.train, 0), ("val", args.val, 1000), ("test", args.test, 2000)):
        pk = Packer(args.out, split)
        for p in range(1, 24):
            libc.srand48(ctypes.c_long(args.seed * 100000 + seed_off + p))    # distinct stream per (split, problem)
            labels = np.random.default_rng([args.seed, seed_off, p]).permutation(np.arange(n) % 2)
            imgs = generate(so, p, labels)
            for i, (im, y) in enumerate(zip(imgs, labels)):
                pil = Image.fromarray(im)
                if args.scale > 1:
                    pil = pil.resize((im.shape[1] * args.scale, im.shape[0] * args.scale), Image.NEAREST)
                ref = pk.add_image(f"p{p:02d}/{i:05d}.png", pil)
                pk.add_row({"id": f"svrt/{split}/p{p}/{i}", "images": [ref], "prompt": PROMPT.format(p=p),
                            "answer": "Yes" if y == 1 else "No", "problem": p,
                            "group": "same_different" if p in SAME_DIFF else "spatial"})
        pk.close({"per_problem": n, "problems": 23, "scale": args.scale})


if __name__ == "__main__":
    main()
