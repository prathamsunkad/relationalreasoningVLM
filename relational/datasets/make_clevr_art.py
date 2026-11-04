"""CLEVR-ART TRAIN split (Webb et al. 2023, OCRA) rendered with Blender 2.79b, exactly like the OOD test package.

The OOD test package (clevr_art_data/...) records its method: Blender 2.79b, OCRA commit f15fa62 render scripts,
CLEVR commit f0ce2c8 assets/utils, CPU, 64 samples, 320x240. The symbolic TRAIN problems ship with OCRA
(RMTS_ood_train.npz: 10,000 x 6 objects; identity_rules_ood_train.npz: 10,000 x 9 objects; train colours/shapes only),
so only rendering is needed: RMTS 2 scenes per problem, identity rules 4 scenes per problem (60k renders).

  python datasets/make_clevr_art.py setup                              # Blender + OCRA scripts/assets (login/CPU)
  sbatch slurm/clevr_art_render.sbatch                                  # array: render chunks -> _chunks/*.zip
                                                                        # (host python3: the container lacks libGLU)
  python datasets/make_clevr_art.py pack                               # chunks -> {rmts,idrules}_{train,val}.jsonl/zip
Output rows are per-candidate (cand), one rendered scene per candidate, prompts identical to make_clevr_art_ood.py.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT

OUT = os.path.join(ROOT, "clevr_art")
BLENDER_URL = "https://download.blender.org/release/Blender2.79/blender-2.79b-linux-glibc219-x86_64.tar.bz2"
OCRA = "https://raw.githubusercontent.com/Shanka123/OCRA/f15fa625d7e5b4458d82c24108b26419037d940d/{}"
CLEVR_UTILS = ("https://raw.githubusercontent.com/facebookresearch/clevr-dataset-gen/"
               "f0ce2c81750bfae09b5bf94d009f42e055f2cb3a/image_generation/utils.py")
OCRA_FILES = ["render_images_rmts.py", "render_images_idrules.py", "RMTS_ood_train.npz", "identity_rules_ood_train.npz",
              "data/base_scene.blend", "data/properties.json", "data/materials/MyMetal.blend",
              "data/materials/Rubber.blend", "data/shapes/SmoothCube_v2.blend", "data/shapes/SmoothCylinder.blend",
              "data/shapes/Sphere.blend"]
TASKS = {"rmts": ("render_images_rmts.py", "RMTS_ood_train.npz", 2),
         "idrules": ("render_images_idrules.py", "identity_rules_ood_train.npz", 4)}


def setup():
    src, tools = os.path.join(OUT, "_src"), os.path.join(OUT, "_tools")
    os.makedirs(src, exist_ok=True)
    os.makedirs(tools, exist_ok=True)
    for f in OCRA_FILES:
        dst = os.path.join(src, f)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            urllib.request.urlretrieve(OCRA.format(f), dst)
    if not os.path.exists(os.path.join(src, "utils.py")):
        urllib.request.urlretrieve(CLEVR_UTILS, os.path.join(src, "utils.py"))
    for task, (script, npz, _) in TASKS.items():                 # render a chunk: [start_idx, start_idx + num_images)
        s = open(os.path.join(src, script)).read()
        s = re.sub(r'np\.load\("[^"]+"\)', f'np.load("{npz}")', s, count=1)
        s = s.replace("for num_ex in range(10000):",
                      "for num_ex in range(args.start_idx, min(args.start_idx + args.num_images, len(prob_arr))):")
        assert "range(args.start_idx" in s, f"could not patch the problem loop of {script}"
        open(os.path.join(src, f"chunk_{script}"), "w").write(s)
    blender = os.path.join(tools, "blender-2.79b-linux-glibc219-x86_64", "blender")
    if not os.path.exists(blender):
        arc = os.path.join(tools, "blender.tar.bz2")
        urllib.request.urlretrieve(BLENDER_URL, arc)
        with tarfile.open(arc) as t:
            t.extractall(tools)
        os.remove(arc)
    print(f"[clevr_art] setup done: {src} {blender}")


def render(task, start, count, threads):
    script, npz, n_c = TASKS[task]
    src = os.path.join(OUT, "_src")
    blender = os.path.join(OUT, "_tools", "blender-2.79b-linux-glibc219-x86_64", "blender")
    chunk = os.path.join(OUT, "_chunks", f"{task}_train_{start:05d}.zip")
    if os.path.exists(chunk):
        print(f"[clevr_art] {chunk} exists, skipping")
        return
    pth = glob.glob(os.path.join(os.path.dirname(blender), "2.79", "python", "lib", "python3*", "site-packages"))[0]
    with open(os.path.join(pth, "clevr_art.pth"), "w") as f:   # the CLEVR way to make `import utils` work in Blender
        f.write(src + "\n")
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))
    # Absolute script path + PWD: Blender resolves relative paths against $PWD, not the process working directory.
    cmd = [blender, "--background", "-noaudio", "--threads", str(threads), "--python", os.path.join(src, f"chunk_{script}"), "--",
           "--start_idx", str(start), "--num_images", str(count), "--output_image_dir", tmp + "/",
           "--render_num_samples", "64", "--width", "320", "--height", "240", "--use_gpu", "0"]
    log = os.path.join(tmp, "blender.log")
    with open(log, "w") as f:      # Blender exits 0 even when its Python script fails, so keep its output for errors
        subprocess.check_call(cmd, cwd=src, env={**os.environ, "PWD": src}, stdout=f, stderr=subprocess.STDOUT)
    os.makedirs(os.path.dirname(chunk), exist_ok=True)
    n = 0
    with zipfile.ZipFile(chunk + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for f in sorted(glob.glob(os.path.join(tmp, "prob_*", "CLEVR_*.png"))):
            z.write(f, os.path.relpath(f, tmp))
            n += 1
    expected = n_c * min(count, len(np.load(os.path.join(src, npz))["y"]) - start)
    if n != expected:
        tail = open(log).read()[-3000:]
        raise SystemExit(f"{task} chunk {start}: {n} images rendered, expected {expected}. Blender output tail:\n{tail}")
    os.replace(chunk + ".tmp", chunk)
    shutil.rmtree(tmp)
    print(f"[clevr_art] {chunk}: {n} images")


def label_check(task, x, y):
    same = lambda a, b: bool(np.all(a == b))
    if task == "rmts":
        src = same(x[0], x[1])
        ok = [k for k in range(2) if same(x[2 + 2 * k], x[3 + 2 * k]) == src]
    else:
        pat = lambda a, b, c: (same(a, b), same(a, c), same(b, c))
        ok = [k for k in range(4) if pat(x[3], x[4], x[5 + k]) == pat(x[0], x[1], x[2])]
    return ok == [int(y)]


def pack(n_val):
    from make_clevr_art_ood import CAND_PROMPT          # needs PIL: run `pack` in the container (render runs on the host)
    src = os.path.join(OUT, "_src")
    for task, (_, npz, n_c) in TASKS.items():
        d = np.load(os.path.join(src, npz))
        xs, ys = d["x"], d["y"]
        bad = sum(not label_check(task, x, y) for x, y in zip(xs, ys))
        print(f"[clevr_art] {task}: {len(ys)} problems, {bad} labels disagree with the symbolic objects")
        chunks = sorted(glob.glob(os.path.join(OUT, "_chunks", f"{task}_train_*.zip")))
        members = {}
        for c in chunks:
            for m in zipfile.ZipFile(c).namelist():
                members[m] = c
        n = len(ys)
        for split, idx in (("train", range(0, n - n_val)), ("val", range(n - n_val, n))):
            zname = f"{task}_{split}.zip"
            rows = []
            with zipfile.ZipFile(os.path.join(OUT, zname + ".tmp"), "w", zipfile.ZIP_STORED) as z:
                for i in idx:
                    names = [f"prob_{i}/CLEVR_{k}.png" for k in range(n_c)]
                    if not all(m in members for m in names):
                        continue
                    for m in names:
                        z.writestr(m, zipfile.ZipFile(members[m]).read(m))
                    rows.append({"id": f"clevr-art/{task}/train/{i}", "candidates": [f"zip:{zname}::{m}" for m in names],
                                 "prompt": CAND_PROMPT[task], "answer": int(ys[i])})
            os.replace(os.path.join(OUT, zname + ".tmp"), os.path.join(OUT, zname))
            with open(os.path.join(OUT, f"{task}_{split}.jsonl"), "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows)
            print(f"[clevr_art] {task}_{split}: {len(rows)} problems ({len(idx) - len(rows)} not rendered yet)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "render", "pack"])
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=250)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=500)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup()
    elif a.cmd == "render":
        render(a.task, a.start, a.count, a.threads)
    else:
        pack(a.n_val)
