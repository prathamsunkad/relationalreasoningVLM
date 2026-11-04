"""I-RAVEN (Hu et al. 2021) -> packed matrix zips, using the official generator (github.com/husheng12345/SRAN/I-RAVEN).

The generator is Python 2.7. We fetch it, convert it with lib2to3, patch Python-2 integer divisions in const.py, and
run it one process per configuration (7 configurations x --num_samples, split 6:2:2 into train/val/test by the
generator itself). Problems are written to node-local scratch and packed into <out>/{train,val,test}.zip with members
"<configuration>/RAVEN_<k>_<split>.npz" (image (16,160,160), target, meta...), which data.MatrixSource reads directly
("raven:<out>/train.zip"). Run `--smoke` first: 5 problems per configuration, to check the port.

  python datasets/make_iraven.py --smoke
  python datasets/make_iraven.py --num_samples 10000 --workers 7
"""
import argparse
import glob
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT

FILES = ["AoT.py", "Attribute.py", "Rule.py", "__init__.py", "api.py", "build_tree.py", "const.py", "constraints.py",
         "main.py", "rendering.py", "sampling.py", "serialize.py", "solver.py"]
URL = "https://raw.githubusercontent.com/husheng12345/SRAN/HEAD/I-RAVEN/{}"
CONFIGS = ["center_single", "distribute_four", "distribute_nine", "left_center_single_right_center_single",
           "up_center_single_down_center_single", "in_center_single_out_center_single",
           "in_distribute_four_out_center_single"]


def fetch_and_port(src):
    if os.path.exists(os.path.join(src, ".ported")):
        return
    os.makedirs(src, exist_ok=True)
    for f in FILES:
        urllib.request.urlretrieve(URL.format(f), os.path.join(src, f))
    subprocess.check_call([sys.executable, "-m", "lib2to3", "-w", "-n", "--no-diffs", src],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    c = open(os.path.join(src, "const.py")).read()
    c = c.replace("IMAGE_SIZE / 2", "IMAGE_SIZE // 2").replace("IMAGE_SIZE / 4", "IMAGE_SIZE // 4")
    open(os.path.join(src, "const.py"), "w").write(c)
    open(os.path.join(src, ".ported"), "w").write("lib2to3 + const.py integer division\n")


def run_config(job):
    src, key, n, seed, save_dir = job
    driver = (
        "import sys, types; sys.path.insert(0, %r)\n"
        "import numpy as np\n"
        "if not hasattr(np, 'int'): np.int = int\n"
        "import main as M\n"
        "a = types.SimpleNamespace(num_samples=%d, save_dir=%r, seed=%d, fuse=0, val=2, test=2)\n"
        "cfgs = {'center_single': M.build_center_single, 'distribute_four': M.build_distribute_four,\n"
        "        'distribute_nine': M.build_distribute_nine,\n"
        "        'left_center_single_right_center_single': M.build_left_center_single_right_center_single,\n"
        "        'up_center_single_down_center_single': M.build_up_center_single_down_center_single,\n"
        "        'in_center_single_out_center_single': M.build_in_center_single_out_center_single,\n"
        "        'in_distribute_four_out_center_single': M.build_in_distribute_four_out_center_single}\n"
        "import os; os.makedirs(os.path.join(a.save_dir, %r), exist_ok=True)\n"
        "M.separate(a, {%r: cfgs[%r]()})\n" % (src, n, save_dir, seed, key, key, key))
    r = subprocess.run([sys.executable, "-c", driver], cwd=src, capture_output=True, text=True)
    tail = (r.stdout + r.stderr)[-1500:]
    return key, r.returncode, tail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "iraven"))
    ap.add_argument("--num_samples", type=int, default=10000, help="per configuration (the standard I-RAVEN size)")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    src = os.path.join(args.out, "_src")
    fetch_and_port(src)
    n = 5 if args.smoke else args.num_samples
    tmp = tempfile.mkdtemp(dir=os.environ.get("TMPDIR", "/tmp"))     # node-local: ~2 files per problem
    jobs = [(src, k, n, args.seed + i, tmp) for i, k in enumerate(CONFIGS)]
    with mp.Pool(min(args.workers, len(jobs))) as pool:
        for key, rc, tail in pool.imap_unordered(run_config, jobs):
            print(f"[iraven] {key}: rc={rc}" + ("" if rc == 0 else "\n" + tail), flush=True)
            if rc != 0:
                raise SystemExit(f"generation failed for {key}")
    out = args.out if not args.smoke else os.path.join(args.out, "_smoke")
    os.makedirs(out, exist_ok=True)
    for split in ("train", "val", "test"):
        files = sorted(glob.glob(os.path.join(tmp, "*", f"RAVEN_*_{split}.npz")))
        with zipfile.ZipFile(os.path.join(out, f"{split}.zip.tmp"), "w", zipfile.ZIP_STORED) as z:
            for f in files:
                z.write(f, os.path.relpath(f, tmp))
        os.replace(os.path.join(out, f"{split}.zip.tmp"), os.path.join(out, f"{split}.zip"))
        print(f"[iraven] {split}: {len(files)} problems -> {out}/{split}.zip", flush=True)
    shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
