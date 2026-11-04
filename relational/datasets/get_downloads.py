"""Download public datasets (data only) and convert them to relvlm's packed jsonl format.

  python datasets/get_downloads.py vsr whatsup cvbench blink mmvp coco clevr

  vsr      VSR (Liu et al. 2023): random + zero-shot splits; binary "is this statement true?"   (train/dev/test)
           images: the COCO-2017 files each row links to, fetched once into vsr/images.zip
  whatsup  What'sUp (Kamath et al. 2023) Controlled_Images_A/B + COCO_QA_two_obj + VG_QA_two_obj; cand over captions
           (caption_options[0] is the correct caption, the What'sUp/ARO convention; we shuffle the order)   [eval only]
  cvbench  CV-Bench 2D (count, relation) + 3D (depth, distance); letter                                  [eval only]
  blink    BLINK val, all 14 subtasks; multi-image questions become one side-by-side sheet, one panel per image [eval]
  mmvp     MMVP 300 questions (a)/(b); letter                                                             [eval only]
  coco     COCO val2017 (5k images) -> image-only rows for Stage 0 (zip kept as downloaded)
  clevr    CLEVR v1.0 (19 GB zip kept as downloaded): image-only rows for Stage 0, and yes/no questions whose program
           uses spatial relations (relate) or attribute comparison (same_* / equal_* / less_than / greater_than)
Writes to $RELVLM_DATA (default ~/scratch/relvlm_data). Every split gets <split>.jsonl (+ .zip or a shared zip).
"""
import io
import json
import os
import random
import sys
import urllib.request
import zipfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import ROOT, Packer

LETTERS = "ABCDEFGH"


def hf_file(repo, path):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, path, repo_type="dataset")


def hf_list(repo):
    from huggingface_hub import list_repo_files
    return list_repo_files(repo, repo_type="dataset")


def parquet_rows(path, columns=None):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=columns)
    for batch in t.to_batches(256):
        yield from batch.to_pylist()


def img_bytes(cell):
    """HF image cell {'bytes': ..., 'path': ...} -> bytes."""
    return cell["bytes"] if isinstance(cell, dict) else cell


def download(url, dst):
    if os.path.exists(dst):
        return dst
    tmp = dst + ".part"
    print(f"[get] {url} -> {dst}", flush=True)
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst)
    return dst


def write_jsonl(path, rows):
    with open(path + ".tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(path + ".tmp", path)
    print(f"[pack] {path}: {len(rows)} rows", flush=True)


# ----------------------------------------------------------------------------- VSR
def vsr():
    out = os.path.join(ROOT, "vsr")
    os.makedirs(out, exist_ok=True)
    splits = {}
    for variant in ("random", "zeroshot"):
        for sp in ("train", "dev", "test"):
            p = hf_file(f"cambridgeltl/vsr_{variant}", f"{sp}.jsonl")
            splits[(variant, sp)] = [json.loads(l) for l in open(p) if l.strip()]
    links = {r["image"]: r["image_link"] for rows in splits.values() for r in rows}
    zpath = os.path.join(out, "images.zip")
    have = set(zipfile.ZipFile(zpath).namelist()) if os.path.exists(zpath) else set()
    missing = [k for k in links if k not in have]
    print(f"[vsr] {len(links)} images, {len(missing)} to fetch", flush=True)
    if missing:
        z = zipfile.ZipFile(zpath, "a", zipfile.ZIP_STORED)
        for i, name in enumerate(missing):
            try:
                z.writestr(name, urllib.request.urlopen(links[name], timeout=60).read())
            except Exception as e:                       # a few COCO links may fail; those rows are dropped
                print(f"[vsr] skip {name}: {e}", flush=True)
            if i % 1000 == 0:
                print(f"[vsr] {i}/{len(missing)}", flush=True)
        z.close()
    have = set(zipfile.ZipFile(zpath).namelist())
    for (variant, sp), rows in splits.items():
        out_rows = [{"id": f"vsr/{variant}/{sp}/{i}", "images": [f"zip:images.zip::{r['image']}"],
                     "prompt": f'Is this statement about the image true? "{r["caption"]}" Answer Yes or No.',
                     "answer": "Yes" if int(r["label"]) == 1 else "No", "relation": r["relation"]}
                    for i, r in enumerate(rows) if r["image"] in have]
        write_jsonl(os.path.join(out, f"{variant}_{sp}.jsonl"), out_rows)


# ----------------------------------------------------------------------------- What'sUp
def whatsup():
    out = os.path.join(ROOT, "whatsup")
    files = [f for f in hf_list("ServiceNow/whatsup_all") if f.startswith("data/") and
             any(f.split("/")[-1].startswith(s + "-") for s in
                 ("Controlled_Images_A", "Controlled_Images_B", "COCO_QA_two_obj", "VG_QA_two_obj"))]
    by_split = {}
    for f in files:
        by_split.setdefault(f.split("/")[-1].split("-0")[0], []).append(f)
    rng = random.Random(0)
    for split, fs in sorted(by_split.items()):
        pk = Packer(out, split)
        i = 0
        for f in sorted(fs):
            for r in parquet_rows(hf_file("ServiceNow/whatsup_all", f)):
                caps = list(r["caption_options"])
                order = list(range(len(caps)))
                rng.shuffle(order)
                ref = pk.add_image(f"{i:05d}.jpg", data=_as_jpeg(img_bytes(r["image_options"])))
                cands = [ref] * len(caps)
                # cand rows share one image; the prompt differs per candidate, so store per-candidate prompts
                pk.add_row({"id": f"whatsup/{split}/{i}", "candidates": cands,
                            "prompts": [f'Does this caption describe the image: "{caps[j]}"? Answer Yes or No.'
                                        for j in order],
                            "prompt": "", "answer": order.index(0), "captions": [caps[j] for j in order]})
                i += 1
        pk.close({"source": "ServiceNow/whatsup_all", "note": "caption_options[0] is correct; order shuffled"})


def _as_jpeg(data):
    im = Image.open(io.BytesIO(data))
    if im.format == "JPEG":
        return data
    buf = io.BytesIO()
    im.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


# ----------------------------------------------------------------------------- CV-Bench / BLINK / MMVP
def _letter_prompt(prompt):
    return prompt.strip() + "\nAnswer with the option's letter from the given choices directly."


def cvbench():
    out = os.path.join(ROOT, "cvbench")
    for part in ("2d", "3d"):
        pk = Packer(out, f"test_{part}")
        for i, r in enumerate(parquet_rows(hf_file("nyu-visionx/CV-Bench", f"test_{part}.parquet"))):
            ref = pk.add_image(f"{i:05d}.jpg", data=_as_jpeg(img_bytes(r["image"])))
            n = len(r["choices"])
            pk.add_row({"id": f"cvbench/{part}/{r.get('idx', i)}", "images": [ref], "prompt": _letter_prompt(r["prompt"]),
                        "choices": list(LETTERS[:n]), "answer": r["answer"].strip("()"), "task": r["task"]})
        pk.close({"source": "nyu-visionx/CV-Bench"})


def _sheet(images, gap=10, height=448):
    """Several images side by side (same height) on white; returns the sheet and one normalised box per image."""
    ims = [Image.open(io.BytesIO(b)).convert("RGB") for b in images]
    ims = [im.resize((max(1, round(im.width * height / im.height)), height)) for im in ims]
    W = sum(im.width for im in ims) + gap * (len(ims) + 1)
    H = height + 2 * gap
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    boxes, x = [], gap
    for im in ims:
        sheet.paste(im, (x, gap))
        boxes.append([round(x / W, 5), round(gap / H, 5), round((x + im.width) / W, 5), round((gap + height) / H, 5)])
        x += im.width + gap
    return sheet, boxes


def blink():
    out = os.path.join(ROOT, "blink")
    files = [f for f in hf_list("BLINK-Benchmark/BLINK") if f.endswith(".parquet") and "/val-" in f]
    pk = Packer(out, "val")
    for f in sorted(files):
        task = f.split("/")[0]
        for r in parquet_rows(hf_file("BLINK-Benchmark/BLINK", f)):
            imgs = [img_bytes(r[k]) for k in ("image_1", "image_2", "image_3", "image_4") if r.get(k)]
            n = len(r["choices"])
            row = {"id": f"blink/{r['idx']}", "prompt": _letter_prompt(r["prompt"]), "choices": list(LETTERS[:n]),
                   "answer": r["answer"].strip("()"), "task": task, "n_images": len(imgs)}
            if len(imgs) == 1:
                row["images"] = [pk.add_image(f"{r['idx']}.jpg", data=_as_jpeg(imgs[0]))]
            else:
                sheet, boxes = _sheet(imgs)
                row["images"] = [pk.add_image(f"{r['idx']}.jpg", sheet, fmt="JPEG")]
                row["panels"] = boxes
                row["prompt"] = (f"The {len(imgs)} images are shown side by side, left to right (image 1 is leftmost). "
                                 + row["prompt"])
            pk.add_row(row)
    pk.close({"source": "BLINK-Benchmark/BLINK", "note": "val split (test answers are hidden)"})


def mmvp():
    import csv
    out = os.path.join(ROOT, "mmvp")
    pk = Packer(out, "test")
    for r in csv.DictReader(open(hf_file("MMVP/MMVP", "Questions.csv"))):
        idx = int(r["Index"])
        opts = r["Options"].replace("(b)", "\n(B)").replace("(a)", "(A)").strip()
        ref = pk.add_image(f"{idx}.jpg", data=open(hf_file("MMVP/MMVP", f"MMVP Images/{idx}.jpg"), "rb").read())
        pk.add_row({"id": f"mmvp/{idx}", "images": [ref], "prompt": _letter_prompt(r["Question"] + "\n" + opts),
                    "choices": ["A", "B"], "answer": r["Correct Answer"].strip().strip("()").upper(),
                    "pair": (idx - 1) // 2})
    pk.close({"source": "MMVP/MMVP", "note": "report pair accuracy too (both questions of a pair right)"})


# ----------------------------------------------------------------------------- COCO / CLEVR (zips kept as is)
def coco():
    out = os.path.join(ROOT, "coco")
    os.makedirs(out, exist_ok=True)
    z = download("http://images.cocodataset.org/zips/val2017.zip", os.path.join(out, "val2017.zip"))
    names = sorted(n for n in zipfile.ZipFile(z).namelist() if n.endswith(".jpg"))
    write_jsonl(os.path.join(out, "stage0_val2017.jsonl"),
                [{"id": f"coco/{os.path.basename(n)}", "images": [f"zip:val2017.zip::{n}"]} for n in names])


CLEVR_RELATIONAL = ("relate", "same_", "equal_", "less_than", "greater_than")


def clevr(n_train=60000, n_val=6000, n_stage0=20000):
    out = os.path.join(ROOT, "clevr")
    os.makedirs(out, exist_ok=True)
    z = download("https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip", os.path.join(out, "CLEVR_v1.0.zip"))
    zf = zipfile.ZipFile(z)
    rng = random.Random(0)
    for split, n in (("train", n_train), ("val", n_val)):
        qs = json.loads(zf.read(f"CLEVR_v1.0/questions/CLEVR_{split}_questions.json"))["questions"]
        keep = [q for q in qs if q["answer"] in ("yes", "no")
                and any(f["function"].startswith(CLEVR_RELATIONAL) for f in q["program"])]
        rng.shuffle(keep)
        yes = [q for q in keep if q["answer"] == "yes"][: n // 2]       # balance yes / no
        no = [q for q in keep if q["answer"] == "no"][: n // 2]
        rows = [{"id": f"clevr/{split}/{q['question_index']}",
                 "images": [f"zip:CLEVR_v1.0.zip::CLEVR_v1.0/images/{split}/{q['image_filename']}"],
                 "prompt": q["question"] + " Answer Yes or No.", "answer": q["answer"].capitalize(),
                 "family": q.get("question_family_index")} for q in yes + no]
        rng.shuffle(rows)
        write_jsonl(os.path.join(out, f"relational_yesno_{split}.jsonl"), rows)
    imgs = sorted(n for n in zf.namelist() if n.startswith("CLEVR_v1.0/images/train/") and n.endswith(".png"))
    rng.shuffle(imgs)
    write_jsonl(os.path.join(out, "stage0_train.jsonl"),
                [{"id": f"clevr/{os.path.basename(n)}", "images": [f"zip:CLEVR_v1.0.zip::{n}"]} for n in imgs[:n_stage0]])


if __name__ == "__main__":
    jobs = {"vsr": vsr, "whatsup": whatsup, "cvbench": cvbench, "blink": blink, "mmvp": mmvp, "coco": coco, "clevr": clevr}
    for name in sys.argv[1:] or list(jobs):
        print(f"===== {name}", flush=True)
        jobs[name]()
