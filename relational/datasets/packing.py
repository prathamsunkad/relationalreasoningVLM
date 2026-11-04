"""Write a dataset split as <split>.jsonl + <split>.zip (images stored inside the zip).

Scratch has a 1M file-count quota (PGM alone uses ~700k), so every new dataset is packed: one jsonl and one zip per
split. Rows reference images as "zip:<split>.zip::<member>", which data.JsonlSource resolves relative to the jsonl.
"""
import io
import json
import os
import zipfile

ROOT = os.environ.get("RELVLM_DATA", os.path.expanduser("~/scratch/relvlm_data"))


class Packer:
    def __init__(self, out_dir, split):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.split = out_dir, split
        self.zip_name = f"{split}.zip"
        self._tmp_zip = os.path.join(out_dir, self.zip_name + ".tmp")
        self._tmp_jsonl = os.path.join(out_dir, f"{split}.jsonl.tmp")
        self.zip = zipfile.ZipFile(self._tmp_zip, "w", zipfile.ZIP_STORED)   # PNG/JPEG are already compressed
        self.rows = open(self._tmp_jsonl, "w")
        self.n_rows = self.n_images = 0
        self.kinds = {}

    def add_image(self, name, image=None, data=None, fmt="PNG"):
        """PIL image (encoded as fmt) or raw encoded bytes -> reference string for a jsonl row."""
        if data is None:
            buf = io.BytesIO()
            image.save(buf, format=fmt)
            data = buf.getvalue()
        self.zip.writestr(name, data)
        self.n_images += 1
        return f"zip:{self.zip_name}::{name}"

    def add_row(self, row):
        self.rows.write(json.dumps(row) + "\n")
        self.n_rows += 1
        k = "cand" if "candidates" in row else ("image" if "answer" not in row else
                                                ("letter" if row.get("choices") not in (None, ["Yes", "No"]) else "binary"))
        self.kinds[k] = self.kinds.get(k, 0) + 1

    def close(self, meta=None):
        self.zip.close()
        self.rows.close()
        os.replace(self._tmp_zip, os.path.join(self.out_dir, self.zip_name))
        os.replace(self._tmp_jsonl, os.path.join(self.out_dir, f"{self.split}.jsonl"))
        info = {"split": self.split, "rows": self.n_rows, "images": self.n_images, "kinds": self.kinds, **(meta or {})}
        with open(os.path.join(self.out_dir, f"{self.split}.info.json"), "w") as f:
            json.dump(info, f, indent=1)
        print(f"[pack] {self.out_dir}/{self.split}: {info}", flush=True)
        return info


def boxes_from_centers(centers, half, W, H):
    """Normalised panel boxes [x0, y0, x1, y1] around pixel centres (row, col) with half-size `half`."""
    return [[round((c - half) / W, 5), round((r - half) / H, 5), round((c + half) / W, 5), round((r + half) / H, 5)]
            for r, c in centers]
