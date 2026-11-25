"""Verify exported bytes and their lossless correspondence to remote SHA256."""
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "PROVENANCE.json").read_text())
for entry in manifest["artifacts"]:
    data = (root / entry["path"]).read_bytes()
    assert len(data) == entry["bytes"], entry["path"]
    assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    raw = gzip.decompress(data) if entry["encoding"] == "gzip" else data
    assert len(raw) == entry["original_bytes"], entry["path"]
    assert hashlib.sha256(raw).hexdigest() == entry["remote_sha256"], entry["path"]
print(f"Verified {len(manifest['artifacts'])} records against remote SHA256.")
