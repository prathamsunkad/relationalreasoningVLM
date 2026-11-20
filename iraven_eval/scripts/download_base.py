"""Download the pinned base model revision for each adapter key (weights as safetensors only)."""
import json, sys
from pathlib import Path
from huggingface_hub import snapshot_download
BUNDLE = Path.home() / 'scratch' / 'pgm-best-lora-2026-11-01-v1'
index = json.loads((BUNDLE / 'INDEX.json').read_text())
for key in sys.argv[1:]:
    e = [x for x in index['adapters'] if x['key'] == key][0]
    p = snapshot_download(e['base_model'], revision=e['base_revision'],
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', '*.jinja', '*.md'])
    print(key, e['base_model'], e['base_revision'], '->', p)
