"""Check the real external teacher in the exact job runtime before VLM loading."""
import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import torch
from pgmvlm.checkpoints import atomic_json
from pgmvlm.pretrained_relations import FrozenRelations


def main(plan, output):
    torch.set_num_threads(2)
    expected = {"colorlog": "6.9.0", "einops": "0.8.1"}
    versions = {}
    for name, version in expected.items():
        module = importlib.import_module(name)
        actual = importlib.metadata.version(name)
        if actual != version:
            raise RuntimeError(f"Expected {name}=={version}, got {actual}")
        versions[name] = {"version": actual, "path": module.__file__}
    # Include the dynamically imported panel preprocessing dependency.
    importlib.import_module("torchvision.transforms")
    teacher = FrozenRelations(plan["teacher_package"], plan["teacher_hashes"], device="cpu")
    counts = {"encoder": sum(p.numel() for p in teacher.encoder.parameters()),
              "abstractor": sum(p.numel() for p in teacher.scorer.parameters())}
    if counts != {"encoder": 169506, "abstractor": 2378817}:
        raise RuntimeError(f"Unexpected teacher parameter counts: {counts}")
    if any(p.requires_grad for model in (teacher.encoder, teacher.scorer) for p in model.parameters()):
        raise RuntimeError("Teacher is not frozen")
    result = {"passed": True, "check": "real reference imports, hashes and strict CPU checkpoint loading",
              "dependencies": versions, "parameters": counts, "torch": torch.__version__}
    atomic_json(output, result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(json.loads(Path(args.plan).read_text()), Path(args.output))
