import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Exact parser from the dataset's comparison protocol
ANSWER_RE = re.compile(r'\s*(?:Answer\s*:\s*)?\(?([A-Z])\)?[.]?\s*', re.IGNORECASE)

def extract_answer(response_text, valid_choices):
    m = ANSWER_RE.match(response_text.strip())
    if not m:
        return None
    letter = m.group(1).upper()
    return letter if letter in valid_choices else None


def load_manifest(manifest_path):
    examples = []
    with open(manifest_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True,
                         help="Path to extracted clevr-art-ood-test-seed1-v1 folder")
    parser.add_argument("--task", choices=["rmts", "idrules"], required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=16)  # per protocol
    parser.add_argument("--flush_every", type=int, default=25)
    parser.add_argument("--prefill_answer", action="store_true",
                        help="Start the assistant reply with 'Answer:' so the model "
                             "continues with a letter (deviates from the protocol)")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    manifest_name = (
        "clevr-art-rmts-test-full.jsonl" if args.task == "rmts"
        else "clevr-art-idrules-test-full.jsonl"
    )
    examples = load_manifest(root / manifest_name)

    shard_examples = examples[args.shard_idx::args.num_shards]
    print(f"[{args.task} shard {args.shard_idx}/{args.num_shards}] "
          f"{len(shard_examples)} examples")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir, f"{args.task}_shard{args.shard_idx}.json"
    )

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"[shard {args.shard_idx}] model loaded on {model.device}")

    results = []
    start = time.time()

    for i, ex in enumerate(shard_examples):
        try:
            image_path = root / ex["images"][0]
            image = Image.open(image_path).convert("RGB")
            prompt_text = ex["prompt"]  # use exactly as given, no modification
            valid_choices = set(ex["choices"])

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            if args.prefill_answer:
                text = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False,
                ) + "Answer:"
                inputs = processor(
                    text=[text], images=[image], return_tensors="pt",
                ).to(model.device)
            else:
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
                )
            response = processor.decode(
                output_ids[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True,
            )
            predicted = extract_answer(response, valid_choices)

            results.append({
                "id": ex["id"],
                "answer": ex["answer"],
                "predicted": predicted,
                "correct": predicted == ex["answer"],
                "raw_response": response,
            })

        except Exception as e:
            results.append({"id": ex.get("id"), "error": str(e)})
            print(f"[shard {args.shard_idx}] ERROR on {ex.get('id')}: {e}")

        if (i + 1) % args.flush_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - start
            print(f"[shard {args.shard_idx}] {i+1}/{len(shard_examples)}, "
                  f"{elapsed/(i+1):.2f}s/example avg")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[shard {args.shard_idx}] finished -> {output_path}")


if __name__ == "__main__":
    main()