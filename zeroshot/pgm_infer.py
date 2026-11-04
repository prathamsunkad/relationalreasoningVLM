import argparse
import json
import os
import re
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForImageTextToText, AutoProcessor


def load_pgm_instance(npz_path, image_key="image", target_key="target"):
    data = np.load(npz_path)
    panels = data[image_key].reshape(16, 160, 160)
    return panels[:8], panels[8:], int(data[target_key])


def panels_to_grid_image(panels, grid_shape, cell_size=160, pad=10, add_labels=False, label_offset=1):
    rows, cols = grid_shape
    width = cols * cell_size + (cols + 1) * pad
    height = rows * cell_size + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    font = None
    if add_labels:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

    for i, panel in enumerate(panels):
        row, col = divmod(i, cols)
        panel = panel.astype(np.uint8)
        panel_img = Image.fromarray(panel, mode="L").convert("RGB")
        x = pad + col * (cell_size + pad)
        y = pad + row * (cell_size + pad)
        canvas.paste(panel_img, (x, y))
        if add_labels:
            label = str(i + label_offset)
            draw.rectangle([x, y, x + 28, y + 28], fill=(255, 255, 0))
            draw.text((x + 5, y + 2), label, fill=(0, 0, 0), font=font)

    return canvas


def extract_answer(response_text):
    matches = re.findall(r'\b([1-8])\b', response_text)
    return int(matches[-1]) - 1 if matches else None


# PROMPT_TEXT = (
#     "The first image shows a 3x3 pattern matrix with the bottom-right panel missing. "
#     "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
#     "and top to bottom. Determine which candidate completes the matrix by analyzing the "
#     "patterns across its rows and columns. "
#     "Your final response must contain exactly 1 line:\n"
#     "Answer: <number>\n"
#     "Do not output anything before or after this line."
# )


# PROMPT_TEXT = (
#     "The first image shows a 3x3 abstract reasoning matrix with the bottom-right panel missing. "
#     "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
#     "and top to bottom.\n\n"
#     "Solve it by checking how each of these attributes changes across the rows and columns: "
#     "shape type, shape size, shape color, number/count of shapes, shape position, line type, "
#     "and line color.\n\n"
#     "Each attribute follows one of these relations: constant (unchanged), progression "
#     "(steadily increases or decreases), XOR, OR, AND, or consistent union.\n\n"
#     "Identify which relation holds for each attribute across the rows, and separately across "
#     "the columns. Use whichever direction (row or column) gives a consistent rule to determine "
#     "what the missing bottom-right panel must look like. Compare that against the 8 candidates "
#     "and pick the best match.\n\n"
#     "Respond with exactly one line and nothing else, in this exact format:\n"
#     "Answer: <number>"
# )


PROMPT_TEXT = (
    "The first image shows a 3x3 abstract reasoning matrix with the bottom-right panel missing. "
    "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
    "and top to bottom.\n\n"
    "Determine which candidate completes the matrix by checking how shape type, size, color, "
    "count, and position, and line type and color, change across rows and columns "
    "(constant, progression, XOR, OR, AND, or consistent union).\n\n"
    "Do not explain your reasoning. Output only the final answer, in this exact format and "
    "nothing else:\n"
    "Answer: <number>"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--shard_idx", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--max_examples", type=int, default=0, help="0 = use all files in this shard")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--flush_every", type=int, default=25)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"results_shard{args.shard_idx}.json")

    # Deterministic sharding: sort filenames, take every Nth
    all_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
    shard_files = all_files[args.shard_idx::args.num_shards]
    if args.max_examples > 0:
        shard_files = shard_files[:args.max_examples]

    print(f"[shard {args.shard_idx}/{args.num_shards}] {len(shard_files)} files to process")

    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto"
    )
    print(f"[shard {args.shard_idx}] model loaded on {model.device}")

    results = []
    start = time.time()

    for i, fname in enumerate(shard_files):
        npz_path = os.path.join(args.npz_dir, fname)
        try:
            context_panels, answer_panels, target = load_pgm_instance(npz_path)
            context_image = panels_to_grid_image(context_panels, (3, 3), add_labels=False)
            answers_image = panels_to_grid_image(answer_panels, (2, 4), add_labels=True, label_offset=1)

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": context_image},
                    {"type": "image", "image": answers_image},
                    {"type": "text", "text": PROMPT_TEXT},
                ]
            }]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
#                     repetition_penalty=1.3, no_repeat_ngram_size=3,
                )
            response = processor.decode(
                output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
            )
            predicted = extract_answer(response)

            results.append({
                "file": fname, "target": target, "predicted": predicted,
                "correct": predicted == target, "raw_response": response,
            })

        except Exception as e:
            # Don't let one bad file kill a multi-hour job
            results.append({"file": fname, "error": str(e)})
            print(f"[shard {args.shard_idx}] ERROR on {fname}: {e}")

        if (i + 1) % args.flush_every == 0:
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            elapsed = time.time() - start
            print(f"[shard {args.shard_idx}] {i+1}/{len(shard_files)} done, "
                  f"{elapsed/(i+1):.2f}s/example avg")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[shard {args.shard_idx}] finished. Results saved to {output_path}")


if __name__ == "__main__":
    main()



# import argparse
# import json
# import os
# import re
# import time

# import numpy as np
# import torch
# from PIL import Image, ImageDraw, ImageFont
# from transformers import AutoModelForImageTextToText, AutoProcessor


# def load_pgm_instance(npz_path, image_key="image", target_key="target"):
#     data = np.load(npz_path)
#     img = data[image_key]
#     # The PGM files store the 16 panels as channels: (160, 160, 16), panel k = img[:, :, k].
#     # (A plain reshape(16, 160, 160) scrambles the pixels, so transpose instead.)
#     if img.shape == (160, 160, 16):
#         panels = np.transpose(img, (2, 0, 1))
#     elif img.shape == (16, 160, 160):
#         panels = img
#     else:
#         raise ValueError(f"unexpected image shape {img.shape}")
#     return panels[:8], panels[8:], int(data[target_key])


# def panels_to_grid_image(panels, grid_shape, cell_size=160, pad=10, add_labels=False, label_offset=1):
#     rows, cols = grid_shape
#     width = cols * cell_size + (cols + 1) * pad
#     height = rows * cell_size + (rows + 1) * pad
#     canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
#     draw = ImageDraw.Draw(canvas)

#     font = None
#     if add_labels:
#         try:
#             font = ImageFont.truetype("DejaVuSans-Bold.ttf", 24)
#         except OSError:
#             font = ImageFont.load_default()

#     for i, panel in enumerate(panels):
#         row, col = divmod(i, cols)
#         panel = panel.astype(np.uint8)
#         panel_img = Image.fromarray(panel, mode="L").convert("RGB")
#         x = pad + col * (cell_size + pad)
#         y = pad + row * (cell_size + pad)
#         canvas.paste(panel_img, (x, y))
#         if add_labels:
#             label = str(i + label_offset)
#             draw.rectangle([x, y, x + 28, y + 28], fill=(255, 255, 0))
#             draw.text((x + 5, y + 2), label, fill=(0, 0, 0), font=font)

#     return canvas


# def extract_answer(response_text):
#     matches = re.findall(r'\b([1-8])\b', response_text)
#     return int(matches[-1]) - 1 if matches else None


# # PROMPT_TEXT = (
# #     "The first image shows a 3x3 pattern matrix with the bottom-right panel missing. "
# #     "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
# #     "and top to bottom. Determine which candidate completes the matrix by analyzing the "
# #     "patterns across its rows and columns. "
# #     "Your final response must contain exactly 1 line:\n"
# #     "Answer: <number>\n"
# #     "Do not output anything before or after this line."
# # )


# # PROMPT_TEXT = (
# #     "The first image shows a 3x3 abstract reasoning matrix with the bottom-right panel missing. "
# #     "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
# #     "and top to bottom.\n\n"
# #     "Solve it by checking how each of these attributes changes across the rows and columns: "
# #     "shape type, shape size, shape color, number/count of shapes, shape position, line type, "
# #     "and line color.\n\n"
# #     "Each attribute follows one of these relations: constant (unchanged), progression "
# #     "(steadily increases or decreases), XOR, OR, AND, or consistent union.\n\n"
# #     "Identify which relation holds for each attribute across the rows, and separately across "
# #     "the columns. Use whichever direction (row or column) gives a consistent rule to determine "
# #     "what the missing bottom-right panel must look like. Compare that against the 8 candidates "
# #     "and pick the best match.\n\n"
# #     "Respond with exactly one line and nothing else, in this exact format:\n"
# #     "Answer: <number>"
# # )


# PROMPT_TEXT = (
#     "The first image shows a 3x3 abstract reasoning matrix with the bottom-right panel missing. "
#     "The second image shows 8 candidate panels, numbered 1 through 8, arranged left to right "
#     "and top to bottom.\n\n"
#     "Determine which candidate completes the matrix by checking how shape type, size, color, "
#     "count, and position, and line type and color, change across rows and columns "
#     "(constant, progression, XOR, OR, AND, or consistent union).\n\n"
#     "Do not explain your reasoning. Output only the final answer, in this exact format and "
#     "nothing else:\n"
#     "Answer: <number>"
# )


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--npz_dir", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
#     parser.add_argument("--shard_idx", type=int, required=True)
#     parser.add_argument("--num_shards", type=int, required=True)
#     parser.add_argument("--max_examples", type=int, default=0, help="0 = use all files in this shard")
#     parser.add_argument("--max_new_tokens", type=int, default=20)
#     parser.add_argument("--flush_every", type=int, default=25)
#     args = parser.parse_args()

#     os.makedirs(args.output_dir, exist_ok=True)
#     output_path = os.path.join(args.output_dir, f"results_shard{args.shard_idx}.json")

#     # Deterministic sharding: sort filenames, take every Nth
#     all_files = sorted(f for f in os.listdir(args.npz_dir) if f.endswith(".npz"))
#     shard_files = all_files[args.shard_idx::args.num_shards]
#     if args.max_examples > 0:
#         shard_files = shard_files[:args.max_examples]

#     print(f"[shard {args.shard_idx}/{args.num_shards}] {len(shard_files)} files to process")

#     processor = AutoProcessor.from_pretrained(args.model_id)
#     model = AutoModelForImageTextToText.from_pretrained(
#         args.model_id, dtype=torch.bfloat16, device_map="auto"
#     )
#     print(f"[shard {args.shard_idx}] model loaded on {model.device}")

#     results = []
#     start = time.time()

#     for i, fname in enumerate(shard_files):
#         npz_path = os.path.join(args.npz_dir, fname)
#         try:
#             context_panels, answer_panels, target = load_pgm_instance(npz_path)
#             context_image = panels_to_grid_image(context_panels, (3, 3), add_labels=False)
#             answers_image = panels_to_grid_image(answer_panels, (2, 4), add_labels=True, label_offset=1)

#             messages = [{
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": context_image},
#                     {"type": "image", "image": answers_image},
#                     {"type": "text", "text": PROMPT_TEXT},
#                 ]
#             }]
#             inputs = processor.apply_chat_template(
#                 messages, add_generation_prompt=True, tokenize=True,
#                 return_dict=True, return_tensors="pt"
#             ).to(model.device)

#             with torch.no_grad():
#                 output_ids = model.generate(
#                     **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
# #                     repetition_penalty=1.3, no_repeat_ngram_size=3,
#                 )
#             response = processor.decode(
#                 output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True
#             )
#             predicted = extract_answer(response)

#             results.append({
#                 "file": fname, "target": target, "predicted": predicted,
#                 "correct": predicted == target, "raw_response": response,
#             })

#         except Exception as e:
#             # Don't let one bad file kill a multi-hour job
#             results.append({"file": fname, "error": str(e)})
#             print(f"[shard {args.shard_idx}] ERROR on {fname}: {e}")

#         if (i + 1) % args.flush_every == 0:
#             with open(output_path, "w") as f:
#                 json.dump(results, f, indent=2)
#             elapsed = time.time() - start
#             print(f"[shard {args.shard_idx}] {i+1}/{len(shard_files)} done, "
#                   f"{elapsed/(i+1):.2f}s/example avg")

#     with open(output_path, "w") as f:
#         json.dump(results, f, indent=2)
#     print(f"[shard {args.shard_idx}] finished. Results saved to {output_path}")


# if __name__ == "__main__":
#     main()