"""Serial native-LFM protocols for a matched, frozen-VLM Abstractor pilot.

Candidate scores share one model and one module across all eight matrices.
No answer label enters rendering or candidate prompts. CE is over the eight
scores, not eight independent binary losses. Direct-answer training preserves
the historical answer-plus-chat-ending objective.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def candidate_images(sheet):
    import numpy as np
    from .panels import split_composite
    from .data import filled_matrices
    panels = [np.asarray(p.convert('L')) for p in split_composite(sheet)]
    return filled_matrices(panels[:8], panels[8:], cell_size=160)


def candidate_ce(scores, target):
    if scores.shape != (8,):
        raise ValueError('Exactly eight scores from one question are required')
    return F.cross_entropy(scores.unsqueeze(0), torch.tensor([target], device=scores.device))


def yes_no_score(hidden, head, yes_id, no_id):
    # FP32 avoids quantizing small differences between candidate scores in BF16.
    value = hidden.float() @ (head.weight[yes_id].float() - head.weight[no_id].float())
    if getattr(head, 'bias', None) is not None:
        value = value + head.bias[yes_id].float() - head.bias[no_id].float()
    return value.squeeze()


def encode_candidate(processor, image, device='cuda', dtype=torch.bfloat16):
    from .panels import messages
    from .data import PROMPT_TEXT
    batch = processor.apply_chat_template(messages(image, PROMPT_TEXT, 'composite'),
        add_generation_prompt=True, tokenize=True, return_dict=True,
        return_tensors='pt', truncation=False)
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()}


def score_matrices(model, processor, matrices, tokens):
    values = []
    for matrix in matrices:
        batch = encode_candidate(processor, matrix)
        output = model(**batch, use_cache=False, logits_to_keep=1, output_hidden_states=True)
        values.append(yes_no_score(output.hidden_states[-1][:, -1],
                                  model.get_output_embeddings(), tokens.yes_id, tokens.no_id))
    return torch.stack(values)


def question_result(model, processor, image, target, protocol, tokens, letter_ids,
                    *, generate=False):
    from .manifests import PGM_PROMPT
    from .protocols import encode_answer, answer_losses, parse_answer
    if protocol == 'candidate':
        scores = score_matrices(model, processor, candidate_images(image), tokens)
        return dict(loss=candidate_ce(scores, target), pred=int(scores.detach().argmax()),
                    scores=scores.detach().cpu().tolist())
    if protocol != 'answer_letter':
        raise ValueError(protocol)
    prefix, full, start = encode_answer(processor, image, PGM_PROMPT, chr(65+target),
                                        'cuda', torch.bfloat16)
    suffix, answer, logits = answer_losses(model, full, start)
    predicted_token = int(logits.detach().argmax(-1))
    pred = letter_ids.index(predicted_token) if predicted_token in letter_ids else -1
    result = dict(loss=suffix, answer_loss=float(answer.detach()), first_token_pred=pred,
                  pred=pred, scores=logits.detach()[0, letter_ids].cpu().tolist())
    if generate:
        output = model.generate(**prefix, max_new_tokens=16, do_sample=False, use_cache=True)
        raw = processor.tokenizer.decode(output[0, start:], skip_special_tokens=True).strip()
        letter = parse_answer(raw, list('ABCDEFGH'))
        result.update(pred=ord(letter)-65 if letter else -1, raw_output=raw)
    return result


def paired_order(names, update):
    names = list(names)
    offset = update % len(names)
    return names[offset:] + names[:offset]
