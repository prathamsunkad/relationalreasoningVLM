"""Task protocols shared by LoRA training and evaluation.

candidate preserves the existing eight Yes/No sequences. answer_letter uses
one contact sheet and masked causal CE over the full vocabulary. These are
different experimental protocols, not interchangeable checkpoint formats.
"""
import inspect
import re

import numpy as np
import torch
import torch.nn.functional as F

from .data import CandidateCollator
from .models import ScoreTokens, candidate_scores, candidate_losses, autocast_ctx
from .manifests import render_pgm, PGM_PROMPT


def parse_answer(raw, choices):
    match = re.fullmatch(r"\s*(?:Answer\s*:\s*)?\(?([A-Z])\)?[.]?\s*", raw, re.I)
    answer = match.group(1).upper() if match else None
    return answer if answer in choices else None


def image_question(item):
    if 'image' in item:
        return item['image'], item['prompt'], item['choices']
    return render_pgm(np.concatenate([item['context'], item['candidates']])), PGM_PROMPT, list('ABCDEFGH')


def encode_answer(processor, image, prompt, answer, device, dtype):
    """Same native template and expanded-image prefix for train and inference."""
    user = {'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}
    options = dict(tokenize=True, return_dict=True, return_tensors='pt', truncation=False)
    prefix = processor.apply_chat_template([user], add_generation_prompt=True, **options)
    full = processor.apply_chat_template([user, {'role': 'assistant', 'content': [{'type': 'text', 'text': answer}]}],
                                         add_generation_prompt=False, **options)
    start = prefix['input_ids'].shape[-1]
    if full['input_ids'].shape[-1] <= start or not torch.equal(prefix['input_ids'], full['input_ids'][:, :start]):
        raise ValueError('Chat template changed the inference prefix; cannot safely mask the loss')
    for key in ('pixel_values', 'pixel_attention_mask', 'spatial_shapes', 'image_grid_thw'):
        if key in prefix and (key not in full or not torch.equal(prefix[key], full[key])):
            raise ValueError(f'Train/evaluation image tensors differ: {key}')
    if processor.tokenizer.decode(full['input_ids'][0, start:], skip_special_tokens=True).strip() != answer:
        raise ValueError('Unexpected supervised assistant suffix')
    move = lambda d: {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype) for k, v in d.items()}
    return move(prefix), move(full), start


def answer_losses(model, full, start):
    """Return suffix CE, first answer-token CE and first-token logits."""
    count = full['input_ids'].shape[-1] - start
    base = model.get_base_model() if hasattr(model, 'get_base_model') else model
    kw = {'logits_to_keep': count + 1} if 'logits_to_keep' in inspect.signature(base.forward).parameters else {}
    logits = model(**full, use_cache=False, **kw).logits[:, -(count + 1):-1].float()
    labels = full['input_ids'][:, start:]
    return (F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)),
            F.cross_entropy(logits[:, 0], labels[:, 0]), logits[:, 0])


class LetterCollator:
    """Keep serial native model calls: supports tiled images and unpadded hooks."""
    def __call__(self, items):
        return {'items': items, 'targets': torch.tensor([i['target'] for i in items]),
                'files': [i['file'] for i in items]}


class CandidateProtocol:
    name = 'candidate'
    def __init__(self, processor, family, args):
        self.tokens = ScoreTokens(processor.tokenizer)
        self.collator = CandidateCollator(processor, family, args.crop_to_patches, cell_size=args.panel_px)
    def describe(self):
        return self.tokens.describe()
    def training(self, model, batch, device, dtype, loss_name):
        scores = candidate_scores(model, batch, self.tokens, device, dtype)
        ce, bce, acc = candidate_losses(scores, batch['targets'])
        return {'loss': {'ce8': ce, 'bce': bce, 'both': ce+bce}[loss_name], 'ce8': ce, 'bce': bce, 'train_acc': acc}
    def records(self, model, batch, device, dtype):
        scores = candidate_scores(model, batch, self.tokens, device, dtype).cpu()
        return [dict(file=f, target=t, pred=int(s.argmax()), correct=int(s.argmax()) == t,
                     scores=[round(float(x), 4) for x in s])
                for f, t, s in zip(batch['files'], batch['targets'].tolist(), scores)]


class AnswerLetterProtocol:
    name = 'answer_letter'
    def __init__(self, processor, family, args):
        self.processor = processor
        self.collator = LetterCollator()
        self.max_new_tokens = getattr(args, 'max_new_tokens', 16)
        self.loss_scope = getattr(args, 'answer_loss_scope', 'suffix')
    def describe(self):
        return f'greedy answer letter; full-vocabulary {self.loss_scope} CE; {self.max_new_tokens} output tokens'
    def encoded(self, item, model, device, dtype):
        image, prompt, choices = image_question(item)
        answer = choices[item['target']]
        prefix, full, start = encode_answer(self.processor, image, prompt, answer, device, dtype)
        limits = [getattr(getattr(model.config, 'text_config', model.config), 'max_position_embeddings', None),
                  getattr(self.processor.tokenizer, 'model_max_length', None)]
        limits = [n for n in limits if isinstance(n, int) and 0 < n < 10**9]
        if limits and max(full['input_ids'].shape[-1], start + self.max_new_tokens) > min(limits):
            raise ValueError('Question exceeds native context budget; truncation is disabled')
        if self.loss_scope == 'answer' and len(self.processor.tokenizer.encode(answer, add_special_tokens=False)) != 1:
            raise ValueError('Answer-only CE requires a single answer token')
        return prefix, full, start, choices
    def training(self, model, batch, device, dtype, loss_name):
        if loss_name != 'causal_ce':
            raise ValueError('answer_letter requires causal_ce (full vocabulary), not candidate ce8')
        values = []
        for item in batch['items']:
            _, full, start, _ = self.encoded(item, model, device, dtype)
            with autocast_ctx(device, dtype):
                suffix, answer, logits = answer_losses(model, full, start)
            acc = (logits.argmax(-1) == full['input_ids'][:, start]).float().mean()
            values.append((suffix, answer, acc))
        suffix, answer, acc = [torch.stack(v).mean() for v in zip(*values)]
        return {'loss': suffix if self.loss_scope == 'suffix' else answer,
                'suffix_loss': suffix, 'answer_loss': answer, 'train_acc': acc}
    def records(self, model, batch, device, dtype):
        records = []
        for item in batch['items']:
            prefix, full, start, choices = self.encoded(item, model, device, dtype)
            with autocast_ctx(device, dtype):
                suffix, answer, _ = answer_losses(model, full, start)
                out = model.generate(**prefix, max_new_tokens=self.max_new_tokens, do_sample=False, use_cache=True)
            raw = self.processor.tokenizer.decode(out[0, start:], skip_special_tokens=True).strip()
            letter = parse_answer(raw, choices)
            pred = choices.index(letter) if letter is not None else -1
            records.append(dict(file=item['file'], target=item['target'], pred=pred, correct=pred == item['target'],
                                protocol=self.name, choices=choices, raw_output=raw, invalid=pred < 0,
                                loss=float(suffix if self.loss_scope == 'suffix' else answer),
                                suffix_loss=float(suffix), answer_loss=float(answer)))
        return records


def add_protocol_args(parser):
    parser.add_argument('--protocol', choices=['candidate', 'answer_letter'], default='candidate')
    parser.add_argument('--answer_loss_scope', choices=['suffix', 'answer'], default='suffix',
                        help='suffix preserves historical LoRA: answer plus chat-ending tokens')
    parser.add_argument('--max_new_tokens', type=int, default=16)


def build_protocol(args, processor, family):
    return (CandidateProtocol if args.protocol == 'candidate' else AnswerLetterProtocol)(processor, family, args)
