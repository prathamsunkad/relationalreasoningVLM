import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
import torch.nn.functional as F

from pgmvlm.protocols import encode_answer, answer_losses, parse_answer, AnswerLetterProtocol
from pgmvlm.datasets import select_files, read_manifest
from pgmvlm.evalcore import summarize


class Processor:
    def __init__(self):
        self.tokenizer = SimpleNamespace(decode=lambda ids, **kw: 'A', encode=lambda *a, **kw: [3], model_max_length=50)
    def apply_chat_template(self, messages, **kwargs):
        ids = [1, 2] if kwargs['add_generation_prompt'] else [1, 2, 3, 4]
        return dict(input_ids=torch.tensor([ids]), pixel_values=torch.ones(1, 2), attention_mask=torch.ones(1, len(ids)))


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.arange(20, dtype=torch.float32).reshape(1, 4, 5) / 10)
    def forward(self, input_ids, use_cache=False, logits_to_keep=0, **kwargs):
        return SimpleNamespace(logits=self.logits[:, -logits_to_keep:])


class ProtocolTests(unittest.TestCase):
    def test_causal_suffix_ce_and_answer_ce_are_not_ce8(self):
        prefix, full, start = encode_answer(Processor(), Image.new('RGB',(8,8)), 'p', 'A', 'cpu', torch.float32)
        self.assertEqual(start, 2)
        self.assertEqual(prefix['input_ids'].tolist(), [[1, 2]])
        model = Toy()
        loss, answer, _ = answer_losses(model, full, start)
        reference = F.cross_entropy(model.logits[:, 1:3].reshape(-1, 5), torch.tensor([3, 4]))
        self.assertTrue(torch.equal(loss, reference))
        self.assertTrue(torch.equal(answer, F.cross_entropy(model.logits[:, 1], torch.tensor([3]))))
        loss.backward()
        self.assertEqual(float(model.logits.grad[:, 0].abs().sum()), 0.)
        self.assertGreater(float(model.logits.grad[:, 1:3].abs().sum()), 0.)

    def test_reject_changed_prefix(self):
        class Broken(Processor):
            def apply_chat_template(self, messages, **kwargs):
                out = super().apply_chat_template(messages, **kwargs)
                if not kwargs['add_generation_prompt']:
                    out['input_ids'][0, 0] = 9
                return out
        with self.assertRaisesRegex(ValueError, 'prefix'):
            encode_answer(Broken(), Image.new('RGB',(8,8)), 'p', 'A', 'cpu', torch.float32)

    def test_parser_does_not_extract_letters_from_reasoning(self):
        self.assertEqual(parse_answer('Answer: (B).', list('ABCD')), 'B')
        for raw in ['I think A or B', 'A B', 'Z', '']:
            self.assertIsNone(parse_answer(raw, list('ABCD')))

    def test_explicit_list_preserves_order_and_rejects_missing_or_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'train.tsv'
            path.write_text('# files\nb\tmetadata\na\tmetadata\n')
            self.assertEqual(select_files(['a', 'b', 'c'], path), ['b', 'a'])
            with self.assertRaises(ValueError):
                select_files(['a'], path)
            path.write_text('a\na\n')
            with self.assertRaises(ValueError):
                select_files(['a'], path)

    def test_summary_separates_protocols_and_invalid_answers(self):
        row = dict(file='a', target=0, pred=-1, correct=False, choices=list('AB'),
                   protocol='answer_letter', loss=2., answer_loss=3.)
        s = summarize([row])
        self.assertEqual(s['scores']['invalid'], 1)
        self.assertEqual(s['scores']['loss'], 2.)
        self.assertNotIn('answer_says_yes', s['scores'])
        with self.assertRaisesRegex(ValueError, 'protocol'):
            summarize([row, dict(file='b', target=0, pred=0, correct=True, scores=[1.]*8)])


if __name__ == '__main__':
    unittest.main()
