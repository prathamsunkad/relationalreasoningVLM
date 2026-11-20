"""Fast regression contracts for the original candidate protocol; no model downloads."""
import unittest
import numpy as np
import torch
from pgmvlm.data import PGMStream, CandidateCollator, filled_matrices, to_panels, make_order
from pgmvlm.models import candidate_losses
from pgmvlm.evalcore import summarize


class CandidateContract(unittest.TestCase):
    def test_panel_bytes_and_candidate_order(self):
        panels = np.stack([np.full((160, 160), i, np.uint8) for i in range(16)])
        self.assertTrue(np.array_equal(to_panels(panels.reshape(160, 160, 16)), panels))
        for k, image in enumerate(filled_matrices(panels[:8], panels[8:])):
            a = np.asarray(image)
            self.assertEqual(image.size, (520, 520))
            self.assertTrue((a[350:510, 350:510] == 8 + k).all())
            self.assertTrue((a[10:170, 10:170] == 0).all())

    def test_loss_and_gradients(self):
        scores = torch.arange(16, dtype=torch.float32).reshape(2, 8).requires_grad_()
        targets = torch.tensor([2, 7])
        ce, bce, acc = candidate_losses(scores, targets)
        self.assertTrue(torch.equal(ce, torch.nn.functional.cross_entropy(scores, targets)))
        self.assertEqual(float(acc), 0.5)
        (ce + bce).backward()
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertGreater(float(scores.grad.abs().sum()), 0)

    def test_resume_order_and_summary_schema(self):
        short = make_order(9, 27, 23)
        long = make_order(9, 45, 23)
        self.assertTrue(np.array_equal(short, long[:27]))
        row = dict(file='p.npz', target=0, pred=0, correct=True, scores=[1.] + [-1.] * 7)
        summary = summarize([row])
        self.assertEqual(summary['scores']['answer_says_yes'], 1.)
        self.assertEqual(summary['scores']['distractor_says_yes'], 0.)
        self.assertEqual(summary['scores']['pred_hist'], [1] + [0] * 7)


if __name__ == '__main__':
    unittest.main()
