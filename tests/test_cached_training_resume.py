"""Exercise production checkpoint publication and exact joint-training resume."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from test_cached_interleave import CachedInterleave, optimizer, setup, step, train_lora
from tools import train_cached_interleave as trainer


class CachedTrainingResumeTests(unittest.TestCase):
    def test_production_checkpoint_restores_progress_rng_and_next_joint_update(self):
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        model, batch, bank, mask, backbone = setup()
        train_lora(model)  # Enables existing LoRA parameters and dropout=.05.
        connector = CachedInterleave(backbone, input_dim=32)
        self.addCleanup(connector.remove)
        opt = optimizer(model, connector)
        for _ in range(5):
            step(model, connector, opt, batch, bank, mask)

        with tempfile.TemporaryDirectory() as directory, \
                patch.object(trainer, 'require_headroom'), \
                patch.object(torch.cuda, 'get_rng_state_all', return_value=[]), \
                patch.object(torch.cuda, 'set_rng_state_all'):
            # Skip GPU/data setup, but use the real save/restore implementation.
            run = trainer.Run.__new__(trainer.Run)
            run.model, run.connector, run.opt = model, connector, opt
            run.out = Path(directory)
            run.plan = {'test': 'native-tiny-internvl'}
            run.identity = trainer.contract(run.plan)
            run.banks = SimpleNamespace(flush=Mock())
            run.progress = dict(update=5, examples=80, best_loss=1.9,
                                best_checkpoint=None, pending_validation=False)
            saved_progress = copy.deepcopy(run.progress)
            run.save()
            run.banks.flush.assert_called_once_with()
            state = torch.load(run.out / 'state.pt', map_location='cpu', weights_only=True)
            self.assertEqual(state['step'], 5)
            self.assertEqual(state['progress'], saved_progress)
            torch.testing.assert_close(torch.get_rng_state(), state['torch_rng'], rtol=0, atol=0)

            expected_loss = step(model, connector, opt, batch, bank, mask)
            expected_model = copy.deepcopy(model.state_dict())
            expected_connector = copy.deepcopy(connector.state_dict())
            expected_rng = torch.get_rng_state()
            run.progress.update(update=6, examples=96, best_loss=.1, pending_validation=True)
            # Restore must undo both the trained weights and the consumed RNG.
            run.restore()
            self.assertEqual(run.progress, saved_progress)
            torch.testing.assert_close(torch.get_rng_state(), state['torch_rng'], rtol=0, atol=0)
            repeated_loss = step(model, connector, opt, batch, bank, mask)
            torch.testing.assert_close(repeated_loss, expected_loss, rtol=0, atol=0)
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0)
            for name, tensor in connector.state_dict().items():
                torch.testing.assert_close(tensor, expected_connector[name], rtol=0, atol=0)
            torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
