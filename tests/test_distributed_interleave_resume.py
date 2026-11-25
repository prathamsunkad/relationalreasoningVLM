"""Production four-rank step and disk resume on native tiny InternVL + PEFT."""
import copy
from datetime import timedelta
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from test_cached_interleave import CachedInterleave, optimizer, score, setup, train_lora
from pgmvlm.extensions.probe_diagnostics import tensor_digest
from pgmvlm.plateau_stop import ConservativePlateau
from tools import train_distributed_interleave as trainer


def _worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=timedelta(seconds=120))
    connector = None
    try:
        model, batch, bank, mask, backbone = setup()
        train_lora(model)
        connector = CachedInterleave(backbone, input_dim=32)
        opt = optimizer(model, connector)
        for group, kind in zip(opt.param_groups, ['connector', 'lora']):
            group['kind'] = kind
        run = trainer.DistributedRun.__new__(trainer.DistributedRun)
        run.rank, run.world, run.device = rank, 4, torch.device('cpu')
        run.model, run.connector, run.opt = model, connector, opt
        run.out = Path(directory)
        run.plan = {'initialization_seed': 923}
        run.identity = trainer.contract({'test':'production-distributed-native-internvl','world_size':4})
        run.train = [{'id':str(i)} for i in range(96)]
        run.banks = SimpleNamespace(extracted=0, flush=Mock())
        run.lora = [(n,p) for n,p in model.named_parameters() if 'lora_' in n]
        run.frozen = [(n,p) for n,p in model.named_parameters() if 'lora_' not in n]
        frozen_digest = tensor_digest(run.frozen)
        run.initial_lora_hash = tensor_digest(run.lora)
        run.all_params = list(connector.parameters()) + [p for _,p in run.lora]
        run.total, run.frozen_steps, run.warmup = 6, 1, 1
        run.pilot_checks = {}
        run.progress = dict(update=0, examples=0, best_loss=1.9, best_checkpoint=None,
                            pending_validation=False, early_stopped=False)
        run.stopper = ConservativePlateau(warmup_updates=0, interval=1, patience=6,
                                         min_joint_updates=0, expected_count=32)
        run.stopper.update(update=0, loss=2., accuracy=.25, count=32)

        def scores(row):
            index = int(row['id'])
            varied_bank = bank + .001 * index
            with connector.question(varied_bank, mask):
                output = score(model, batch)
            return output, torch.tensor([index % 8])
        run.scores = scores
        run.assert_synchronized()
        first = run.step(0)
        assert first['phase'] == 'connector_only'
        assert tensor_digest(run.lora) == run.initial_lora_hash
        second = run.step(1, verify_serial=True)
        assert second['phase'] == 'joint'
        assert run.pilot_checks['global_batch16_serial_equivalence'] is True
        assert tensor_digest(run.lora) != run.initial_lora_hash
        run.assert_synchronized()
        run.progress.update(update=2, examples=32)
        run.stopper.update(update=1, loss=1.9, accuracy=.3, count=32)
        run.stopper.update(update=2, loss=1.9, accuracy=.3, count=32)

        # Different ambient RNG per rank proves that rank-specific state restores.
        torch.rand(rank+1)
        fake_cuda_rng = torch.arange(16, dtype=torch.uint8) + rank
        original_cuda_rng = fake_cuda_rng.clone()
        original_cpu_rng = torch.get_rng_state().clone()
        with patch.object(trainer, 'require_headroom'), \
                patch.object(torch.cuda, 'get_rng_state', side_effect=lambda device:fake_cuda_rng.clone()), \
                patch.object(torch.cuda, 'set_rng_state', side_effect=lambda value,device:fake_cuda_rng.copy_(value)):
            best = run.save('best')
            assert best
            run.progress['best_checkpoint'] = best
            saved_progress = copy.deepcopy(run.progress)
            saved_stopper = copy.deepcopy(run.stopper.state_dict())
            run.save()
            if rank == 0:
                saved = torch.load(run.out/'state.pt', map_location='cpu', weights_only=True)
                assert saved['step'] == 2 and saved['world_size'] == 4
                assert len(saved['rng_by_rank']) == 4
                assert len({bytes(s['cpu'].tolist()) for s in saved['rng_by_rank']}) == 4
                assert saved['stopper'] == saved_stopper
            expected_loss = run.step(2)['loss']
            expected_model = copy.deepcopy(model.state_dict())
            expected_connector = copy.deepcopy(connector.state_dict())
            run.progress.update(update=3, examples=48, best_loss=.2, early_stopped=True)
            run.stopper.update(update=3, loss=.2, accuracy=.8, count=32)
            torch.rand(37)
            fake_cuda_rng.zero_()
            run.restore()
            assert run.progress == saved_progress
            assert run.stopper.state_dict() == saved_stopper
            assert torch.equal(torch.get_rng_state(), original_cpu_rng)
            assert torch.equal(fake_cuda_rng, original_cuda_rng)
            repeated_loss = run.step(2)['loss']
            assert repeated_loss == expected_loss
            for name,tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0)
            for name,tensor in connector.state_dict().items():
                torch.testing.assert_close(tensor, expected_connector[name], rtol=0, atol=0)
            assert tensor_digest(run.frozen) == frozen_digest
            assert all(p.grad is None and not p.requires_grad for _,p in run.frozen)
            run.assert_synchronized()
            assert run.banks.flush.call_count == 2
    finally:
        if connector is not None:
            connector.remove()
        dist.destroy_process_group()


class DistributedInterleaveResumeTests(unittest.TestCase):
    def test_production_step_disk_checkpoint_rank_rng_and_stopper_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(str(Path(directory)/'gloo-rendezvous'), directory), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
