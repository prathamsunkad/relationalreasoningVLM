import os
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pgmvlm.distributed import DistributedContext, rank_positions


def gradient_worker(rank, rendezvous, output):
    # Exercise the exact averaging method with a CPU process group, no GPUs/downloads.
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    ctx = object.__new__(DistributedContext)
    ctx.rank, ctx.world, ctx.primary, ctx.device = rank, 2, rank == 0, torch.device('cpu')
    torch.manual_seed(17)
    model = torch.nn.Linear(3, 2)
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)/10
    y = torch.tensor([0, 1, 1, 0])
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = torch.nn.functional.cross_entropy(model(x[rank::2]), y[rank::2])
    loss.backward()
    ctx.gradients(list(model.parameters()))
    gradients = [p.grad.clone() for p in model.parameters()]
    opt.step()
    torch.save(dict(weights=model.state_dict(), gradients=gradients), Path(output)/f'{rank}.pt')
    self_missing = torch.nn.Parameter(torch.ones(2))
    if rank == 0:
        self_missing.grad = torch.tensor([2., 4.])
    ctx.gradients([self_missing])
    assert torch.equal(self_missing.grad, torch.tensor([1., 2.]))
    ctx.close()


class DistributedTests(unittest.TestCase):
    def test_shards_exactly_cover_global_order_without_overlap(self):
        for world in (1, 2, 4):
            indices = [list(rank_positions(16, 48, 2, world, rank)) for rank in range(world)]
            flat = [i for rank in indices for i in rank]
            self.assertEqual(sorted(flat), list(range(16, 48)))
            self.assertEqual(len(flat), len(set(flat)))

    def test_two_rank_gradient_and_adam_update_match_global_batch(self):
        with tempfile.TemporaryDirectory() as d:
            mp.spawn(gradient_worker, args=(str(Path(d)/'rendezvous'), d), nprocs=2, join=True)
            torch.manual_seed(17)
            model = torch.nn.Linear(3, 2)
            opt = torch.optim.AdamW(model.parameters(), lr=0.01)
            x = torch.arange(12, dtype=torch.float32).reshape(4, 3)/10
            loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 1, 0]))
            loss.backward()
            gradients = [p.grad.clone() for p in model.parameters()]
            opt.step()
            for rank in (0, 1):
                actual = torch.load(Path(d)/f'{rank}.pt', weights_only=True)
                for a, b in zip(actual['gradients'], gradients):
                    self.assertTrue(torch.allclose(a, b, atol=1e-7))
                for name, expected in model.state_dict().items():
                    self.assertTrue(torch.allclose(actual['weights'][name], expected, atol=1e-7))


if __name__ == '__main__':
    unittest.main()
