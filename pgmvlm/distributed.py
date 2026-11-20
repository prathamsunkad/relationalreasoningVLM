"""Optional torchrun support for the existing serial model/optimizer loop.

Explicit gradient averaging supports the native LFM extension hooks without
wrapping their model in a second forward dispatcher. Single-process is a no-op.
"""
from datetime import timedelta
import os

import torch
import torch.distributed as dist


def rank_positions(start, end, batch_size, world_size, rank):
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid rank topology')
    for group in range(start, end, batch_size * world_size):
        yield from range(group + rank*batch_size, min(group + (rank+1)*batch_size, end))


class DistributedContext:
    def __init__(self):
        self.rank = int(os.environ.get('RANK', 0))
        self.world = int(os.environ.get('WORLD_SIZE', 1))
        local = int(os.environ.get('LOCAL_RANK', 0))
        self.primary = self.rank == 0
        self.device = torch.device(f'cuda:{local}' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group('nccl' if self.device.type == 'cuda' else 'gloo', timeout=timedelta(minutes=60))

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def any(self, value):
        if self.world == 1:
            return bool(value)
        x = torch.tensor(int(value), device=self.device)
        dist.all_reduce(x, op=dist.ReduceOp.MAX)
        return bool(x.item())

    def gradients(self, parameters):
        if self.world == 1:
            return
        present = torch.tensor([int(p.grad is not None) for p in parameters], device=self.device)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        for count, p in zip(present.tolist(), parameters):
            if not count:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(self.world)

    def metrics(self, values):
        if self.world == 1:
            return values
        keys = sorted(values)
        x = torch.tensor([values[k] for k in keys], dtype=torch.float64, device=self.device)
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return dict(zip(keys, (x/self.world).tolist()))

    def gather(self, value):
        if self.world == 1:
            return [value]
        values = [None] * self.world if self.primary else None
        dist.gather_object(value, values, dst=0)
        return values

    def close(self):
        if self.world > 1:
            dist.destroy_process_group()
