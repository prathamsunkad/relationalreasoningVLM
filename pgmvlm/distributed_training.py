"""Small synchronous data-parallel helpers for a fixed global question batch.

Every rank must call collectives in the same order. Each local question loss is
already divided by the *global* batch size; gradients must therefore be summed,
not averaged. Base-model weights can remain frozen without wrapping the model
in DistributedDataParallel.
"""
from contextlib import contextmanager
import hashlib
import json

import torch
import torch.distributed as dist


def _distributed(group=None):
    return dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1


def _collective_device(device=None, parameters=(), group=None):
    if 'nccl' in str(dist.get_backend(group)).lower():
        if device is not None:
            result = torch.device(device)
            if result.type != 'cuda':
                raise ValueError('NCCL collectives require a CUDA device')
            return result
        for parameter in parameters:
            if parameter.device.type == 'cuda':
                return parameter.device
        return torch.device('cuda', torch.cuda.current_device())
    return torch.device('cpu')


@torch.no_grad()
def sum_gradients(parameters, *, group=None):
    """SUM dense gradients in one FP32 buffer, retaining the original dtype.

The ordered parameter layout, trainability and gradient-presence masks are
checked collectively first. A mismatch raises on every rank before a gradient
collective can hang or silently combine different parameters. Parameters frozen
on all ranks and parameters unused on all ranks are skipped. Sparse gradients
are unsupported and rejected collectively. No division by world size occurs.
    """
    parameters = tuple(parameters)
    if not _distributed(group):
        return
    world = dist.get_world_size(group)
    device = _collective_device(parameters=parameters, group=group)
    layout = json.dumps([(list(p.shape), str(p.dtype)) for p in parameters], separators=(',', ':'))
    digest = torch.tensor(list(hashlib.sha256(layout.encode()).digest()), dtype=torch.uint8, device=device)
    digests = [torch.empty_like(digest) for _ in range(world)]
    dist.all_gather(digests, digest, group=group)
    if any(not torch.equal(digest, other) for other in digests):
        raise RuntimeError('Distributed parameter layout mismatch; no gradients reduced')
    if not parameters:
        return
    flags = torch.tensor([
        [int(p.requires_grad), int(p.requires_grad and p.grad is not None),
         int(p.requires_grad and p.grad is not None and p.grad.layout != torch.strided)]
        for p in parameters
    ], dtype=torch.int32, device=device)
    dist.all_reduce(flags, op=dist.ReduceOp.SUM, group=group)
    counts = flags.cpu().tolist()
    if any(active not in (0, world) for active, present, sparse in counts):
        raise RuntimeError('Distributed parameter trainability mismatch; no gradients reduced')
    if any(present not in (0, world) for active, present, sparse in counts):
        raise RuntimeError('Distributed gradient presence mismatch; no gradients reduced')
    if any(sparse for active, present, sparse in counts):
        raise RuntimeError('Distributed sparse gradients are unsupported; no gradients reduced')
    active = [p for p, (trainable, present, sparse) in zip(parameters, counts)
              if trainable == world and present == world]
    if not active:
        return
    flat = torch.cat([p.grad.detach().reshape(-1).to(device=device, dtype=torch.float32) for p in active])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
    offset = 0
    for parameter in active:
        count = parameter.numel()
        parameter.grad.copy_(flat[offset:offset + count].view_as(parameter.grad))
        offset += count


@contextmanager
def question_rng(identifier, update, seed, device=None):
    """Use rank-independent per-question dropout while restoring caller RNG.

The seed depends only on question ID, optimizer-update index and experiment
seed. CPU RNG and the selected CUDA device are forked; unrelated CUDA devices
are neither seeded nor modified. Pass the model's local CUDA device explicitly.
    """
    selected = torch.device('cpu' if device is None else device)
    if selected.type not in ('cpu', 'cuda'):
        raise ValueError('question_rng supports CPU and CUDA devices')
    devices = []
    if selected.type == 'cuda':
        devices = [torch.cuda.current_device() if selected.index is None else selected.index]
    encoded = json.dumps([str(identifier), int(update), int(seed)], separators=(',', ':')).encode()
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], 'big') % (2**63 - 1)
    with torch.random.fork_rng(devices=devices):
        # torch.manual_seed also seeds all GPUs, which would modify unforked RNGs.
        torch.random.default_generator.manual_seed(value)
        if devices:
            with torch.cuda.device(devices[0]):
                torch.cuda.manual_seed(value)
        yield


def stop_vote(local_stop, device=None, *, group=None):
    """Return True on every rank if any rank requested a graceful stop."""
    if not _distributed(group):
        return bool(local_stop)
    value = torch.tensor(int(bool(local_stop)), dtype=torch.uint8,
                         device=_collective_device(device=device, group=group))
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=group)
    return bool(value.item())
