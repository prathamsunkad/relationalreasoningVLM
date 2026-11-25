"""Four-rank CPU checks of global-batch scaling, dropout and exact replay."""
import copy
import datetime
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pgmvlm.distributed_training import question_rng, stop_vote, sum_gradients


class TinyHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(6, 8).requires_grad_(False)
        self.lora_a = nn.Linear(8, 3, bias=False)
        self.lora_b = nn.Linear(3, 8, bias=False)
        self.connector = nn.Sequential(nn.Linear(5, 8), nn.GELU(), nn.Dropout(.2), nn.Linear(8, 8))
        self.dropout = nn.Dropout(.35)
        self.head = nn.Linear(8, 4).requires_grad_(False)

    def forward(self, image, relations):
        native = self.base(image)
        combined = native + self.lora_b(self.lora_a(self.dropout(native))) + self.connector(relations)
        return self.head(combined)


def _optimizer(model):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.003, weight_decay=.01)


def _backward(model, images, relations, targets, indices, update):
    for index in indices:
        with question_rng('question-' + str(index), update, 731, device='cpu'):
            logits = model(images[index:index+1], relations[index:index+1])
            # The entire distributed batch has16 questions, not4 per rank.
            (nn.functional.cross_entropy(logits, targets[index:index+1]) / 16).backward()


def _expect_collective_error(call, text):
    try:
        call()
    except RuntimeError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Expected collective error: ' + text)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4,
                            timeout=datetime.timedelta(seconds=90))
    try:
        torch.manual_seed(115)
        model = TinyHybrid()
        serial = copy.deepcopy(model)
        optimizer, serial_optimizer = _optimizer(model), _optimizer(serial)
        generator = torch.Generator().manual_seed(991)
        images = torch.randn(16, 6, generator=generator)
        relations = torch.randn(16, 5, generator=generator)
        targets = torch.randint(4, (16,), generator=generator)
        frozen = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        initial_rng = torch.get_rng_state().clone()

        _backward(model, images, relations, targets, range(rank, 16, 4), update=0)
        _backward(serial, images, relations, targets, range(16), update=0)
        assert torch.equal(torch.get_rng_state(), initial_rng), 'Question dropout consumed caller RNG'
        sum_gradients(model.parameters())
        for (name,p), (serial_name,q) in zip(model.named_parameters(), serial.named_parameters()):
            assert name == serial_name
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-6, atol=3e-8)
            else:
                assert p.grad is None and q.grad is None
        optimizer.step(); serial_optimizer.step()
        for p,q in zip(model.parameters(), serial.parameters()):
            torch.testing.assert_close(p, q, rtol=2e-6, atol=3e-8)
        for name,p in model.named_parameters():
            if name in frozen:
                assert torch.equal(p, frozen[name]), 'Frozen parameter changed'

        # Replay a real second AdamW update from weights/optimizer/RNG state.
        saved_model, saved_optimizer = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        saved_rng = torch.get_rng_state().clone()
        optimizer.zero_grad(set_to_none=True)
        _backward(model, images, relations, targets, range(rank,16,4), update=1)
        sum_gradients(model.parameters()); optimizer.step()
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved_model); optimizer.load_state_dict(saved_optimizer)
        torch.set_rng_state(saved_rng)
        optimizer.zero_grad(set_to_none=True)
        _backward(model, images, relations, targets, range(rank,16,4), update=1)
        sum_gradients(model.parameters()); optimizer.step()
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert torch.equal(torch.get_rng_state(), saved_rng)

        # Explicitly prove SUM, including unused/frozen parameter behavior.
        active, unused, frozen_parameter = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(2), requires_grad=False)
        active.grad = torch.full_like(active, rank+1.)
        frozen_parameter.grad = torch.full_like(frozen_parameter, 19.)
        sum_gradients([active, unused, frozen_parameter])
        assert torch.equal(active.grad, torch.full_like(active, 10.))
        assert unused.grad is None
        assert torch.equal(frozen_parameter.grad, torch.full_like(frozen_parameter, 19.))
        assert stop_vote(rank == 3) is True
        assert stop_vote(False) is False

        mismatch = nn.Parameter(torch.ones(2))
        if rank != 0:
            mismatch.grad = torch.ones_like(mismatch)
        _expect_collective_error(lambda:sum_gradients([mismatch]), 'presence')
        mismatch.requires_grad_(rank != 0)
        _expect_collective_error(lambda:sum_gradients([mismatch]), 'trainability')
        _expect_collective_error(lambda:sum_gradients([nn.Parameter(torch.ones(2 if rank else 3))]), 'layout')
        sparse = nn.Parameter(torch.ones(2))
        sparse.grad = torch.sparse_coo_tensor(torch.tensor([[0,1]]), torch.tensor([1.,2.]), (2,)) if rank == 0 else torch.ones(2)
        _expect_collective_error(lambda:sum_gradients([sparse]), 'sparse')
        sum_gradients([])
        sum_gradients([nn.Parameter(torch.ones(1))])
    finally:
        dist.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_four_rank_global_batch_gradient_update_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(str(Path(directory)/'gloo-rendezvous'),), nprocs=4, join=True)

    def test_question_rng_restores_state_and_distinguishes_updates(self):
        torch.manual_seed(91)
        state = torch.get_rng_state().clone()
        with question_rng('id', 0, 22):
            first = torch.rand(20)
        self.assertTrue(torch.equal(torch.get_rng_state(), state))
        torch.rand(17)
        later = torch.get_rng_state().clone()
        with question_rng('id', 0, 22):
            repeated = torch.rand(20)
        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.equal(torch.get_rng_state(), later))
        with question_rng('id', 1, 22):
            next_update = torch.rand(20)
        self.assertFalse(torch.equal(first, next_update))
        self.assertTrue(torch.equal(torch.get_rng_state(), later))

    def test_question_rng_restores_state_after_exception(self):
        state = torch.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'intentional'):
            with question_rng('id', 0, 22):
                torch.rand(20)
                raise RuntimeError('intentional')
        self.assertTrue(torch.equal(torch.get_rng_state(), state))

    def test_non_distributed_calls_are_noops(self):
        parameter = nn.Parameter(torch.ones(3))
        parameter.grad = torch.full_like(parameter, 7.)
        sum_gradients([parameter])
        self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 7.)))
        self.assertTrue(stop_vote(True))
        self.assertFalse(stop_vote(False))


if __name__ == '__main__':
    unittest.main()
