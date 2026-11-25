import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from tools.dispatch_interleave_allocation import contract, dispatch


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plans = {}
        entries = []
        for name in ('epoch6', 'epoch4'):
            plan = dict(pilot_output=str(self.root / name / 'pilot'),
                        training_output=str(self.root / name / 'train'), model=name)
            path = self.root / (name + '.json');path.write_text(json.dumps(plan))
            entries.append(dict(name=name, plan=str(path))); self.plans[name] = plan
        self.campaign = self.root / 'campaign.json'
        self.campaign.write_text(json.dumps(dict(models=entries)))
        self.clock = Clock(); self.calls = []

    def output(self, name, mode='train'):
        return Path(self.plans[name]['pilot_output' if mode == 'pilot' else 'training_output'])

    def write(self, name, mode='train', *, complete=True, update=100, passed=True, checkpoint=False, **extra):
        output = self.output(name, mode);output.mkdir(parents=True, exist_ok=True)
        value = dict(mode=mode, complete=complete, passed=passed, update=update,
                     plan_sha256=contract(self.plans[name]), pending_validation=False, **extra)
        (output / 'summary.json').write_text(json.dumps(value))
        if checkpoint:
            (output / '.checkpoints').mkdir(exist_ok=True)
            (output / '.checkpoints/last.json').write_text('{}')
        return value

    def runner(self, *, mode='train', complete=True, advance=100, duration=100, exit_code=0):
        def run(command, check):
            self.assertFalse(check)
            self.assertIn('--nproc_per_node=4', command)
            name = Path(command[command.index('--plan') + 1]).stem
            self.calls.append(command)
            self.clock.now += duration
            if exit_code == 0:
                self.write(name, mode, complete=complete, update=advance, checkpoint=not complete)
            return SimpleNamespace(returncode=exit_code)
        return run

    def go(self, runner, mode='train', seconds=13800):
        return dispatch(self.campaign, self.root / 'source', seconds, mode, runner=runner, clock=self.clock)

    def test_epoch6_then_epoch4_in_same_allocation_and_time_budget_decreases(self):
        result = self.go(self.runner(duration=4000))
        self.assertTrue(result['complete'])
        self.assertEqual([x['model'] for x in result['invocations']], ['epoch6', 'epoch4'])
        self.assertEqual([c[c.index('--seconds') + 1] for c in self.calls], ['13800', '9800'])
        self.assertEqual(json.loads((self.root / 'campaign-summary.json').read_text())['mode'], 'train')

    def test_incomplete_epoch6_with_advancing_checkpoint_defers_epoch4(self):
        self.write('epoch6', complete=False, update=100, checkpoint=True)
        result = self.go(self.runner(complete=False, advance=200))
        self.assertFalse(result['complete'])
        self.assertEqual(result['status'], 'checkpointed_needs_continuation')
        self.assertEqual(len(self.calls), 1)
        self.assertIn('--resume', self.calls[0])

    def test_complete_matching_epoch6_is_skipped_and_epoch4_runs(self):
        self.write('epoch6', early_stopped=True, stop_reason='validation_plateau')
        result = self.go(self.runner())
        self.assertTrue(result['complete'])
        self.assertEqual([x['model'] for x in result['invocations']], ['epoch4'])

    def test_no_gpu_child_for_already_complete_campaign(self):
        for name in self.plans:self.write(name)
        result = self.go(lambda *_a, **_k: self.fail('unexpected child'))
        self.assertTrue(result['complete'])
        self.assertEqual(result['invocations'], [])

    def test_failed_child_blocks_second_model(self):
        with self.assertRaisesRegex(RuntimeError, 'subprocess exited 7'):
            self.go(self.runner(exit_code=7))
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(json.loads((self.root / 'campaign-summary.json').read_text())['complete'])

    def test_stale_or_wrong_mode_summary_is_rejected(self):
        for altered in [{'plan_sha256': 'wrong'}, {'mode': 'pilot'}]:
            with self.subTest(altered=altered):
                val = self.write('epoch6');val.update(altered)
                (self.output('epoch6')/'summary.json').write_text(json.dumps(val))
                with self.assertRaises(RuntimeError):
                    self.go(lambda *_a, **_k: self.fail('unexpected child'))

    def test_incomplete_pilot_is_a_failed_gate(self):
        with self.assertRaisesRegex(RuntimeError, 'Pilot did not complete'):
            self.go(self.runner(mode='pilot', complete=False, advance=8), mode='pilot', seconds=1500)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(json.loads((self.root/'campaign-summary.json').read_text())['complete'])

    def test_both_pilots_pass_without_leaking_formal_completion(self):
        result = self.go(self.runner(mode='pilot'), mode='pilot', seconds=1500)
        self.assertTrue(result['complete']);self.assertEqual(result['mode'], 'pilot')
        self.assertFalse(self.output('epoch6', 'train').exists())

    def test_no_progress_or_missing_checkpoint_is_rejected(self):
        self.write('epoch6', complete=False, update=100, checkpoint=True)
        with self.assertRaisesRegex(RuntimeError, 'advancing checkpoint'):
            self.go(self.runner(complete=False, advance=100))
        self.assertEqual(len(self.calls), 1)
        (self.output('epoch6')/'.checkpoints/last.json').unlink()
        with self.assertRaisesRegex(RuntimeError, 'no resumable checkpoint'):
            self.go(lambda *_a, **_k: self.fail('unexpected child'))

    def test_rank_failure_even_with_completed_summary_blocks_further_training(self):
        self.write('epoch6')
        (self.output('epoch6')/'failure-rank2.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'recorded failure'):
            self.go(lambda *_a, **_k: self.fail('unexpected child'))

    def test_reserve_defers_next_model_when_less_than_ten_minutes_remain(self):
        result = self.go(self.runner(duration=1000), seconds=1500)
        self.assertFalse(result['complete']);self.assertEqual(len(self.calls), 1)
        self.assertEqual(result['current_model'], 'epoch4')
        self.assertEqual(result['status'], 'allocation_boundary')

    def test_order_is_enforced(self):
        data=json.loads(self.campaign.read_text());data['models'].reverse()
        self.campaign.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'epoch6 then epoch4'):
            self.go(lambda *_a, **_k: self.fail('unexpected child'))


if __name__ == '__main__':
    unittest.main()
