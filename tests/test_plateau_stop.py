import copy
import json
import unittest

from pgmvlm.plateau_stop import ConservativePlateau


def observe(stop, step, loss=1., accuracy=.6, **kw):
    return stop.update(update=step, loss=loss, accuracy=accuracy, count=1000, **kw)


class ConservativePlateauTests(unittest.TestCase):
    def test_no_stop_during_warmup_and_six_complete_joint_checks_required(self):
        stop = ConservativePlateau()
        self.assertFalse(observe(stop, 0)["eligible"])
        self.assertFalse(observe(stop, 500)["eligible"])
        for step in range(1000, 4000, 500):
            self.assertFalse(observe(stop, step)["stop"])
        result = observe(stop, 4000)
        self.assertTrue(result["stop"])
        self.assertEqual(result["bad_checks"], 6)
        self.assertEqual(result["reason"], "validation_plateau")

    def test_gradual_small_loss_gains_accumulate_and_prevent_stop(self):
        stop = ConservativePlateau()
        for i, step in enumerate(range(1000, 10000, 500)):
            result = observe(stop, step, loss=1. - i * .0003)
            self.assertFalse(result["stop"])

    def test_accuracy_gain_resets_patience_even_when_loss_is_flat(self):
        stop = ConservativePlateau()
        for step in range(1000, 4000, 500):
            observe(stop, step)
        result = observe(stop, 4000, accuracy=.602)
        self.assertFalse(result["stop"])
        self.assertEqual(result["bad_checks"], 0)

    def test_recovery_toward_old_best_is_not_mistaken_for_plateau(self):
        stop = ConservativePlateau()
        observe(stop, 1000, loss=.8)
        for step, loss in zip(range(1500, 4500, 500), [1.2, 1.15, 1.1, 1.05, 1., .95]):
            result = observe(stop, step, loss=loss)
        self.assertFalse(result["stop"])
        self.assertEqual(result["reason"], "recent_improving_trend")
        for step in range(4500, 8000, 500):
            result = observe(stop, step, loss=.95)
            if result["stop"]:
                break
        self.assertTrue(result["stop"])

    def test_recent_accuracy_recovery_prevents_stop(self):
        stop = ConservativePlateau()
        observe(stop, 1000, accuracy=.7)
        for step, accuracy in zip(range(1500, 4500, 500), [.6, .61, .62, .63, .64, .65]):
            result = observe(stop, step, accuracy=accuracy)
        self.assertFalse(result["stop"])
        self.assertEqual(result["reason"], "recent_improving_trend")

    def test_incomplete_validation_does_not_count_or_change_state(self):
        stop = ConservativePlateau()
        observe(stop, 1000)
        before = stop.state_dict()
        result = stop.update(update=1500, loss=None, accuracy=None, count=34, complete=False)
        self.assertFalse(result["stop"])
        self.assertEqual(stop.state_dict(), before)
        with self.assertRaises(ValueError):
            stop.update(update=1500, loss=1., accuracy=.6, count=999)

    def test_missing_checks_restart_sustained_plateau_window(self):
        stop = ConservativePlateau()
        for step in range(1000, 3500, 500):
            observe(stop, step)
        result = observe(stop, 4500)
        self.assertEqual(result["reason"], "new_window")
        self.assertEqual(result["bad_checks"], 0)
        self.assertFalse(result["stop"])

    def test_resume_matches_uninterrupted_patience_and_decision(self):
        stop = ConservativePlateau()
        for step in range(0, 2500, 500):
            observe(stop, step)
        restored = ConservativePlateau.from_state_dict(json.loads(json.dumps(stop.state_dict())))
        for step in range(2500, 4500, 500):
            self.assertEqual(observe(stop, step), observe(restored, step))
        self.assertTrue(restored.stopped)
        final = ConservativePlateau.from_state_dict(restored.state_dict())
        self.assertEqual(final.state_dict(), restored.state_dict())

    def test_duplicate_is_idempotent_but_conflicting_or_reordered_is_rejected(self):
        stop = ConservativePlateau()
        result = observe(stop, 1000)
        self.assertEqual(observe(stop, 1000), result)
        self.assertEqual(len(stop.observations), 1)
        with self.assertRaises(ValueError): observe(stop, 1000, loss=.9)
        with self.assertRaises(ValueError): observe(stop, 500)

    def test_invalid_metrics_and_inconsistent_resume_are_rejected(self):
        for loss, acc in [(float('nan'), .6), (float('inf'), .6), (-1, .6), (1., 1.1)]:
            with self.assertRaises(ValueError): observe(ConservativePlateau(), 1000, loss, acc)
        stop = ConservativePlateau(); observe(stop, 1000)
        state = copy.deepcopy(stop.state_dict());state['stopped'] = True
        with self.assertRaises(ValueError):ConservativePlateau.from_state_dict(state)

    def test_relative_loss_threshold_and_minimum_joint_updates(self):
        stop = ConservativePlateau(relative_delta=.01, min_joint_updates=5000)
        observe(stop, 1000, loss=3.)
        for step in range(1500, 6000, 500):
            result = observe(stop, step, loss=2.98)
            if step < 5500:self.assertFalse(result['stop'])
        self.assertTrue(result['stop'])


if __name__ == '__main__':
    unittest.main()
