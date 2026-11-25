"""Scientific aggregation and exact-ID matching of the inference-only control."""
import copy
import unittest
import numpy as np
from tools.evaluate_internvl_average import log_softmax, summarize


class AverageControlTests(unittest.TestCase):
    def fixture(self):
        records = [dict(id='a', target=0, lora_scores=[1, 2, 0, 0, 0, 0, 0, 0],
                        abstractor_scores=[5, 0, 0, 0, 0, 0, 0, 0]),
                   dict(id='b', target=1, lora_scores=[0, 4, 0, 0, 0, 0, 0, 0],
                        abstractor_scores=[2, 1, 0, 0, 0, 0, 0, 0])]
        hybrid = dict(complete=True, count=2, update=3500, predictions=[
            dict(id='b', target=1, scores=[0, 3, 0, 0, 0, 0, 0, 0]),
            dict(id='a', target=0, scores=[3, 0, 0, 0, 0, 0, 0, 0])])
        return records, hybrid

    def test_matches_repo_rule_and_probability_mixture(self):
        records, hybrid = self.fixture()
        result = summarize(records, hybrid)
        self.assertEqual(result['metrics']['lora']['accuracy'], .5)
        self.assertEqual(result['metrics']['abstractor']['accuracy'], .5)
        self.assertEqual(result['metrics']['average_log_probability']['accuracy'], 1.)
        self.assertEqual(result['metrics']['hybrid']['accuracy'], 1.)
        a = log_softmax([r['lora_scores'] for r in records])
        b = log_softmax([r['abstractor_scores'] for r in records])
        mixture = (.5 * np.exp(a) + .5 * np.exp(b))
        expected = -np.log(mixture[[0, 1], [0, 1]]).mean()
        self.assertAlmostEqual(result['metrics']['average_probability']['loss'], expected)
        self.assertEqual(result['overlap_lora_abstractor']['both_correct'], 0)
        self.assertEqual(result['overlap_lora_abstractor']['neither_correct'], 0)

    def test_refuses_partial_duplicate_mismatched_or_invalid_predictions(self):
        records, hybrid = self.fixture()
        for change in ('partial', 'duplicate', 'wrong_id', 'wrong_label', 'nan'):
            a, b = copy.deepcopy(records), copy.deepcopy(hybrid)
            if change == 'partial': b['complete'] = False
            if change == 'duplicate': b['predictions'][1]['id'] = 'b'
            if change == 'wrong_id': b['predictions'][1]['id'] = 'other'
            if change == 'wrong_label': b['predictions'][1]['target'] = 3
            if change == 'nan': a[0]['lora_scores'][0] = float('nan')
            with self.subTest(change=change), self.assertRaises(ValueError):
                summarize(a, b)

    def test_normalization_is_stable_but_does_not_claim_calibration(self):
        scores = [[10000, 9999, -10000, 1, 2, 3, 4, 5]]
        a = log_softmax(scores)
        b = log_softmax(np.asarray(scores) + 100000)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(np.exp(a).sum()), 1.)
        result = summarize(*self.fixture())
        self.assertFalse(result['calibrated_temperatures'])
        self.assertFalse(result['tuned_weights'])


if __name__ == '__main__':
    unittest.main()
