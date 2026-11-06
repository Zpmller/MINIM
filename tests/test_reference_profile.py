import unittest
import numpy as np
from scripts.evaluate_reference_profile import policy_metrics, selection_key
from evaluate_metrics import aggregate_dataset_metrics


class ReferenceProfileTests(unittest.TestCase):
    def test_vectorized_metrics_match_official_aggregation_and_empty_critical_case(self):
        nodes = [{'role': {'value': 'button'}, 'name': {'value': 'a'}},
                 {'role': {'value': 'text'}, 'name': {'value': 'a b c'}}]
        graph = {'gs': np.array([10., 8.]), 'gn': np.array([7., 0.]),
                 'sp': np.array([6., 6.]), 'np': np.array([2., 2.]),
                 'tok': np.array([1., 3.]), 'interactive': np.array([True, False])}
        empty = {**graph, 'gn': np.array([0., 0.]), 'np': np.zeros(2)}
        actual = policy_metrics([graph, empty], 1, 5)
        expected = aggregate_dataset_metrics([(nodes, graph['gs'], graph['gn'], ['A', 'A']),
                                              (nodes, empty['gs'], empty['gn'], ['R', 'R'])])
        for k in ('tcnp', 'tcnp_i', 'tisl'):
            self.assertAlmostEqual(actual[k], expected[k])

    def test_selection_does_not_reward_leaking_more_or_lowering_good_utility(self):
        ref = {'tcnp': .9491, 'tcnp_i': .9931, 'tisl': .101}
        good = {'tcnp': 1., 'tcnp_i': 1., 'tisl': .01}
        close = {**ref}
        self.assertGreater(selection_key(good, ref, 1, 5), selection_key(close, ref, 1, 5))
        all_keep = {'tcnp': 1., 'tcnp_i': 1., 'tisl': 1.}
        self.assertGreater(selection_key(good, ref, 1, 5), selection_key(all_keep, ref, 1, 5))


if __name__ == '__main__':
    unittest.main()
