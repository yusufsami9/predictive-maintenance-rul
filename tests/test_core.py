import itertools
import unittest

import numpy as np
import pandas as pd

from src.evaluation import nasa_score
from src.scheduling import (
    evaluate_schedule,
    expected_costs,
    point_estimate_costs,
    solve_schedule,
)
from src.sequences import create_sequences


class EvaluationTests(unittest.TestCase):
    def test_nasa_score_uses_asymmetric_penalties(self):
        truth = np.array([10.0, 10.0])
        pred = np.array([0.0, 20.0])
        expected = np.expm1(10 / 13) + np.expm1(10 / 10)
        self.assertAlmostEqual(nasa_score(truth, pred), expected)


class SequenceTests(unittest.TestCase):
    def test_windows_do_not_cross_engine_boundaries(self):
        df = pd.DataFrame(
            {
                "unit": [1, 1, 1, 2, 2, 2],
                "sensor": [10, 11, 12, 20, 21, 22],
                "RUL": [2, 1, 0, 2, 1, 0],
            }
        )
        X, y = create_sequences(df, ["sensor"], window=2)
        np.testing.assert_array_equal(
            X[:, :, 0],
            np.array([[10, 11], [11, 12], [20, 21], [21, 22]]),
        )
        np.testing.assert_array_equal(y, np.array([1, 0, 1, 0]))


class SchedulingCostTests(unittest.TestCase):
    def test_point_cost_matrix_matches_realized_evaluator(self):
        truth = np.array([2.0, 5.0, 8.0])
        horizon = 5
        cost, defer = point_estimate_costs(truth, horizon)

        for i in range(len(truth)):
            for period in range(1, horizon + 1):
                realized = evaluate_schedule({i: period}, truth, horizon)["total_cost"]
                self.assertEqual(cost[i, period], realized)
            realized_defer = evaluate_schedule({i: None}, truth, horizon)["total_cost"]
            self.assertEqual(defer[i], realized_defer)

    def test_expected_cost_is_mean_of_same_realized_cost(self):
        pred = np.array([4.0, 9.0])
        errors = np.array([-2.0, 0.0, 3.0])
        horizon = 5
        cost, defer = expected_costs(pred, errors, horizon)
        possible_truth = pred[:, None] - errors[None, :]

        for i in range(len(pred)):
            for period in range(1, horizon + 1):
                scenario_costs = [
                    evaluate_schedule({0: period}, np.array([truth]), horizon)["total_cost"]
                    for truth in possible_truth[i]
                ]
                self.assertEqual(cost[i, period], round(np.mean(scenario_costs)))
            scenario_defer = [
                evaluate_schedule({0: None}, np.array([truth]), horizon)["total_cost"]
                for truth in possible_truth[i]
            ]
            self.assertEqual(defer[i], round(np.mean(scenario_defer)))

    def test_expected_cost_rejects_empty_calibration_sample(self):
        with self.assertRaises(ValueError):
            expected_costs(np.array([5.0]), np.array([]), horizon=3)


class SolverTests(unittest.TestCase):
    @staticmethod
    def _priced_cost(schedule, cost, defer):
        return sum(
            defer[i] if period is None else cost[i, period]
            for i, period in schedule.items()
        )

    def test_solver_is_feasible_deterministic_and_primary_optimal(self):
        cost = np.array(
            [
                [0, 2, 4, 100],
                [0, 1, 3, 100],
                [0, 5, 2, 1],
            ]
        )
        defer = np.array([100, 100, 0])
        horizon, capacity = 3, 1

        schedules = [solve_schedule(cost, defer, horizon, capacity) for _ in range(5)]
        self.assertTrue(all(schedule == schedules[0] for schedule in schedules[1:]))
        schedule = schedules[0]

        assigned = [period for period in schedule.values() if period is not None]
        self.assertTrue(all(1 <= period <= horizon for period in assigned))
        self.assertTrue(all(assigned.count(period) <= capacity for period in set(assigned)))

        choices = [None, *range(1, horizon + 1)]
        feasible_costs = []
        for assignment in itertools.product(choices, repeat=len(cost)):
            if all(assignment.count(period) <= capacity for period in range(1, horizon + 1)):
                candidate = dict(enumerate(assignment))
                feasible_costs.append(self._priced_cost(candidate, cost, defer))

        self.assertEqual(self._priced_cost(schedule, cost, defer), min(feasible_costs))


if __name__ == "__main__":
    unittest.main()
