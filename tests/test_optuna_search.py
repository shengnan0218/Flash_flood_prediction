from __future__ import annotations

import inspect
import unittest
from unittest import mock

import optuna

from config import load_config
from optimization import optuna_search
from optimization.optuna_search import (
    ALLOWED_SEARCH_PARAMETERS,
    apply_trial_parameters,
    build_pruner,
    build_sampler,
    report_selection_for_pruning,
    run_optimization,
    trial_output_directory,
)


class TestOptunaSearch(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config("configs/hunan_e4_multitask.yaml")

    def test_tpe_and_median_pruner_configuration(self) -> None:
        sampler = build_sampler(self.config)
        pruner = build_pruner(self.config)
        self.assertIsInstance(sampler, optuna.samplers.TPESampler)
        self.assertIsInstance(pruner, optuna.pruners.MedianPruner)
        self.assertEqual(pruner._n_startup_trials, 5)
        self.assertEqual(pruner._n_warmup_steps, 10)
        self.assertEqual(pruner._interval_steps, 1)
        self.assertEqual(
            self.config["hyperparameter_optimization"]["sampler"]["seed"], 42
        )

    def test_trial_output_directories_are_isolated(self) -> None:
        first = trial_output_directory("outputs/hpo", 0)
        second = trial_output_directory("outputs/hpo", 1)
        self.assertEqual(first.name, "trial_0000")
        self.assertEqual(second.name, "trial_0001")
        self.assertNotEqual(first, second)

    def test_only_six_shared_parameters_can_be_applied(self) -> None:
        values = {
            "learning_rate": 0.001,
            "weight_decay": 1.0e-5,
            "hidden_dim": 64,
            "q_peak_weight": 0.2,
            "q_volume_weight": 0.3,
            "z_slope_weight": 0.4,
        }
        self.assertEqual(set(values), ALLOWED_SEARCH_PARAMETERS)
        updated = apply_trial_parameters(self.config, values)
        self.assertEqual(updated["hidden_dim"], 64)
        self.assertEqual(updated["solver"], self.config["solver"])
        self.assertEqual(updated["physical_bounds"], self.config["physical_bounds"])
        with self.assertRaisesRegex(ValueError, "越权"):
            apply_trial_parameters(self.config, {**values, "cfl": 0.5})

    def test_pruning_reports_selection_score_not_val_loss(self) -> None:
        trial = mock.Mock(number=7)
        trial.should_prune.return_value = False
        score = report_selection_for_pruning(
            trial,
            12,
            {"validation_selection_score": 0.73, "val_loss": 0.01},
        )
        self.assertEqual(score, 0.73)
        trial.report.assert_called_once_with(0.73, step=12)

    def test_pruning_raises_trial_pruned(self) -> None:
        trial = mock.Mock(number=7)
        trial.should_prune.return_value = True
        with self.assertRaises(optuna.TrialPruned):
            report_selection_for_pruning(
                trial, 12, {"validation_selection_score": 0.2}
            )

    def test_disabled_search_refuses_before_loading_any_dataset(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "默认关闭"):
            run_optimization("configs/hunan_e4_multitask.yaml")

    def test_optimization_module_has_no_test_loader_path(self) -> None:
        source = inspect.getsource(optuna_search)
        self.assertNotIn("setup_evaluation", source)
        self.assertNotIn("test_split", source)


if __name__ == "__main__":
    unittest.main()
