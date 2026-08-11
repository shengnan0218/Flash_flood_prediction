from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from datasets.hunan import event_graph_balancing_weights
from losses import FloodMultitaskLoss, water_level_first_differences


def config(*, mode: str = "multitask") -> dict:
    return {
        "loss_weights": {"discharge": 2.0, "water_level": 1.0},
        "loss": {
            "mode": mode,
            "q_scale_mode": "global",
            "q_scale_floor_m3s": 1.0,
            "discharge_weight": 2.0,
            "water_level_weight": 1.0,
            "q_point_weight": 1.0,
            "q_peak_weight": 0.25,
            "q_volume_weight": 0.25,
            "z_level_weight": 1.0,
            "z_slope_weight": 0.25,
        },
        "_runtime": {
            "loss_scales": {"discharge": 2.0, "water_level": 4.0}
        },
    }


def batch() -> SimpleNamespace:
    q_target = torch.tensor([[[0.0], [999.0]], [[0.0], [0.0]]])
    q_mask = torch.tensor([[[True], [False]], [[True], [True]]])
    z_target = torch.tensor([[[4.0], [6.0]], [[4.0], [6.0]]])
    z_mask = torch.ones_like(z_target, dtype=torch.bool)
    return SimpleNamespace(
        q_target=q_target,
        q_target_mask=q_mask,
        z_target=z_target,
        z_target_mask=z_mask,
        z_history=torch.tensor([[[1.0], [2.0]], [[1.0], [2.0]]]),
        z_mask=torch.ones(2, 2, 1, dtype=torch.bool),
        sample_weight=torch.tensor([1.0, 3.0]),
    )


class TestEventGraphBalancing(unittest.TestCase):
    def test_graph_event_window_total_weights_are_equal(self) -> None:
        graphs = ["G1"] * 6 + ["G2"] * 4
        events = ["E1"] + ["E2"] * 5 + ["E3"] * 2 + ["E4"] * 2
        weights = event_graph_balancing_weights(graphs, events)
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
        graph_totals = {
            graph: sum(weight for weight, item in zip(weights, graphs) if item == graph)
            for graph in set(graphs)
        }
        self.assertAlmostEqual(graph_totals["G1"], graph_totals["G2"])
        event_totals = {
            event: sum(weight for weight, item in zip(weights, events) if item == event)
            for event in set(events)
        }
        self.assertAlmostEqual(event_totals["E1"], event_totals["E2"])
        self.assertAlmostEqual(event_totals["E3"], event_totals["E4"])
        self.assertAlmostEqual(weights[1], weights[0] / 5.0)

    def test_event_id_cannot_cross_graphs(self) -> None:
        with self.assertRaisesRegex(ValueError, "跨越多个GRAPH_ID"):
            event_graph_balancing_weights(["G1", "G2"], ["E1", "E1"])


class TestMultitaskLoss(unittest.TestCase):
    def test_legacy_matches_original_normalized_huber(self) -> None:
        item = batch()
        prediction = torch.zeros_like(item.q_target, requires_grad=True)
        engine = FloodMultitaskLoss(config(mode="legacy"))
        stats = engine.batch_statistics({"q": prediction, "z": prediction}, item)
        loss = engine.combine(stats)
        # Q valid errors are zero. Z errors 1 and 1.5 in standardised space:
        # Huber means (0.5 + 1.0 + 0.5 + 1.0) / 4 = 0.75.
        self.assertAlmostEqual(loss.item(), 0.75)

    def test_q_point_peak_volume_are_masked_and_sample_weighted(self) -> None:
        item = batch()
        q_prediction = torch.tensor(
            [[[2.0], [1.0e9]], [[2.0], [2.0]]], requires_grad=True
        )
        z_prediction = item.z_target.clone().requires_grad_()
        engine = FloodMultitaskLoss(config())
        stats = engine.batch_statistics(
            {"q": q_prediction, "z": z_prediction}, item
        )
        report = engine.report(
            {
                name: (float(term.numerator.detach()), term.denominator)
                for name, term in stats.items()
            },
            q_valid_count=3,
            z_valid_count=4,
        )
        # Each sample has standardised Q error 1.  Mean-one sample weights are
        # deliberately [1,3], so (1*value + 3*value)/2 doubles each mean.
        self.assertAlmostEqual(float(report["q_point_loss"]), 1.0)
        self.assertAlmostEqual(float(report["q_peak_loss"]), 2.0)
        self.assertAlmostEqual(float(report["q_volume_loss"]), 2.0)
        self.assertTrue(torch.isfinite(engine.combine(stats)))
        engine.combine(stats).backward()
        self.assertTrue(torch.isfinite(q_prediction.grad).all())
        self.assertTrue(torch.isfinite(z_prediction.grad).all())

    def test_per_graph_q_scale_changes_only_normalized_supervision(self) -> None:
        losses: dict[str, float] = {}
        for graph_id in ("G_SMALL", "G_LARGE"):
            item = batch()
            item.graph_id = (graph_id, graph_id)
            q_prediction = torch.where(
                item.q_target_mask,
                item.q_target + 2.0,
                torch.zeros_like(item.q_target),
            ).requires_grad_()
            physical_prediction = q_prediction.detach().clone()
            cfg = config()
            cfg["loss"]["q_scale_mode"] = "per_graph"
            cfg["_runtime"]["loss_scales"]["discharge_by_graph"] = {
                "G_SMALL": 1.0,
                "G_LARGE": 10.0,
            }
            statistics = FloodMultitaskLoss(cfg).batch_statistics(
                {"q": q_prediction, "z": item.z_target}, item
            )
            losses[graph_id] = float(
                statistics["q_point"].numerator.detach()
                / statistics["q_point"].denominator
            )
            # Loss scaling must not mutate physical m3/s predictions.
            torch.testing.assert_close(q_prediction.detach(), physical_prediction)
        self.assertGreater(losses["G_SMALL"], losses["G_LARGE"])

    def test_per_graph_q_scale_never_falls_back_to_global(self) -> None:
        item = batch()
        item.graph_id = ("G_MISSING", "G_MISSING")
        cfg = config()
        cfg["loss"]["q_scale_mode"] = "per_graph"
        cfg["_runtime"]["loss_scales"]["discharge_by_graph"] = {}
        with self.assertRaisesRegex(ValueError, "G_MISSING.*禁止回退"):
            FloodMultitaskLoss(cfg).batch_statistics(
                {"q": item.q_target, "z": item.z_target}, item
            )

    def test_water_level_only_graph_does_not_require_unused_q_scale(self) -> None:
        item = batch()
        item.q_target_mask[:] = False
        item.graph_id = ("G_LEVEL", "G_LEVEL")
        cfg = config()
        cfg["loss"]["q_scale_mode"] = "per_graph"
        cfg["_runtime"]["loss_scales"]["discharge_by_graph"] = {
            "G_FLOW": 1.0
        }
        statistics = FloodMultitaskLoss(cfg).batch_statistics(
            {"q": item.q_target, "z": item.z_target}, item
        )
        self.assertEqual(statistics["q_point"].denominator, 0)

    def test_q_components_ignore_a_sample_with_no_valid_q_without_nan(self) -> None:
        item = batch()
        item.q_target_mask[0] = False
        q_prediction = torch.tensor(
            [[[float("inf")], [float("inf")]], [[2.0], [2.0]]],
            requires_grad=True,
        )
        z_prediction = item.z_target.clone().requires_grad_()
        engine = FloodMultitaskLoss(config())
        statistics = engine.batch_statistics(
            {"q": q_prediction, "z": z_prediction}, item
        )
        loss = engine.combine(statistics)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(q_prediction.grad).all())

    def test_z_level_and_first_difference_components(self) -> None:
        item = batch()
        q_prediction = item.q_target.clone().requires_grad_()
        z_prediction = torch.tensor(
            [[[5.0], [7.0]], [[5.0], [7.0]]], requires_grad=True
        )
        stats = FloodMultitaskLoss(config()).batch_statistics(
            {"q": q_prediction, "z": z_prediction}, item
        )
        self.assertEqual(stats["z_level"].denominator, 4)
        self.assertEqual(stats["z_slope"].denominator, 4)
        self.assertGreater(float(stats["z_level"].numerator.detach()), 0.0)
        self.assertGreater(float(stats["z_slope"].numerator.detach()), 0.0)

    def test_first_hour_uses_latest_valid_history(self) -> None:
        prediction = torch.tensor([[[5.0], [7.0], [8.0]]])
        target = torch.tensor([[[4.0], [6.0], [9.0]]])
        target_mask = torch.ones_like(target, dtype=torch.bool)
        history = torch.tensor([[[1.0], [2.0], [999.0]]])
        history_mask = torch.tensor([[[True], [True], [False]]])
        pred_diff, obs_diff, mask = water_level_first_differences(
            prediction, target, target_mask, history, history_mask
        )
        self.assertTrue(torch.equal(mask, torch.ones_like(mask)))
        self.assertTrue(torch.equal(pred_diff, torch.tensor([[[3.0], [2.0], [1.0]]])))
        self.assertTrue(torch.equal(obs_diff, torch.tensor([[[2.0], [2.0], [3.0]]])))

    def test_slope_masks_missing_history_and_adjacent_targets(self) -> None:
        prediction = torch.tensor([[[5.0], [7.0], [8.0]]])
        target = torch.tensor([[[4.0], [6.0], [9.0]]])
        target_mask = torch.tensor([[[True], [False], [True]]])
        history = torch.zeros(1, 3, 1)
        history_mask = torch.zeros_like(history, dtype=torch.bool)
        _, _, mask = water_level_first_differences(
            prediction, target, target_mask, history, history_mask
        )
        self.assertTrue(torch.equal(mask, torch.zeros_like(mask)))


if __name__ == "__main__":
    unittest.main()
