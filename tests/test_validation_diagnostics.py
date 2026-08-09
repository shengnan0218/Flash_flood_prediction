from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from metrics.flood_metrics import (
    masked_regression_sums,
    regression_metric_status,
    regression_metrics,
)
from metrics.validation_diagnostics import (
    ForecastPoint,
    ValidationDiagnosticsAccumulator,
    deduplicate_shortest_lead,
    latest_history_baseline,
)


START = datetime(2024, 6, 1, 0)


def point(
    *,
    graph: str,
    event: str,
    station: str,
    hour: int,
    observed: float,
    predicted: float,
    lead: int = 1,
    sample: str | None = None,
    variable: str = "Q",
    baseline: float | None = None,
) -> ForecastPoint:
    target_time = START + timedelta(hours=hour)
    forecast_time = target_time - timedelta(hours=lead)
    return ForecastPoint(
        variable=variable,
        graph_id=graph,
        event_id=event,
        sample_id=sample or f"{event}_{hour}_{lead}",
        station_id=station,
        forecast_time=forecast_time,
        target_time=target_time,
        lead_hours=lead,
        observed=observed,
        predicted=predicted,
        event_rain_start="2024-05-31 20:00:00",
        event_rain_end="2024-06-01 03:00:00",
        event_hydro_start="2024-06-01 00:00:00",
        event_hydro_end="2024-06-01 06:00:00",
        event_peak_time="2024-06-01 02:00:00",
        event_sample_start="2024-05-31 00:00:00",
        event_sample_end="2024-06-02 00:00:00",
        baseline_value=baseline,
        baseline_time=forecast_time if baseline is not None else None,
    )


class TestValidationDiagnostics(unittest.TestCase):
    def test_graph_event_peak_volume_sse_and_shortest_lead_dedup(self) -> None:
        accumulator = ValidationDiagnosticsAccumulator()
        accumulator.q_points = [
            point(graph="G1", event="E1", station="S1", hour=1, observed=1, predicted=1),
            # Same real target hour: lead 1 must win over the deliberately bad lead 2.
            point(
                graph="G1", event="E1", station="S1", hour=2,
                observed=4, predicted=99, lead=2,
            ),
            point(graph="G1", event="E1", station="S1", hour=2, observed=4, predicted=2, lead=1),
            point(graph="G1", event="E1", station="S1", hour=3, observed=2, predicted=5),
            point(graph="G2", event="E2", station="S2", hour=1, observed=10, predicted=20),
            point(graph="G2", event="E2", station="S2", hour=2, observed=20, predicted=40),
        ]
        accumulator.event_sample_ids = {
            ("G1", "E1"): {"a", "b", "c"},
            ("G2", "E2"): {"d", "e"},
        }

        selected = deduplicate_shortest_lead(accumulator.q_points)
        retained = next(
            item
            for item in selected
            if item.event_id == "E1" and item.target_time.hour == 2
        )
        self.assertEqual(retained.lead_hours, 1)
        self.assertEqual(retained.predicted, 2)
        self.assertEqual(retained.candidate_count, 2)

        diagnostics = accumulator.finalize()
        event = next(row for row in diagnostics.q_by_event if row["EVENT_ID"] == "E1")
        self.assertEqual(event["valid_q_count"], 3)
        self.assertEqual(event["raw_valid_q_forecast_point_count"], 4)
        self.assertEqual(event["peak_obs"], 4)
        self.assertEqual(event["peak_pred"], 5)
        self.assertEqual(event["peak_error"], 1)
        self.assertEqual(event["peak_timing_error_hours"], 1)
        self.assertEqual(event["observed_volume"], 7 * 3600)
        self.assertEqual(event["predicted_volume"], 8 * 3600)
        self.assertEqual(event["q_sse"], 13)
        self.assertEqual(len(diagnostics.q_by_graph), 2)
        self.assertAlmostEqual(diagnostics.summary["total_q_sse"], 513)
        self.assertAlmostEqual(
            diagnostics.summary["top_1_event_sse_fraction"], 500 / 513
        )
        self.assertAlmostEqual(
            diagnostics.q_top20_sse_events[0]["cumulative_sse_fraction"],
            500 / 513,
        )

    def test_zero_variance_nse_kge_are_nan_with_reasons(self) -> None:
        target = torch.tensor([2.0, 2.0, 2.0])
        prediction = torch.tensor([1.0, 2.0, 3.0])
        mask = torch.ones_like(target, dtype=torch.bool)
        sums = masked_regression_sums(prediction, target, mask)
        metrics = regression_metrics(sums)
        status = regression_metric_status(sums)

        self.assertTrue(torch.isnan(torch.tensor(metrics["nse"])))
        self.assertTrue(torch.isnan(torch.tensor(metrics["kge"])))
        self.assertEqual(status["nse"], "ZERO_OBS_VARIANCE")
        self.assertEqual(status["kge"], "ZERO_OBS_VARIANCE")

    def test_station_z_and_delta_z_use_causal_history_baseline(self) -> None:
        accumulator = ValidationDiagnosticsAccumulator()
        accumulator.z_points = [
            point(
                graph="G1", event="E1", station="ZS", hour=1,
                observed=101, predicted=100.5, variable="Z", baseline=100,
            ),
            point(
                graph="G1", event="E1", station="ZS", hour=2,
                observed=103, predicted=102, variable="Z", baseline=101,
            ),
            point(
                graph="G1", event="E1", station="ZS", hour=3,
                observed=102, predicted=102.5, variable="Z", baseline=101.5,
            ),
        ]
        diagnostics = accumulator.finalize()

        self.assertEqual(len(diagnostics.z_by_station), 1)
        self.assertEqual(diagnostics.z_by_station[0]["station_id"], "ZS")
        self.assertEqual(len(diagnostics.delta_z_by_station), 1)
        self.assertAlmostEqual(
            diagnostics.summary_metrics["delta_z_station_mae_median"],
            (0.5 + 1.0 + 0.5) / 3,
        )

        forecast_time = datetime(2024, 6, 1, 10)
        history = torch.tensor([[1.0], [2.0], [999.0]])
        history_mask = torch.tensor([[True], [True], [False]])
        baseline = latest_history_baseline(history, history_mask, 0, forecast_time)
        self.assertIsNotNone(baseline)
        value, baseline_time = baseline  # type: ignore[misc]
        self.assertEqual(value, 2.0)
        self.assertEqual(baseline_time, datetime(2024, 6, 1, 9))
        self.assertLessEqual(baseline_time, forecast_time)

    def test_masks_exclude_missing_targets_and_missing_baseline_skips_delta(self) -> None:
        batch = SimpleNamespace(
            q_target=torch.tensor([[[1.0], [999.0]]]),
            z_target=torch.tensor([[[999.0], [51.0]]]),
            q_target_mask=torch.tensor([[[True], [False]]]),
            z_target_mask=torch.tensor([[[False], [True]]]),
            z_history=torch.tensor([[[50.0], [50.0]]]),
            z_mask=torch.tensor([[[False], [False]]]),
            station_ids=("S1",),
            sample_id=("SAMPLE1",),
            event_id=("EVENT1",),
            graph_id=("GRAPH1",),
            target_station_id=("S1",),
            forecast_time=("2024-06-01 00:00:00",),
            event_rain_start=("2024-05-31 20:00:00",),
            event_rain_end=("2024-06-01 03:00:00",),
            event_hydro_start=("2024-06-01 00:00:00",),
            event_hydro_end=("",),
            event_peak_time=("2024-06-01 02:00:00",),
            event_sample_start=("2024-05-31 00:00:00",),
            event_sample_end=("2024-06-02 00:00:00",),
        )
        output = {
            "q": torch.tensor([[[2.0], [1.0]]]),
            "z": torch.tensor([[[50.0], [50.5]]]),
        }
        accumulator = ValidationDiagnosticsAccumulator()
        accumulator.add_batch(batch, output)
        diagnostics = accumulator.finalize()

        self.assertEqual(len(accumulator.q_points), 1)
        self.assertEqual(len(accumulator.z_points), 1)
        self.assertEqual(diagnostics.q_by_event[0]["valid_q_count"], 1)
        self.assertEqual(diagnostics.z_by_station[0]["valid_count"], 1)
        self.assertEqual(diagnostics.delta_z_by_station, [])
        self.assertEqual(
            diagnostics.summary["delta_z_overall_metrics"]["skipped_missing_baseline_count"],
            1,
        )

    def test_writer_creates_bounded_validation_artifacts(self) -> None:
        accumulator = ValidationDiagnosticsAccumulator()
        accumulator.q_points = [
            point(graph="G1", event="E1", station="S1", hour=1, observed=2, predicted=3)
        ]
        diagnostics = accumulator.finalize()
        with tempfile.TemporaryDirectory() as directory:
            written = diagnostics.write(directory, split="VALIDATION")
            expected = {
                "validation_q_by_graph.csv",
                "validation_q_by_event.csv",
                "validation_q_top20_error_events.csv",
                "validation_q_top20_sse_events.csv",
                "validation_z_by_station.csv",
                "validation_delta_z_by_station.csv",
                "validation_diagnostics_summary.json",
            }
            self.assertEqual(set(written), expected)
            self.assertTrue(all(Path(path).is_file() for path in written.values()))


if __name__ == "__main__":
    unittest.main()
