from __future__ import annotations

import unittest

import torch

from models.hybrid_model import HybridFloodModel


class _IdentityObservation(torch.nn.Module):
    """Deterministic H(Q,S)=Q+0.1S consistency relation."""

    def forward(self, q, channel_state=None, station_index=None):
        if channel_state is None:
            channel_state = torch.zeros_like(q)
        return q + 0.1 * channel_state


class _FixedIndependentHead(torch.nn.Module):
    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))

    def forward(self, history_context, q_future, *args, **kwargs):
        batch, steps, nodes = q_future.shape
        if steps != self.values.numel():
            raise ValueError("test head horizon mismatch")
        return self.values.view(1, steps, 1).expand(batch, -1, nodes)


class _Batch:
    pass


class TestP3ForecastOriginZ(unittest.TestCase):
    def _model_shell(self, values: list[float]) -> HybridFloodModel:
        model = object.__new__(HybridFloodModel)
        torch.nn.Module.__init__(model)
        model.observation = _IdentityObservation()
        model.hidden_dim = 2
        model.independent_z_head = _FixedIndependentHead(values)
        return model

    def _batch(self) -> _Batch:
        batch = _Batch()
        batch.q_history = torch.tensor([[[5.0], [7.0], [10.0]]])
        batch.q_mask = torch.ones_like(batch.q_history, dtype=torch.bool)
        batch.z_history = torch.tensor([[[99.7], [99.9], [100.0]]])
        batch.z_mask = torch.ones_like(batch.z_history, dtype=torch.bool)
        batch.z_reference = torch.tensor([[100.0]])
        batch.z_reference_mask = torch.tensor([[True]])
        batch.edge_index = torch.empty((2, 0), dtype=torch.long)
        batch.station_index = torch.tensor([0], dtype=torch.long)
        return batch

    @staticmethod
    def _diagnostics() -> dict[str, torch.Tensor]:
        return {
            "initial_edge_storage_m3": torch.empty((1, 0)),
            "history_context": torch.zeros((1, 1, 2)),
        }

    def test_independent_head_is_primary_z_prediction(self) -> None:
        model = self._model_shell([0.2, 0.4, -0.1])
        batch = self._batch()
        future_q = torch.tensor([[[12.0], [15.0], [8.0]]])
        delta, audit = model._independent_forecast_origin_z(
            batch, future_q, torch.zeros_like(future_q), self._diagnostics()
        )
        torch.testing.assert_close(
            delta[:, :, 0], torch.tensor([[0.2, 0.4, -0.1]])
        )
        torch.testing.assert_close(
            audit["absolute_z_forecast_m"][:, :, 0],
            torch.tensor([[100.2, 100.4, 99.9]]),
        )

    def test_monotone_qz_is_only_consistency_target(self) -> None:
        model = self._model_shell([0.2, 0.4, -0.1])
        batch = self._batch()
        future_q = torch.tensor([[[12.0], [15.0], [8.0]]])
        delta, audit = model._independent_forecast_origin_z(
            batch, future_q, torch.zeros_like(future_q), self._diagnostics()
        )
        torch.testing.assert_close(
            audit["qz_consistency_delta_z_m"][:, :, 0],
            torch.tensor([[2.0, 5.0, -2.0]]),
        )
        self.assertFalse(
            torch.allclose(delta, audit["qz_consistency_delta_z_m"])
        )

    def test_observed_z0_is_absolute_anchor(self) -> None:
        model = self._model_shell([0.0])
        batch = self._batch()
        future_q = torch.tensor([[[10.0]]])
        delta, audit = model._independent_forecast_origin_z(
            batch, future_q, torch.zeros_like(future_q), self._diagnostics()
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(
            audit["absolute_z_forecast_m"], torch.tensor([[[100.0]]])
        )

    def test_recent_z_trend_uses_last_two_valid_observations(self) -> None:
        values = torch.tensor([[[1.0], [5.0], [3.0], [8.0]]])
        mask = torch.tensor([[[True], [False], [True], [True]]])
        trend, valid = HybridFloodModel._recent_observed_trend(values, mask)
        torch.testing.assert_close(trend, torch.tensor([[5.0]]))
        self.assertTrue(bool(valid.item()))

    def test_initial_edge_storage_uses_routing_destination_convention(self) -> None:
        model = self._model_shell([0.0])
        batch = self._batch()
        batch.edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        reference = torch.zeros((1, 2, 2))
        storage = model._initial_node_channel_storage(
            batch,
            {"initial_edge_storage_m3": torch.tensor([[30.0]])},
            reference,
        )
        torch.testing.assert_close(storage, torch.tensor([[0.0, 30.0]]))


if __name__ == "__main__":
    unittest.main()
