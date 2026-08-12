from __future__ import annotations

import unittest

import torch

from models.hybrid_model import HybridFloodModel


class _IdentityObservation(torch.nn.Module):
    """Deterministic H(Q,S)=Q+0.1S for forecast-origin delta tests."""

    def forward(self, q, channel_state=None, station_index=None):
        if channel_state is None:
            channel_state = torch.zeros_like(q)
        return q + 0.1 * channel_state


class _ZeroResidual(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x[..., :1])


class _ConstantResidual(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full_like(x[..., :1], self.value)


class _Batch:
    pass


class TestP3ForecastOriginZ(unittest.TestCase):
    def _model_shell(self, residual: torch.nn.Module | None = None) -> HybridFloodModel:
        model = object.__new__(HybridFloodModel)
        torch.nn.Module.__init__(model)
        model.observation = _IdentityObservation()
        model.hidden_dim = 2
        model.stage_residual_head = residual if residual is not None else _ZeroResidual()
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

    def test_delta_is_future_response_minus_forecast_origin_response_when_residual_zero(self) -> None:
        model = self._model_shell()
        batch = self._batch()
        future_q = torch.tensor([[[12.0], [15.0], [8.0]]])
        delta, audit = model._forecast_origin_delta_z(
            batch, future_q, torch.zeros_like(future_q), self._diagnostics()
        )
        torch.testing.assert_close(
            delta[:, :, 0], torch.tensor([[2.0, 5.0, -2.0]])
        )
        torch.testing.assert_close(
            audit["absolute_z_forecast_m"][:, :, 0],
            torch.tensor([[102.0, 105.0, 98.0]]),
        )

    def test_feed_forward_stage_memory_residual_enters_delta_z(self) -> None:
        model = self._model_shell(_ConstantResidual(0.25))
        batch = self._batch()
        future_q = torch.tensor([[[12.0], [15.0], [8.0]]])
        delta, audit = model._forecast_origin_delta_z(
            batch, future_q, torch.zeros_like(future_q), self._diagnostics()
        )
        torch.testing.assert_close(
            delta[:, :, 0], torch.tensor([[2.25, 5.25, -1.75]])
        )
        torch.testing.assert_close(
            audit["stage_memory_residual_m"][:, :, 0],
            torch.tensor([[0.25, 0.25, 0.25]]),
        )

    def test_observed_z0_is_absolute_anchor(self) -> None:
        model = self._model_shell()
        batch = self._batch()
        future_q = torch.tensor([[[10.0]]])
        delta, audit = model._forecast_origin_delta_z(
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
        model = self._model_shell()
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
