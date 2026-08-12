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


class _Batch:
    pass


class TestP3ForecastOriginZ(unittest.TestCase):
    def _model_shell(self) -> HybridFloodModel:
        model = object.__new__(HybridFloodModel)
        torch.nn.Module.__init__(model)
        model.observation = _IdentityObservation()
        return model

    def _batch(self) -> _Batch:
        batch = _Batch()
        batch.q_history = torch.tensor([[[5.0], [7.0], [10.0]]])
        batch.q_mask = torch.ones_like(batch.q_history, dtype=torch.bool)
        batch.z_reference = torch.tensor([[100.0]])
        batch.z_reference_mask = torch.tensor([[True]])
        batch.edge_index = torch.empty((2, 0), dtype=torch.long)
        batch.station_index = torch.tensor([0], dtype=torch.long)
        return batch

    def test_delta_is_future_response_minus_forecast_origin_response(self) -> None:
        model = self._model_shell()
        batch = self._batch()
        future_q = torch.tensor([[[12.0], [15.0], [8.0]]])
        diagnostics = {"initial_edge_storage_m3": torch.empty((1, 0))}
        delta, audit = model._forecast_origin_delta_z(
            batch, future_q, torch.zeros_like(future_q), diagnostics
        )
        torch.testing.assert_close(
            delta[:, :, 0], torch.tensor([[2.0, 5.0, -2.0]])
        )
        torch.testing.assert_close(
            audit["absolute_z_forecast_m"][:, :, 0],
            torch.tensor([[102.0, 105.0, 98.0]]),
        )

    def test_observed_z0_is_absolute_anchor_not_free_station_offset(self) -> None:
        model = self._model_shell()
        batch = self._batch()
        future_q = torch.tensor([[[10.0]]])
        diagnostics = {"initial_edge_storage_m3": torch.empty((1, 0))}
        delta, audit = model._forecast_origin_delta_z(
            batch, future_q, torch.zeros_like(future_q), diagnostics
        )
        torch.testing.assert_close(delta, torch.zeros_like(delta))
        torch.testing.assert_close(
            audit["absolute_z_forecast_m"], torch.tensor([[[100.0]]])
        )

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
