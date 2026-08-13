"""P3 model whose Z path is exclusively a frozen TRAIN-calibrated rating function."""
from __future__ import annotations

from typing import Any

import torch

from models.hybrid_model import HybridFloodModel
from models.observation import TrainFittedMonotoneRating


class P3RatingCalibratedModel(HybridFloodModel):
    """Remove the neural Z shortcut while preserving the agreed P3 hydrology.

    The base rating-aligned P3 still provides:
    * TRAIN-aligned runtime inputs (through the dedicated runtime adapter),
    * exact-Q0 / inverse-rating-Z0 informed state initialization,
    * WaterBalanceLSTM runoff and kinematic-wave routing,
    * Z-loss backpropagation through a differentiable rating function into Q.

    This subclass changes only the observation mapping. Paired TRAIN Q/Z points
    calibrate one frozen monotone piecewise-linear station rating function before
    training. No IndependentDeltaZHead exists in this model, so there is no
    trainable Z residual path capable of absorbing hydrologic Q error.
    """

    def __init__(self, cfg: dict[str, Any], stations: int) -> None:
        super().__init__(cfg, stations)
        if not self.rating_aligned_p3:
            raise ValueError("P3RatingCalibratedModel要求p3_rating_aligned runtime")
        if not bool(cfg.get("_runtime", {}).get("p3_rating_calibrated", False)):
            raise ValueError("P3RatingCalibratedModel要求p3_rating_calibrated runtime")
        # Explicitly remove the neural residual from the parameter graph.
        self.independent_z_head = None
        self.rating_curve = TrainFittedMonotoneRating(stations)

    def _rating_backed_forecast_origin_z(
        self,
        batch: Any,
        q_future: torch.Tensor,
        channel_future: torch.Tensor | None,
        routing_diagnostics: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict delta-Z solely from the frozen calibrated Q->Z function."""
        del channel_future  # observation mapping intentionally depends on Q only
        if self.rating_curve is None:
            raise RuntimeError("P3 calibrated rating module未构建")
        z_reference = getattr(batch, "z_reference", None)
        z_reference_mask = getattr(batch, "z_reference_mask", None)
        if z_reference is None or z_reference_mask is None:
            raise RuntimeError("P3 calibrated delta-Z要求forecast-origin z_reference")
        reference_valid = z_reference_mask.bool() & torch.isfinite(z_reference)
        reference = torch.where(reference_valid, z_reference, torch.zeros_like(z_reference))

        # No detach: every valid Z loss is a direct supervisory path into Q.
        calibrated_level, station_rating_available = self.rating_curve(
            q_future, getattr(batch, "station_index", None)
        )
        rating_valid = reference_valid & station_rating_available.view(1, -1)
        z_delta = calibrated_level - reference.unsqueeze(1)
        z_delta = torch.where(
            rating_valid.unsqueeze(1), z_delta, torch.zeros_like(z_delta)
        )

        q_anchor = routing_diagnostics.get("q_origin_anchor_m3s")
        q_anchor_available = routing_diagnostics.get("q_origin_anchor_available")
        if q_anchor is None or q_anchor_available is None:
            raise RuntimeError("P3 calibrated Z路径缺少forecast-origin Q anchor diagnostics")
        diagnostics = {
            "forecast_origin_q_m3s": q_anchor,
            "forecast_origin_q_available": q_anchor_available,
            "forecast_origin_z_m": reference,
            "forecast_origin_z_available": reference_valid,
            "rating_available": rating_valid,
            "rating_delta_z_m": z_delta,
            "calibrated_rating_delta_z_m": z_delta,
            "absolute_z_forecast_m": reference.unsqueeze(1) + z_delta,
        }
        return z_delta, diagnostics
