"""Mask-aware losses and flood-forecast metrics.

The public mean-returning helpers are kept for backwards compatibility.  The
``*_stats`` helpers additionally return an element sum and a valid-element
count so callers can aggregate across batches without giving a small final
batch the same weight as a full batch.

Mask semantics are deliberately strict:

* non-finite values outside the mask are ignored;
* a non-finite target inside the mask is a data-contract error;
* a non-finite prediction inside the mask is a numerical error;
* an empty mask has sum/count ``0/0``.  Losses return a differentiable zero,
  while reported means (for example horizon MAE) are ``NaN``.
"""

from __future__ import annotations

import math

import torch


def _checked_mask(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    if pred.shape != target.shape or mask.shape != target.shape:
        raise ValueError(
            "pred、target、mask形状必须完全一致，"
            f"实际为{tuple(pred.shape)}、{tuple(target.shape)}、{tuple(mask.shape)}"
        )
    if mask.dtype != torch.bool:
        if torch.is_floating_point(mask) and not torch.isfinite(mask).all():
            raise ValueError("mask包含NaN/Inf")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("mask只能包含布尔值或0/1")
    valid = mask.bool()
    if valid.any() and not torch.isfinite(target[valid]).all():
        raise ValueError("有效mask内的target包含NaN/Inf；请修正数据或关闭对应mask")
    if valid.any() and not torch.isfinite(pred[valid]).all():
        raise FloatingPointError("有效mask内的模型预测包含NaN/Inf")
    return valid


def valid_target_count(target: torch.Tensor, mask: torch.Tensor) -> int:
    """Return the number of valid targets after enforcing the mask contract."""

    # Reuse the complete validation path without allocating a data-sized tensor.
    # ``target`` is also a finite placeholder prediction wherever the mask is on.
    valid = _checked_mask(target, target, mask)
    return int(valid.sum().item())


def masked_huber_stats(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> tuple[torch.Tensor, int]:
    """Return ``(Huber element sum, valid element count)``.

    The zero returned for an empty mask remains attached to ``pred`` so a
    missing observation type can coexist with another trainable loss term.
    """

    if not math.isfinite(delta) or delta <= 0:
        raise ValueError(f"delta必须是有限正数，实际为{delta}")
    valid = _checked_mask(pred, target, mask)
    count = int(valid.sum().item())
    if count == 0:
        # Summing the empty selected tensor keeps the graph connection while
        # avoiding ``Inf * 0 -> NaN`` from masked-off predictions.
        return pred[valid].sum(), 0
    element_loss = torch.nn.functional.huber_loss(
        pred[valid], target[valid], delta=delta, reduction="sum"
    )
    return element_loss, count


def masked_huber(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Return the valid-element mean Huber loss (or differentiable zero)."""

    loss_sum, count = masked_huber_stats(pred, target, mask, delta)
    return loss_sum / count if count else loss_sum


def masked_mae_stats(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, int]:
    """Return ``(absolute-error sum, valid element count)``."""

    valid = _checked_mask(pred, target, mask)
    count = int(valid.sum().item())
    if count == 0:
        return pred.detach()[valid].sum(), 0
    return (pred[valid] - target[valid]).abs().sum(), count


def horizon_metric_stats(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, tuple[float, int]]:
    """Return per-horizon ``(absolute-error sum, count)`` pairs."""

    if pred.ndim < 2:
        raise ValueError(f"逐时指标要求至少二维张量[B,H,...]，实际维度为{pred.ndim}")
    # Validate once here so invalid values are rejected even on an empty horizon.
    _checked_mask(pred, target, mask)
    out: dict[str, tuple[float, int]] = {}
    for horizon in range(pred.shape[1]):
        error_sum, count = masked_mae_stats(
            pred[:, horizon], target[:, horizon], mask[:, horizon]
        )
        out[f"h{horizon + 1}_mae"] = (float(error_sum.item()), count)
    return out


def horizon_metrics(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float]:
    """Return per-horizon MAE; a horizon with no observations is ``NaN``."""

    return {
        name: error_sum / count if count else float("nan")
        for name, (error_sum, count) in horizon_metric_stats(
            pred, target, mask
        ).items()
    }


def masked_regression_sums(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float | int]:
    """Return mergeable double-precision sums for physical-unit metrics."""

    valid = _checked_mask(pred, target, mask)
    count = int(valid.sum().item())
    if not count:
        return {
            "count": 0,
            "absolute_error": 0.0,
            "squared_error": 0.0,
            "error": 0.0,
            "prediction": 0.0,
            "target": 0.0,
            "prediction_squared": 0.0,
            "target_squared": 0.0,
            "cross": 0.0,
        }
    prediction = pred[valid].double()
    observation = target[valid].double()
    error = prediction - observation
    return {
        "count": count,
        "absolute_error": float(error.abs().sum().item()),
        "squared_error": float(error.square().sum().item()),
        "error": float(error.sum().item()),
        "prediction": float(prediction.sum().item()),
        "target": float(observation.sum().item()),
        "prediction_squared": float(prediction.square().sum().item()),
        "target_squared": float(observation.square().sum().item()),
        "cross": float((prediction * observation).sum().item()),
    }


def _regression_report(
    sums: dict[str, float | int],
) -> tuple[dict[str, float | int], dict[str, str]]:
    """Return metrics and explicit definition statuses from the same rules."""

    count = int(sums["count"])
    if count == 0:
        metrics: dict[str, float | int] = {
            name: float("nan") for name in ("mae", "rmse", "bias", "nse", "kge")
        }
        metrics["valid_count"] = 0
        return metrics, {
            "nse": "NO_VALID_OBSERVATIONS",
            "kge": "NO_VALID_OBSERVATIONS",
        }
    n = float(count)
    prediction_mean = float(sums["prediction"]) / n
    target_mean = float(sums["target"]) / n
    prediction_variance = max(
        0.0, float(sums["prediction_squared"]) - n * prediction_mean**2
    )
    target_variance = max(
        0.0, float(sums["target_squared"]) - n * target_mean**2
    )
    covariance = float(sums["cross"]) - n * prediction_mean * target_mean
    nse = (
        1.0 - float(sums["squared_error"]) / target_variance
        if target_variance > 0
        else float("nan")
    )
    if prediction_variance > 0 and target_variance > 0 and target_mean != 0:
        correlation = covariance / math.sqrt(prediction_variance * target_variance)
        variability_ratio = math.sqrt(prediction_variance / target_variance)
        bias_ratio = prediction_mean / target_mean
        kge = 1.0 - math.sqrt(
            (correlation - 1.0) ** 2
            + (variability_ratio - 1.0) ** 2
            + (bias_ratio - 1.0) ** 2
        )
    else:
        kge = float("nan")
    metrics = {
        "valid_count": count,
        "mae": float(sums["absolute_error"]) / n,
        "rmse": math.sqrt(float(sums["squared_error"]) / n),
        "bias": float(sums["error"]) / n,
        "nse": nse,
        "kge": kge,
    }
    if target_variance <= 0:
        nse_status = (
            "INSUFFICIENT_VALID_COUNT" if count < 2 else "ZERO_OBS_VARIANCE"
        )
    else:
        nse_status = "DEFINED"
    if count < 2:
        kge_status = "INSUFFICIENT_VALID_COUNT"
    elif target_variance <= 0:
        kge_status = "ZERO_OBS_VARIANCE"
    elif prediction_variance <= 0:
        kge_status = "ZERO_PRED_VARIANCE"
    elif target_mean == 0:
        kge_status = "ZERO_OBS_MEAN"
    else:
        kge_status = "DEFINED"
    return metrics, {"nse": nse_status, "kge": kge_status}


def regression_metrics(sums: dict[str, float | int]) -> dict[str, float | int]:
    """Convert mergeable sums into MAE/RMSE/bias/NSE/KGE metrics."""

    return _regression_report(sums)[0]


def regression_metric_status(sums: dict[str, float | int]) -> dict[str, str]:
    """Explain why NSE/KGE are defined or safely reported as ``NaN``."""

    return _regression_report(sums)[1]


def hydrograph_sample_sums(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> dict[str, float | int]:
    """Aggregate outlet-series peak, timing, and volume errors per sample/node.

    Peak magnitude and timing are evaluated over the jointly valid time steps
    of each ``[sample, node]`` series.  Signed errors use the following stable
    conventions:

    * ``peak_signed_error = predicted_peak - observed_peak`` (positive means
      that the peak magnitude is overestimated);
    * ``peak_relative_error = peak_signed_error / observed_peak`` (included
      only when the observed peak is non-zero, and stored as a ratio rather
      than a percentage);
    * ``peak_timing_signed_error = predicted_peak_index - observed_peak_index``
      (positive means that the prediction is late, negative means early).

    The existing absolute-error fields are retained for backwards
    compatibility.  Every returned error field is a sum, with ``peak_count``
    or ``peak_relative_count`` providing the corresponding denominator for
    aggregation across batches.
    """

    _checked_mask(pred, target, mask)
    if pred.ndim != 3:
        raise ValueError("洪水过程指标要求[B,H,N]")
    peak_error = 0.0
    peak_signed_error = 0.0
    peak_relative_error = 0.0
    timing_error = 0.0
    timing_signed_error = 0.0
    relative_volume_error = 0.0
    peak_count = 0
    peak_relative_count = 0
    volume_count = 0
    for batch_index in range(pred.shape[0]):
        for node_index in range(pred.shape[2]):
            valid_times = mask[batch_index, :, node_index].bool().nonzero(
                as_tuple=False
            ).flatten()
            if not valid_times.numel():
                continue
            prediction = pred[batch_index, valid_times, node_index].double()
            observation = target[batch_index, valid_times, node_index].double()
            predicted_peak = prediction.max()
            observed_peak = observation.max()
            signed_peak_difference = float((predicted_peak - observed_peak).item())
            peak_error += abs(signed_peak_difference)
            peak_signed_error += signed_peak_difference
            observed_peak_value = float(observed_peak.item())
            if observed_peak_value != 0:
                peak_relative_error += signed_peak_difference / observed_peak_value
                peak_relative_count += 1
            predicted_peak_time = int(valid_times[int(prediction.argmax())].item())
            observed_peak_time = int(valid_times[int(observation.argmax())].item())
            signed_timing_difference = predicted_peak_time - observed_peak_time
            timing_error += abs(signed_timing_difference)
            timing_signed_error += signed_timing_difference
            peak_count += 1
            observed_volume = float(observation.sum().item())
            if observed_volume != 0:
                relative_volume_error += float(
                    (prediction.sum() - observation.sum()).item()
                ) / observed_volume
                volume_count += 1
    return {
        "peak_absolute_error": peak_error,
        "peak_signed_error": peak_signed_error,
        "peak_relative_error": peak_relative_error,
        "peak_timing_absolute_error": timing_error,
        "peak_timing_signed_error": timing_signed_error,
        "peak_count": peak_count,
        "peak_relative_count": peak_relative_count,
        "relative_volume_error": relative_volume_error,
        "volume_count": volume_count,
    }
