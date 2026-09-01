"""Q-only trainer with persistence-aware, station-macro checkpoint selection."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping

import torch

from losses.hydrologic_loss import HydrologicLoss
from metrics.flood_metrics import (
    masked_regression_sums,
    regression_metrics,
    valid_target_count,
)
from trainers.trainer import Trainer


def _empty_regression() -> dict[str, float | int]:
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


def _merge(
    target: dict[str, float | int], source: Mapping[str, float | int]
) -> None:
    for key in target:
        target[key] = target[key] + source[key]


def _skill(model: Mapping[str, float | int], baseline: Mapping[str, float | int]) -> float:
    baseline_sse = float(baseline["squared_error"])
    if int(baseline["count"]) == 0 or baseline_sse <= 0.0:
        return float("nan")
    return 1.0 - float(model["squared_error"]) / baseline_sse


def _finite_median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("nan")
    midpoint = len(finite) // 2
    return finite[midpoint] if len(finite) % 2 else 0.5 * (finite[midpoint - 1] + finite[midpoint])


class HydrologicTrainer(Trainer):
    """Fit the model while refusing to select a checkpoint by loss alone."""

    def __init__(
        self, model: torch.nn.Module, cfg: dict, device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        opt = cfg["optimizer"]
        trainable = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("model没有可训练参数")
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=float(opt["lr"]),
            weight_decay=float(opt["weight_decay"]),
        )
        self.loss_engine = HydrologicLoss(cfg)
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.selection_mode = str(
            cfg.get("validation_selection", {}).get("mode", "val_loss")
        )
        if self.selection_mode not in {"val_loss", "station_macro_persistence_skill"}:
            raise ValueError(
                "Q-only checkpoint selection必须是val_loss或"
                "station_macro_persistence_skill"
            )
        self.best = float("-inf") if self.selection_mode != "val_loss" else float("inf")
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None

    @property
    def selection_metric_name(self) -> str:
        return "validation_selection_score" if self.selection_mode != "val_loss" else "val_loss"

    @property
    def selection_direction(self) -> str:
        return "maximize" if self.selection_mode != "val_loss" else "minimize"

    def _selection_report(
        self,
        row: Mapping[str, float | int],
        validation_summary: Mapping[str, float | int],
    ) -> dict[str, float | int]:
        del validation_summary
        if self.selection_mode == "val_loss":
            return {}
        score = float(row.get("val_q_station_macro_persistence_skill_median", float("nan")))
        if not math.isfinite(score):
            raise FloatingPointError("station-macro persistence skill未定义，不能选择checkpoint")
        return {
            "validation_selection_score": score,
            "validation_selection_station_improved_fraction": float(
                row["val_q_station_improved_fraction"]
            ),
            "validation_selection_delta_q_nse_median": float(
                row["val_delta_q_station_macro_nse_median"]
            ),
        }

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Any],
        *,
        include_group_metrics: bool = False,
        include_group_details: bool = False,
        include_validation_diagnostics: bool = False,
        include_diagnostic_details: bool = False,
    ) -> dict[str, Any]:
        """Evaluate Q, Delta-Q, every lead, and every station in one pass."""

        del (
            include_group_metrics,
            include_group_details,
            include_validation_diagnostics,
            include_diagnostic_details,
        )
        self.model.eval()
        loss_totals = {name: [0.0, 0] for name in self.loss_engine.coefficients()}
        q_valid_total = 0
        all_q = _empty_regression()
        model_q0_subset = _empty_regression()
        persistence = _empty_regression()
        delta_q = _empty_regression()
        station_model: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
        station_persistence: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
        station_delta: dict[str, dict[str, float | int]] = defaultdict(_empty_regression)
        lead_model: list[dict[str, float | int]] | None = None
        lead_persistence: list[dict[str, float | int]] | None = None
        batch_count = 0

        for batch in loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            statistics = self.loss_engine.batch_statistics(output, batch)
            for name, term in statistics.items():
                loss_totals[name][0] += float(term.numerator.detach().item())
                loss_totals[name][1] += int(term.denominator)
            q_valid_total += valid_target_count(batch.q_target, batch.q_target_mask)
            _merge(all_q, masked_regression_sums(output["q"], batch.q_target, batch.q_target_mask))

            q0_available = output["diagnostics"]["q_origin_observed_available"].bool()
            q0 = output["q0_analysis"]
            q0_mask = batch.q_target_mask.bool() & q0_available.unsqueeze(1)
            q0_expanded = q0.unsqueeze(1).expand_as(batch.q_target)
            _merge(model_q0_subset, masked_regression_sums(output["q"], batch.q_target, q0_mask))
            _merge(persistence, masked_regression_sums(q0_expanded, batch.q_target, q0_mask))
            _merge(delta_q, masked_regression_sums(output["q"] - q0_expanded, batch.q_target - q0_expanded, q0_mask))

            horizon = int(batch.q_target.shape[1])
            if lead_model is None:
                lead_model = [_empty_regression() for _ in range(horizon)]
                lead_persistence = [_empty_regression() for _ in range(horizon)]
            elif len(lead_model) != horizon:
                raise ValueError("validation horizon在batch之间不一致")
            assert lead_model is not None and lead_persistence is not None
            for lead in range(horizon):
                lead_mask = q0_mask[:, lead : lead + 1]
                _merge(
                    lead_model[lead],
                    masked_regression_sums(
                        output["q"][:, lead : lead + 1],
                        batch.q_target[:, lead : lead + 1],
                        lead_mask,
                    ),
                )
                _merge(
                    lead_persistence[lead],
                    masked_regression_sums(
                        q0_expanded[:, lead : lead + 1],
                        batch.q_target[:, lead : lead + 1],
                        lead_mask,
                    ),
                )

            station_ids = tuple(str(identifier) for identifier in batch.obs_station_ids)
            if len(station_ids) != batch.q_target.shape[2]:
                raise ValueError("obs_station_ids与Q目标站点维度不一致")
            for station_index, station_id in enumerate(station_ids):
                station_mask = q0_mask[:, :, station_index : station_index + 1]
                station_q0 = q0_expanded[:, :, station_index : station_index + 1]
                station_target = batch.q_target[:, :, station_index : station_index + 1]
                station_prediction = output["q"][:, :, station_index : station_index + 1]
                _merge(
                    station_model[station_id],
                    masked_regression_sums(station_prediction, station_target, station_mask),
                )
                _merge(
                    station_persistence[station_id],
                    masked_regression_sums(station_q0, station_target, station_mask),
                )
                _merge(
                    station_delta[station_id],
                    masked_regression_sums(
                        station_prediction - station_q0,
                        station_target - station_q0,
                        station_mask,
                    ),
                )
            batch_count += 1

        if batch_count == 0:
            raise ValueError("evaluation DataLoader为空")
        result = self.loss_engine.report(
            {name: (float(value), int(count)) for name, (value, count) in loss_totals.items()},
            q_valid_count=q_valid_total,
            z_valid_count=0,
        )
        q_metrics = regression_metrics(all_q)
        model_q0_metrics = regression_metrics(model_q0_subset)
        persistence_metrics = regression_metrics(persistence)
        delta_metrics = regression_metrics(delta_q)
        result.update(
            {
                "q_mae": float(q_metrics["mae"]),
                "q_rmse": float(q_metrics["rmse"]),
                "q_nse": float(q_metrics["nse"]),
                "q_kge": float(q_metrics["kge"]),
                "q_valid_count": int(q_metrics["valid_count"]),
                "q0_observed_valid_count": int(model_q0_metrics["valid_count"]),
                "q0_subset_model_nse": float(model_q0_metrics["nse"]),
                "q0_persistence_nse": float(persistence_metrics["nse"]),
                "q_skill_over_persistence": _skill(model_q0_subset, persistence),
                "delta_q_rmse": float(delta_metrics["rmse"]),
                "delta_q_nse": float(delta_metrics["nse"]),
            }
        )

        station_skills = [_skill(station_model[name], station_persistence[name]) for name in sorted(station_model)]
        station_delta_nse = [
            float(regression_metrics(station_delta[name])["nse"])
            for name in sorted(station_delta)
        ]
        valid_station_skills = [value for value in station_skills if math.isfinite(value)]
        valid_delta_nse = [value for value in station_delta_nse if math.isfinite(value)]
        result.update(
            {
                "q_station_macro_persistence_skill_mean": (
                    sum(valid_station_skills) / len(valid_station_skills)
                    if valid_station_skills else float("nan")
                ),
                "q_station_macro_persistence_skill_median": _finite_median(valid_station_skills),
                "q_station_macro_persistence_skill_defined_count": len(valid_station_skills),
                "q_station_improved_fraction": (
                    sum(value > 0.0 for value in valid_station_skills) / len(valid_station_skills)
                    if valid_station_skills else float("nan")
                ),
                "delta_q_station_macro_nse_mean": (
                    sum(valid_delta_nse) / len(valid_delta_nse)
                    if valid_delta_nse else float("nan")
                ),
                "delta_q_station_macro_nse_median": _finite_median(valid_delta_nse),
                "delta_q_station_macro_nse_defined_count": len(valid_delta_nse),
            }
        )
        assert lead_model is not None and lead_persistence is not None
        for lead, (model_sums, persistence_sums) in enumerate(zip(lead_model, lead_persistence), start=1):
            result[f"q_lead_{lead}_skill_over_persistence"] = _skill(model_sums, persistence_sums)
        return result
