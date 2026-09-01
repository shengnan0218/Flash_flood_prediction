"""Training, evaluation and restart support for flood-forecast models."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
import time
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Callable

import torch

from data.device import device_report
from losses import FloodMultitaskLoss
from metrics.flood_metrics import (
    hydrograph_sample_sums,
    horizon_metric_stats,
    masked_regression_sums,
    regression_metrics,
    valid_target_count,
)
from metrics.validation_diagnostics import (
    ValidationDiagnostics,
    ValidationDiagnosticsAccumulator,
)
from metrics.validation_selection import validation_selection_score


_REGRESSION_SUM_FIELDS = (
    "count",
    "absolute_error",
    "squared_error",
    "error",
    "prediction",
    "target",
    "prediction_squared",
    "target_squared",
    "cross",
)

_HYDROGRAPH_SUM_FIELDS = (
    "peak_absolute_error",
    "peak_signed_error",
    "peak_relative_error",
    "peak_timing_absolute_error",
    "peak_timing_signed_error",
    "peak_count",
    "peak_relative_count",
    "relative_volume_error",
    "volume_count",
)


def _empty_regression_sums() -> dict[str, float | int]:
    return {name: 0 if name == "count" else 0.0 for name in _REGRESSION_SUM_FIELDS}


def _empty_hydrograph_sums() -> dict[str, float | int]:
    integer_fields = {"peak_count", "peak_relative_count", "volume_count"}
    return {
        name: 0 if name in integer_fields else 0.0
        for name in _HYDROGRAPH_SUM_FIELDS
    }


def _add_sums(
    destination: dict[str, float | int], source: dict[str, float | int]
) -> None:
    if destination.keys() != source.keys():
        raise ValueError(
            "指标聚合字段不一致: "
            f"destination={sorted(destination)}, source={sorted(source)}"
        )
    for name, value in source.items():
        destination[name] += value


def _batch_group_ids(
    batch: Any, name: str, batch_size: int
) -> tuple[str, ...] | None:
    """Return strict per-sample metadata IDs, or None for synthetic loaders."""

    value = getattr(batch, name, None)
    if value is None:
        return None
    if isinstance(value, str):
        identifiers = (value,)
    elif isinstance(value, (tuple, list)):
        identifiers = tuple(value)
    else:
        raise ValueError(f"{name}必须是字符串或逐样本字符串序列")
    if len(identifiers) != batch_size:
        raise ValueError(
            f"{name}数量必须等于batch size={batch_size}，实际={len(identifiers)}"
        )
    if any(not isinstance(identifier, str) or not identifier.strip() for identifier in identifiers):
        raise ValueError(f"{name}必须全部为非空字符串")
    return identifiers


def _hydrograph_report(sums: dict[str, float | int]) -> dict[str, float | int]:
    peak_count = int(sums["peak_count"])
    relative_peak_count = int(sums["peak_relative_count"])
    volume_count = int(sums["volume_count"])

    def mean(name: str, count: int) -> float:
        return float(sums[name]) / count if count else float("nan")

    return {
        "sample_peak_mae": mean("peak_absolute_error", peak_count),
        "sample_peak_bias": mean("peak_signed_error", peak_count),
        "sample_relative_peak_bias": mean(
            "peak_relative_error", relative_peak_count
        ),
        "sample_peak_timing_mae_hours": mean(
            "peak_timing_absolute_error", peak_count
        ),
        "sample_peak_timing_bias_hours": mean(
            "peak_timing_signed_error", peak_count
        ),
        "sample_relative_volume_bias": mean("relative_volume_error", volume_count),
        "hydrograph_count": peak_count,
        "relative_peak_count": relative_peak_count,
        "volume_count": volume_count,
    }


def _finite_macro(
    reports: list[dict[str, float | int]], name: str
) -> tuple[float, int]:
    values = [
        float(report[name])
        for report in reports
        if math.isfinite(float(report[name]))
    ]
    return (sum(values) / len(values) if values else float("nan"), len(values))


def _append_csv_row(path: Path, row: dict[str, Any]) -> None:
    """Append while extending an existing log header with new trailing fields.

    This keeps resumed pre-diagnostics runs readable: old rows receive empty
    cells for newly introduced validation summaries instead of being followed
    by wider rows under a stale header.
    """

    fieldnames = list(row)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"训练日志缺少CSV表头: {path}")
        existing_fields = list(reader.fieldnames)
        old_rows = list(reader)
    new_fields = [name for name in fieldnames if name not in existing_fields]
    if not new_fields:
        with path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=existing_fields).writerow(row)
        return
    extended_fields = [*existing_fields, *new_fields]
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            descriptor = -1
            writer = csv.DictWriter(handle, fieldnames=extended_fields)
            writer.writeheader()
            writer.writerows(old_rows)
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary_path.unlink(missing_ok=True)


class Trainer:
    def __init__(
        self, model: torch.nn.Module, cfg: dict[str, Any], device: torch.device
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        opt = cfg["optimizer"]
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("模型没有可训练参数")
        self.optimizer = torch.optim.AdamW(
            trainable, lr=opt["lr"], weight_decay=opt["weight_decay"]
        )
        self.loss_engine = FloodMultitaskLoss(cfg)
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        # torch.amp.GradScaler was not exported until PyTorch 2.3.  Keep the
        # supported >=2.2 environment usable without changing checkpoint format.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover - exercised only by older supported PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.selection_mode = str(
            cfg.get("validation_selection", {}).get("mode", "val_loss")
        )
        if self.selection_mode not in {"val_loss", "composite"}:
            raise ValueError(
                "validation_selection.mode必须是val_loss/composite，"
                f"实际={self.selection_mode!r}"
            )
        self.best = (
            float("-inf") if self.selection_mode == "composite" else float("inf")
        )
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None

    @property
    def selection_metric_name(self) -> str:
        return (
            "validation_selection_score"
            if self.selection_mode == "composite"
            else "val_loss"
        )

    @property
    def selection_direction(self) -> str:
        return "maximize" if self.selection_mode == "composite" else "minimize"

    def _selection_report(
        self,
        row: dict[str, float | int],
        validation_summary: dict[str, float | int],
    ) -> dict[str, float]:
        if self.selection_mode == "val_loss":
            value = float(row.get("val_loss", row["loss"]))
            if not math.isfinite(value):
                raise FloatingPointError("val_loss早停指标出现NaN/Inf")
            return {}
        if not validation_summary:
            raise ValueError(
                "composite selection要求正式VALIDATION diagnostics，不能在无诊断时回退val_loss"
            )
        return validation_selection_score(
            validation_summary,
            self.cfg.get("_runtime", {}).get("loss_scales", {}),
            self.cfg["validation_selection"],
        )

    def _loss(
        self, out: dict[str, Any], batch: Any
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Backwards-compatible per-batch mean loss helper."""

        statistics = self.loss_engine.batch_statistics(out, batch)
        loss = self.loss_engine.combine(statistics)
        parts = self.loss_engine.report(
            {
                name: (float(term.numerator.detach().item()), term.denominator)
                for name, term in statistics.items()
            },
            q_valid_count=valid_target_count(batch.q_target, batch.q_target_mask),
            z_valid_count=valid_target_count(batch.z_target, batch.z_target_mask),
        )
        parts.pop("loss")
        return loss, parts

    def _batch_groups(self, loader: Iterable[Any], accumulation: int) -> Iterator[list[Any]]:
        maximum: int | None = None
        if self.cfg.get("debug_mode"):
            maximum = int(self.cfg.get("debug_max_batches", 10**9))
            if maximum < 0:
                raise ValueError("debug_max_batches不能为负数")
        group: list[Any] = []
        for index, batch in enumerate(loader):
            if maximum is not None and index >= maximum:
                break
            group.append(batch)
            if len(group) == accumulation:
                yield group
                group = []
        if group:
            yield group

    def _clip_gradients(self) -> torch.Tensor:
        """Clip unscaled gradients while preserving GradScaler overflow handling."""

        return torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(self.cfg["training"]["gradient_clip"]),
            # GradScaler.unscale_ has already recorded AMP overflows.  Let
            # scaler.step/update skip that optimizer step and lower the scale;
            # retain fail-fast behaviour for ordinary full-precision training.
            error_if_nonfinite=not self.amp,
        )

    def train_epoch(
        self, loader: Iterable[Any], epoch: int = 0
    ) -> dict[str, float | int]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = int(self.cfg["gradient_accumulation_steps"])
        if accumulation < 1:
            raise ValueError("gradient_accumulation_steps必须>=1")

        loss_totals = {name: [0.0, 0] for name in self.loss_engine.coefficients()}
        q_valid_total = 0
        z_valid_total = 0
        batch_times: list[float] = []
        explicit_equivalent_substeps: list[float] = []
        batch_count = 0

        for group in self._batch_groups(loader, accumulation):
            # Knowing the complete group's denominators before forward passes
            # makes accumulation exactly valid-element weighted and also gives a
            # short final group the correct (not 1/accumulation) scale.
            batch_denominators = [
                self.loss_engine.denominators(batch) for batch in group
            ]
            group_denominators = {
                name: sum(values[name] for values in batch_denominators)
                for name in self.loss_engine.coefficients()
            }
            if not any(group_denominators.values()):
                raise ValueError("一个梯度累积组内没有任何有效Q/Z监督目标")

            try:
                for batch in group:
                    tick = time.perf_counter()
                    batch = batch.to(self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.float16,
                        enabled=self.amp,
                    ):
                        out = self.model(batch)
                        statistics = self.loss_engine.batch_statistics(out, batch)
                        contribution = self.loss_engine.combine(
                            statistics, group_denominators
                        )
                    if not torch.isfinite(contribution.detach()).all():
                        raise FloatingPointError("训练损失出现NaN/Inf")
                    self.scaler.scale(contribution).backward()

                    for name, term in statistics.items():
                        loss_totals[name][0] += float(term.numerator.detach().item())
                        loss_totals[name][1] += term.denominator
                    q_valid_total += valid_target_count(
                        batch.q_target, batch.q_target_mask
                    )
                    z_valid_total += valid_target_count(
                        batch.z_target, batch.z_target_mask
                    )
                    batch_count += 1
                    batch_times.append(time.perf_counter() - tick)
                    diagnostic = out.get("diagnostics", {}).get(
                        "routing_explicit_equivalent_substeps"
                    )
                    if isinstance(diagnostic, torch.Tensor) and diagnostic.numel():
                        explicit_equivalent_substeps.append(
                            float(diagnostic.float().mean().item())
                        )

                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "CUDA显存不足：请降低batch_size、history_length或hidden_dim；"
                    "配置不会被静默修改"
                ) from exc

        if batch_count == 0:
            raise ValueError("训练DataLoader为空（或debug_max_batches=0）")
        result = self.loss_engine.report(
            {name: (float(value), int(count)) for name, (value, count) in loss_totals.items()},
            q_valid_count=q_valid_total,
            z_valid_count=z_valid_total,
        )
        result["batch_seconds"] = sum(batch_times) / len(batch_times)
        # Muskingum and pure-GNN routes have no explicit CFL substeps. Omit the
        # field in those modes instead of emitting a misleading NaN.
        if explicit_equivalent_substeps:
            result["routing_mean_explicit_equivalent_substeps"] = (
                sum(explicit_equivalent_substeps) / len(explicit_equivalent_substeps)
            )
        return result

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
        """Evaluate physical-unit metrics.

        Grouped metrics are opt-in because calculating per-event/per-graph
        window aggregates every validation epoch adds unnecessary overhead and
        produces very wide training logs.  The formal evaluation entry point
        enables them for the final JSON report.  Requesting details implicitly
        enables the corresponding macro metrics.
        """

        self.model.eval()
        collect_group_metrics = include_group_metrics or include_group_details
        collect_validation_diagnostics = (
            include_validation_diagnostics or include_diagnostic_details
        )
        diagnostic_accumulator = (
            ValidationDiagnosticsAccumulator()
            if collect_validation_diagnostics
            else None
        )
        loss_totals = {name: [0.0, 0] for name in self.loss_engine.coefficients()}
        q_valid_total = 0
        z_valid_total = 0
        metric_totals: dict[str, list[float | int]] = {}
        regression_totals = {
            prefix: _empty_regression_sums() for prefix in ("q", "z")
        }
        hydrograph_totals = _empty_hydrograph_sums()
        grouped_regression_totals: dict[
            str, dict[str, dict[str, dict[str, float | int]]]
        ] = {
            scope: {prefix: {} for prefix in ("q", "z")}
            for scope in ("event", "graph")
        }
        grouped_hydrograph_totals: dict[
            str, dict[str, dict[str, float | int]]
        ] = {scope: {} for scope in ("event", "graph")}
        metadata_presence: dict[str, bool | None] = {
            scope: None for scope in ("event", "graph")
        }
        batch_count = 0

        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch)
            statistics = self.loss_engine.batch_statistics(out, batch)
            for name, term in statistics.items():
                loss_totals[name][0] += float(term.numerator.item())
                loss_totals[name][1] += term.denominator
            q_valid_total += valid_target_count(batch.q_target, batch.q_target_mask)
            z_valid_total += valid_target_count(batch.z_target, batch.z_target_mask)
            batch_count += 1
            if diagnostic_accumulator is not None:
                diagnostic_accumulator.add_batch(batch, out)

            batch_size = int(out["q"].shape[0])
            group_ids: dict[str, tuple[str, ...] | None] = {
                scope: None for scope in ("event", "graph")
            }
            if collect_group_metrics:
                group_ids = {
                    "event": _batch_group_ids(batch, "event_id", batch_size),
                    "graph": _batch_group_ids(batch, "graph_id", batch_size),
                }
                for scope, identifiers in group_ids.items():
                    present = identifiers is not None
                    previous = metadata_presence[scope]
                    if previous is None:
                        metadata_presence[scope] = present
                    elif previous != present:
                        raise ValueError(
                            f"评估loader的{scope}_id元数据不一致："
                            "部分batch存在、部分缺失"
                        )

            for prefix, pred, target, mask in (
                ("q_", out["q"], batch.q_target, batch.q_target_mask),
                ("z_", out["z"], batch.z_target, batch.z_target_mask),
            ):
                target_name = prefix.rstrip("_")
                physical = masked_regression_sums(pred, target, mask)
                _add_sums(regression_totals[target_name], physical)
                for name, (error_sum, count) in horizon_metric_stats(
                    pred, target, mask
                ).items():
                    key = prefix + name
                    aggregate = metric_totals.setdefault(key, [0.0, 0])
                    aggregate[0] = float(aggregate[0]) + error_sum
                    aggregate[1] = int(aggregate[1]) + count
                for scope, identifiers in group_ids.items():
                    if identifiers is None:
                        continue
                    group_totals = grouped_regression_totals[scope][target_name]
                    for sample_index, identifier in enumerate(identifiers):
                        sample_sums = masked_regression_sums(
                            pred[sample_index : sample_index + 1],
                            target[sample_index : sample_index + 1],
                            mask[sample_index : sample_index + 1],
                        )
                        aggregate = group_totals.setdefault(
                            identifier, _empty_regression_sums()
                        )
                        _add_sums(aggregate, sample_sums)

            q_hydrograph = hydrograph_sample_sums(
                out["q"], batch.q_target, batch.q_target_mask
            )
            _add_sums(hydrograph_totals, q_hydrograph)
            if collect_group_metrics:
                for sample_index in range(batch_size):
                    sample_hydrograph = hydrograph_sample_sums(
                        out["q"][sample_index : sample_index + 1],
                        batch.q_target[sample_index : sample_index + 1],
                        batch.q_target_mask[sample_index : sample_index + 1],
                    )
                    for scope, identifiers in group_ids.items():
                        if identifiers is None:
                            continue
                        identifier = identifiers[sample_index]
                        aggregate = grouped_hydrograph_totals[scope].setdefault(
                            identifier, _empty_hydrograph_sums()
                        )
                        _add_sums(aggregate, sample_hydrograph)

        if batch_count == 0:
            raise ValueError("评估DataLoader为空")
        result = self.loss_engine.report(
            {name: (float(value), int(count)) for name, (value, count) in loss_totals.items()},
            q_valid_count=q_valid_total,
            z_valid_count=z_valid_total,
        )
        result.update(
            {
                name: float(error_sum) / int(count)
                if int(count)
                else float("nan")
                for name, (error_sum, count) in metric_totals.items()
            }
        )
        for prefix, sums in regression_totals.items():
            result.update(
                {
                    f"{prefix}_{name}": value
                    for name, value in regression_metrics(sums).items()
                }
            )
        result.update(
            {
                f"q_{name}": value
                for name, value in _hydrograph_report(hydrograph_totals).items()
            }
        )

        window_group_metrics: dict[str, dict[str, dict[str, Any]]] = {}
        for scope in ("event", "graph"):
            if not metadata_presence[scope]:
                continue
            identifiers = sorted(
                set(grouped_regression_totals[scope]["q"])
                | set(grouped_regression_totals[scope]["z"])
            )
            scope_details: dict[str, dict[str, Any]] = {}
            for identifier in identifiers:
                group_detail: dict[str, Any] = {}
                for target_name in ("q", "z"):
                    sums = grouped_regression_totals[scope][target_name].get(
                        identifier, _empty_regression_sums()
                    )
                    target_report = regression_metrics(sums)
                    if target_name == "q":
                        target_report.update(
                            _hydrograph_report(
                                grouped_hydrograph_totals[scope].get(
                                    identifier, _empty_hydrograph_sums()
                                )
                            )
                        )
                    group_detail[target_name] = target_report
                scope_details[identifier] = group_detail
            window_group_metrics[scope] = scope_details

            for target_name in ("q", "z"):
                reports = [
                    scope_details[identifier][target_name]
                    for identifier in identifiers
                ]
                result[f"{target_name}_{scope}_window_group_count"] = sum(
                    int(report["valid_count"]) > 0 for report in reports
                )
                for metric_name in ("mae", "rmse", "bias", "nse", "kge"):
                    macro, defined_count = _finite_macro(reports, metric_name)
                    base_name = (
                        f"{target_name}_{scope}_window_macro_{metric_name}"
                    )
                    result[base_name] = macro
                    result[f"{base_name}_defined_count"] = defined_count

            q_reports = [
                scope_details[identifier]["q"] for identifier in identifiers
            ]
            for metric_name in (
                "sample_peak_mae",
                "sample_peak_bias",
                "sample_relative_peak_bias",
                "sample_peak_timing_mae_hours",
                "sample_peak_timing_bias_hours",
                "sample_relative_volume_bias",
            ):
                macro, defined_count = _finite_macro(q_reports, metric_name)
                base_name = f"q_{scope}_window_macro_{metric_name}"
                result[base_name] = macro
                result[f"{base_name}_defined_count"] = defined_count

        if include_group_details and window_group_metrics:
            result["window_group_metrics"] = window_group_metrics
        if diagnostic_accumulator is not None:
            validation_diagnostics = diagnostic_accumulator.finalize()
            result.update(validation_diagnostics.summary_metrics)
            if include_diagnostic_details:
                # Private transport used by fit/evaluate entry points.  It is
                # removed before CSV/JSON metric logging and never checkpointed.
                result["_validation_diagnostics"] = validation_diagnostics
        return result

    @staticmethod
    def last_checkpoint_path(path: str | Path) -> Path:
        path = Path(path)
        return path.with_name(f"{path.stem}.last{path.suffix}")

    @staticmethod
    def q_scale_audit_path(path: str | Path) -> Path:
        path = Path(path)
        stem = path.stem
        if stem.endswith("_best"):
            stem = stem[: -len("_best")]
        return path.with_name(f"{stem}_q_scales.json")

    @staticmethod
    def target_scale_audit_path(path: str | Path) -> Path:
        path = Path(path)
        stem = path.stem
        if stem.endswith("_best"):
            stem = stem[: -len("_best")]
        return path.with_name(f"{stem}_target_scales.json")

    def _write_q_scale_audit(self, checkpoint_path: Path) -> Path | None:
        audit = self.cfg.get("_runtime", {}).get("q_scale_audit")
        if audit is None:
            return None
        if not isinstance(audit, dict):
            raise ValueError("_runtime.q_scale_audit必须是JSON对象")
        path = self.q_scale_audit_path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(
            "q_loss_scale_audit",
            json.dumps(
                {"path": str(path.resolve()), **audit},
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        return path

    def _write_target_scale_audit(self, checkpoint_path: Path) -> Path | None:
        audit = self.cfg.get("_runtime", {}).get("target_scale_audit")
        if audit is None:
            return None
        if not isinstance(audit, dict):
            raise ValueError("_runtime.target_scale_audit必须是JSON对象")
        path = self.target_scale_audit_path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("target_scale_audit", json.dumps({"path": str(path.resolve())}, ensure_ascii=False))
        return path

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        try:
            import numpy as np

            state["numpy"] = np.random.get_state()
        except ImportError:  # NumPy is optional for this core trainer.
            pass
        return state

    @staticmethod
    def _loader_generator(loader: Iterable[Any]) -> torch.Generator | None:
        """Find the generator used by a DataLoader or its sampler, if exposed."""

        candidates = (
            getattr(loader, "generator", None),
            getattr(getattr(loader, "sampler", None), "generator", None),
            getattr(getattr(loader, "batch_sampler", None), "generator", None),
            getattr(
                getattr(getattr(loader, "batch_sampler", None), "sampler", None),
                "generator",
                None,
            ),
        )
        return next(
            (item for item in candidates if isinstance(item, torch.Generator)), None
        )

    @staticmethod
    def _set_loader_epoch(loader: Iterable[Any], epoch: int) -> None:
        """Propagate an epoch to custom/distributed samplers exactly once each."""

        candidates = (
            getattr(loader, "batch_sampler", None),
            getattr(loader, "sampler", None),
            getattr(getattr(loader, "batch_sampler", None), "sampler", None),
        )
        seen: set[int] = set()
        for candidate in candidates:
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            setter = getattr(candidate, "set_epoch", None)
            if callable(setter):
                setter(int(epoch))

    def _capture_loader_rng_state(self, loader: Iterable[Any]) -> None:
        generator = self._loader_generator(loader)
        self._train_loader_rng_state = (
            generator.get_state().cpu() if generator is not None else None
        )

    def _restore_loader_rng_state(self, loader: Iterable[Any]) -> None:
        if self._train_loader_rng_state is None:
            return
        generator = self._loader_generator(loader)
        if generator is None:
            warnings.warn(
                "checkpoint含训练DataLoader RNG状态，但当前loader未暴露generator；"
                "batch顺序不能保证精确续接",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        generator.set_state(self._train_loader_rng_state.cpu())
        # Apply a restored state only once.  Later epochs advance and recapture it.
        self._train_loader_rng_state = None

    @staticmethod
    def _restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        torch.set_rng_state(state["torch"].cpu())
        if "cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
        if "numpy" in state:
            try:
                import numpy as np

                np.random.set_state(state["numpy"])
            except ImportError:
                warnings.warn(
                    "checkpoint含NumPy RNG状态，但当前环境未安装NumPy；该状态未恢复",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        metrics: dict[str, float | int],
        *,
        kind: str = "manual",
    ) -> None:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.last_epoch = int(epoch)
        self.last_metrics = dict(metrics)
        payload = {
            "format_version": 3,
            "checkpoint_kind": kind,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.last_epoch,
            "best": self.best,
            "selection_metric": self.selection_metric_name,
            "selection_direction": self.selection_direction,
            "stale": self.stale,
            "last_epoch": self.last_epoch,
            "last_metrics": self.last_metrics,
            # Keep the old field for downstream readers.
            "metrics": self.last_metrics,
            "rng_state": self._capture_rng_state(),
            "train_loader_rng_state": self._train_loader_rng_state,
            "config": self.cfg,
        }

        # The temporary file must live beside the destination so os.replace is
        # a same-filesystem atomic operation.  Passing the already-open stream
        # to torch.save avoids Windows path-sharing conflicts.
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                torch.save(payload, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException as exc:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                note = (
                    f"清理失败的checkpoint临时文件失败: {temporary_path}: "
                    f"{cleanup_error}"
                )
                add_note = getattr(exc, "add_note", None)
                if callable(add_note):
                    add_note(note)
                else:  # pragma: no cover - Python 3.10 compatibility
                    warnings.warn(note, RuntimeWarning, stacklevel=2)
            raise

    @staticmethod
    def _read_checkpoint(path: str | Path) -> Any:
        return torch.load(Path(path), map_location="cpu", weights_only=False)

    def load_weights(
        self, path: str | Path, *, strict: bool = True
    ) -> dict[str, Any]:
        """Load model weights only, without changing optimizer/epoch/RNG state."""

        checkpoint = self._read_checkpoint(path)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        elif isinstance(checkpoint, dict):
            # Also accept a plain state_dict.
            state = checkpoint
            checkpoint = {"model": checkpoint}
        else:
            raise TypeError("checkpoint必须是字典或模型state_dict")
        self.model.load_state_dict(state, strict=strict)
        return checkpoint

    def resume_checkpoint(
        self,
        path: str | Path,
        *,
        strict: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Strictly resume training state; older v1 checkpoints remain usable."""

        checkpoint = self._read_checkpoint(path)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError("续训checkpoint缺少model状态；纯权重文件请使用load_weights")
        self.model.load_state_dict(checkpoint["model"], strict=strict)
        if "optimizer" not in checkpoint:
            raise ValueError("续训checkpoint缺少optimizer状态；请改用load_weights")
        try:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, KeyError) as exc:
            raise ValueError(
                "optimizer参数组与当前模型不兼容；若改变了可训练参数，请使用load_weights"
            ) from exc
        if "epoch" not in checkpoint:
            raise ValueError("续训checkpoint缺少epoch")

        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        else:
            warnings.warn(
                "旧checkpoint没有AMP scaler状态；已按新scaler继续",
                RuntimeWarning,
                stacklevel=2,
            )
        self.last_epoch = int(checkpoint.get("last_epoch", checkpoint["epoch"]))
        self.start_epoch = self.last_epoch + 1
        metrics = checkpoint.get("last_metrics", checkpoint.get("metrics", {}))
        self.last_metrics = dict(metrics) if isinstance(metrics, dict) else {}
        saved_metric = checkpoint.get("selection_metric")
        saved_direction = checkpoint.get("selection_direction")
        if saved_metric is not None and saved_metric != self.selection_metric_name:
            raise ValueError(
                "checkpoint选择指标与当前配置不一致: "
                f"saved={saved_metric}, current={self.selection_metric_name}"
            )
        if saved_direction is not None and saved_direction != self.selection_direction:
            raise ValueError(
                "checkpoint选择方向与当前配置不一致: "
                f"saved={saved_direction}, current={self.selection_direction}"
            )
        default_best = (
            float("-inf") if self.selection_direction == "maximize" else float("inf")
        )
        fallback_best = self.last_metrics.get(
            self.selection_metric_name,
            self.last_metrics.get("loss", default_best),
        )
        self.best = float(checkpoint.get("best", fallback_best))
        self.stale = int(checkpoint.get("stale", 0))
        self._train_loader_rng_state = checkpoint.get("train_loader_rng_state")

        if restore_rng:
            if "rng_state" in checkpoint:
                self._restore_rng_state(checkpoint["rng_state"])
            else:
                warnings.warn(
                    "旧checkpoint没有RNG状态；模型/优化器已恢复，但随机序列不能精确续接",
                    RuntimeWarning,
                    stacklevel=2,
                )
        return checkpoint

    def load_checkpoint(
        self,
        path: str | Path,
        *,
        resume: bool = True,
        strict: bool = True,
        restore_rng: bool = True,
    ) -> dict[str, Any]:
        """Compatibility wrapper; new code should choose resume or weights explicitly."""

        if resume:
            return self.resume_checkpoint(
                path, strict=strict, restore_rng=restore_rng
            )
        return self.load_weights(path, strict=strict)

    def fit(
        self,
        train_loader: Iterable[Any],
        val_loader: Iterable[Any] | None = None,
        *,
        overwrite: bool = False,
        epoch_callback: Callable[[int, dict[str, float | int]], None] | None = None,
    ) -> list[dict[str, float | int]]:
        history: list[dict[str, float | int]] = []
        log = Path(self.cfg["training"]["log_csv"])
        log.parent.mkdir(parents=True, exist_ok=True)
        report = device_report(self.device)
        print(
            "runtime",
            report,
            "amp",
            self.amp,
            "batch",
            self.cfg["batch_size"],
            "accum",
            self.cfg["gradient_accumulation_steps"],
        )
        patience = int(self.cfg["training"]["patience"])
        early_stopping = bool(self.cfg["training"].get("early_stopping", True))
        checkpoint_path = Path(self.cfg["training"]["checkpoint"])
        last_path = self.last_checkpoint_path(checkpoint_path)
        final_checkpoint_value = self.cfg["training"].get("final_checkpoint")
        final_path = Path(final_checkpoint_value) if final_checkpoint_value else None
        audit_path = (
            self.q_scale_audit_path(checkpoint_path)
            if self.cfg.get("_runtime", {}).get("q_scale_audit") is not None
            else None
        )
        target_audit_path = (
            self.target_scale_audit_path(checkpoint_path)
            if self.cfg.get("_runtime", {}).get("target_scale_audit") is not None
            else None
        )
        if self.start_epoch == 0:
            existing = [
                path
                for path in (
                    log,
                    checkpoint_path,
                    last_path,
                    final_path,
                    audit_path,
                    target_audit_path,
                )
                if path is not None and path.exists()
            ]
            if existing and not overwrite:
                raise FileExistsError(
                    "检测到已有训练输出；为防止混合日志或覆盖checkpoint，请改用新输出路径、"
                    f"--resume，或显式--overwrite: {[str(path) for path in existing]}"
                )
            if overwrite and log.exists():
                log.write_text("", encoding="utf-8")
        self._write_q_scale_audit(checkpoint_path)
        self._write_target_scale_audit(checkpoint_path)
        self._restore_loader_rng_state(train_loader)

        for epoch in range(self.start_epoch, int(self.cfg["training"]["epochs"])):
            if early_stopping and self.stale >= patience:
                break
            self._set_loader_epoch(train_loader, epoch)
            tick = time.perf_counter()
            row: dict[str, float | int] = {
                "epoch": epoch,
                **self.train_epoch(train_loader, epoch),
            }
            row["epoch_seconds"] = time.perf_counter() - tick
            validation_diagnostics: ValidationDiagnostics | None = None
            validation_summary: dict[str, float | int] = {}
            if val_loader is not None:
                formal_validation = (
                    self.cfg.get("data", {}).get("mode") == "hunan"
                    and self.cfg.get("data", {}).get("dataset_type", "event") == "event"
                )
                validation_metrics = self.evaluate(
                    val_loader,
                    include_validation_diagnostics=formal_validation,
                    include_diagnostic_details=formal_validation,
                )
                validation_diagnostics = validation_metrics.pop(
                    "_validation_diagnostics", None
                )
                if validation_diagnostics is not None:
                    validation_summary = {
                        key: validation_metrics.pop(key)
                        for key in validation_diagnostics.summary_metrics
                    }
                row.update(
                    {f"val_{key}": value for key, value in validation_metrics.items()}
                )
            row.update(self._selection_report(row, validation_summary))
            score = float(row.get(self.selection_metric_name, row["loss"]))
            if not math.isfinite(score):
                raise FloatingPointError(
                    f"{self.selection_metric_name}早停指标出现NaN/Inf"
                )
            peak = (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            )
            row["peak_gpu_bytes"] = peak
            # Preserve the historical column order and append only new fields.
            row.update(
                {f"val_{key}": value for key, value in validation_summary.items()}
            )
            history.append(row)
            print(row)
            _append_csv_row(log, row)
            if epoch_callback is not None:
                epoch_callback(epoch, row)

            improved = (
                score > self.best
                if self.selection_direction == "maximize"
                else score < self.best
            )
            if improved:
                self.best = score
                self.stale = 0
            else:
                self.stale += 1
            self.last_epoch = epoch
            self.last_metrics = dict(row)
            self._capture_loader_rng_state(train_loader)
            # The configured path remains the best checkpoint.  A deterministic
            # sibling always records the exact last epoch for strict resumption.
            self.save_checkpoint(last_path, epoch, row, kind="last")
            if (
                final_path is not None
                and epoch == int(self.cfg["training"]["epochs"]) - 1
            ):
                self.save_checkpoint(final_path, epoch, row, kind="final")
            if improved:
                self.save_checkpoint(checkpoint_path, epoch, row, kind="best")
                if validation_diagnostics is not None:
                    diagnostics_dir = checkpoint_path.parent / (
                        f"{checkpoint_path.stem}_validation_diagnostics"
                    )
                    validation_diagnostics.write(
                        diagnostics_dir,
                        split="VALIDATION",
                        context={
                            "epoch": epoch,
                            "checkpoint": str(checkpoint_path.resolve()),
                            "selection_metric": self.selection_metric_name,
                            "selection_metric_value": score,
                        },
                    )
            if early_stopping and self.stale >= patience:
                break
        return history
