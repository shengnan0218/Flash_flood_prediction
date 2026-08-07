"""Training, evaluation and restart support for flood-forecast models."""

from __future__ import annotations

import csv
import math
import os
import random
import tempfile
import time
import warnings
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import torch

from data.device import device_report
from metrics.flood_metrics import (
    hydrograph_sample_sums,
    horizon_metric_stats,
    masked_huber_stats,
    masked_regression_sums,
    regression_metrics,
    valid_target_count,
)


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
        self.amp = bool(cfg["amp"] and device.type == "cuda")
        # torch.amp.GradScaler was not exported until PyTorch 2.3.  Keep the
        # supported >=2.2 environment usable without changing checkpoint format.
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        else:  # pragma: no cover - exercised only by older supported PyTorch
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.best = float("inf")
        self.start_epoch = 0
        self.stale = 0
        self.last_epoch = -1
        self.last_metrics: dict[str, float | int] = {}
        self._train_loader_rng_state: torch.Tensor | None = None

    def _loss_sums(
        self, out: dict[str, Any], batch: Any
    ) -> tuple[torch.Tensor, int, torch.Tensor, int]:
        runtime_scales = self.cfg.get("_runtime", {}).get("loss_scales", {})
        q_scale = float(runtime_scales.get("discharge", 1.0))
        z_scale = float(runtime_scales.get("water_level", 1.0))
        if (
            not math.isfinite(q_scale)
            or not math.isfinite(z_scale)
            or q_scale <= 0
            or z_scale <= 0
        ):
            raise ValueError(
                "TRAIN目标标准差必须为有限正数，"
                f"实际discharge={q_scale}, water_level={z_scale}"
            )
        q_sum, q_count = masked_huber_stats(
            out["q"] / q_scale,
            batch.q_target / q_scale,
            batch.q_target_mask,
        )
        z_sum, z_count = masked_huber_stats(
            out["z"] / z_scale,
            batch.z_target / z_scale,
            batch.z_target_mask,
        )
        return q_sum, q_count, z_sum, z_count

    def _loss_weights(self) -> tuple[float, float]:
        values = (
            float(self.cfg["loss_weights"]["discharge"]),
            float(self.cfg["loss_weights"]["water_level"]),
        )
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f"loss_weights必须是有限非负数，实际为{values}")
        return values

    def _weighted_loss(
        self,
        q_sum: torch.Tensor,
        q_count: int,
        z_sum: torch.Tensor,
        z_count: int,
    ) -> torch.Tensor:
        if q_count + z_count == 0:
            raise ValueError("当前batch没有任何有效Q/Z监督目标")
        q_weight, z_weight = self._loss_weights()
        loss = (q_sum + z_sum) * 0.0
        if q_count:
            loss = loss + q_weight * q_sum / q_count
        if z_count:
            loss = loss + z_weight * z_sum / z_count
        return loss

    def _reported_losses(
        self,
        q_sum: float,
        q_count: int,
        z_sum: float,
        z_count: int,
    ) -> dict[str, float | int]:
        if q_count + z_count == 0:
            raise ValueError("数据中没有任何有效Q/Z监督目标")
        q_loss = q_sum / q_count if q_count else float("nan")
        z_loss = z_sum / z_count if z_count else float("nan")
        q_weight, z_weight = self._loss_weights()
        total = 0.0
        if q_count:
            total += q_weight * q_loss
        if z_count:
            total += z_weight * z_loss
        return {
            "loss": total,
            "q_loss": q_loss,
            "z_loss": z_loss,
            "q_valid_count": q_count,
            "z_valid_count": z_count,
        }

    def _loss(
        self, out: dict[str, Any], batch: Any
    ) -> tuple[torch.Tensor, dict[str, float | int]]:
        """Backwards-compatible per-batch mean loss helper."""

        q_sum, q_count, z_sum, z_count = self._loss_sums(out, batch)
        loss = self._weighted_loss(q_sum, q_count, z_sum, z_count)
        parts = self._reported_losses(
            float(q_sum.detach().item()),
            q_count,
            float(z_sum.detach().item()),
            z_count,
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

        q_total = 0.0
        z_total = 0.0
        q_count_total = 0
        z_count_total = 0
        batch_times: list[float] = []
        substeps: list[float] = []
        batch_count = 0

        for group in self._batch_groups(loader, accumulation):
            # Knowing the complete group's denominators before forward passes
            # makes accumulation exactly valid-element weighted and also gives a
            # short final group the correct (not 1/accumulation) scale.
            q_group_count = sum(
                valid_target_count(batch.q_target, batch.q_target_mask)
                for batch in group
            )
            z_group_count = sum(
                valid_target_count(batch.z_target, batch.z_target_mask)
                for batch in group
            )
            if q_group_count + z_group_count == 0:
                raise ValueError("一个梯度累积组内没有任何有效Q/Z监督目标")
            q_weight, z_weight = self._loss_weights()

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
                        q_sum, q_count, z_sum, z_count = self._loss_sums(out, batch)
                        contribution = (q_sum + z_sum) * 0.0
                        if q_group_count:
                            contribution = (
                                contribution + q_weight * q_sum / q_group_count
                            )
                        if z_group_count:
                            contribution = (
                                contribution + z_weight * z_sum / z_group_count
                            )
                    if not torch.isfinite(contribution.detach()).all():
                        raise FloatingPointError("训练损失出现NaN/Inf")
                    self.scaler.scale(contribution).backward()

                    q_total += float(q_sum.detach().item())
                    z_total += float(z_sum.detach().item())
                    q_count_total += q_count
                    z_count_total += z_count
                    batch_count += 1
                    batch_times.append(time.perf_counter() - tick)
                    diagnostic = out.get("diagnostics", {}).get("substeps")
                    if isinstance(diagnostic, torch.Tensor) and diagnostic.numel():
                        substeps.append(float(diagnostic.float().mean().item()))

                self.scaler.unscale_(self.optimizer)
                self._clip_gradients()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            except torch.OutOfMemoryError as exc:
                self.optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "CUDA显存不足：请降低batch_size、history_length、solver cells(dx相关)"
                    "或maximum_substeps；配置不会被静默修改"
                ) from exc

        if batch_count == 0:
            raise ValueError("训练DataLoader为空（或debug_max_batches=0）")
        result = self._reported_losses(
            q_total, q_count_total, z_total, z_count_total
        )
        result.update(
            {
                "batch_seconds": sum(batch_times) / len(batch_times),
                "wave_mean_substeps": (
                    sum(substeps) / len(substeps) if substeps else float("nan")
                ),
            }
        )
        return result

    @torch.no_grad()
    def evaluate(
        self,
        loader: Iterable[Any],
        *,
        include_group_metrics: bool = False,
        include_group_details: bool = False,
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
        q_total = 0.0
        z_total = 0.0
        q_count_total = 0
        z_count_total = 0
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
            q_sum, q_count, z_sum, z_count = self._loss_sums(out, batch)
            q_total += float(q_sum.item())
            z_total += float(z_sum.item())
            q_count_total += q_count
            z_count_total += z_count
            batch_count += 1

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
        result = self._reported_losses(
            q_total, q_count_total, z_total, z_count_total
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
        return result

    @staticmethod
    def last_checkpoint_path(path: str | Path) -> Path:
        path = Path(path)
        return path.with_name(f"{path.stem}.last{path.suffix}")

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
            "format_version": 2,
            "checkpoint_kind": kind,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "epoch": self.last_epoch,
            "best": self.best,
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
        fallback_best = self.last_metrics.get(
            "val_loss", self.last_metrics.get("loss", float("inf"))
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
        checkpoint_path = Path(self.cfg["training"]["checkpoint"])
        last_path = self.last_checkpoint_path(checkpoint_path)
        if self.start_epoch == 0:
            existing = [
                path
                for path in (log, checkpoint_path, last_path)
                if path.exists()
            ]
            if existing and not overwrite:
                raise FileExistsError(
                    "检测到已有训练输出；为防止混合日志或覆盖checkpoint，请改用新输出路径、"
                    f"--resume，或显式--overwrite: {[str(path) for path in existing]}"
                )
            if overwrite and log.exists():
                log.write_text("", encoding="utf-8")
        self._restore_loader_rng_state(train_loader)

        for epoch in range(self.start_epoch, int(self.cfg["training"]["epochs"])):
            if self.stale >= patience:
                break
            self._set_loader_epoch(train_loader, epoch)
            tick = time.perf_counter()
            row: dict[str, float | int] = {
                "epoch": epoch,
                **self.train_epoch(train_loader, epoch),
            }
            row["epoch_seconds"] = time.perf_counter() - tick
            if val_loader is not None:
                row.update(
                    {f"val_{key}": value for key, value in self.evaluate(val_loader).items()}
                )
            score = float(row.get("val_loss", row["loss"]))
            if not math.isfinite(score):
                raise FloatingPointError("早停指标出现NaN/Inf")
            peak = (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda"
                else 0
            )
            row["peak_gpu_bytes"] = peak
            history.append(row)
            print(row)
            with log.open("a", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=row.keys())
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)

            improved = score < self.best
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
            if improved:
                self.save_checkpoint(checkpoint_path, epoch, row, kind="best")
            if self.stale >= patience:
                break
        return history
