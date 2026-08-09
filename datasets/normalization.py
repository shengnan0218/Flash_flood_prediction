"""Training-statistics loaders and small tensor scaler utilities."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(frozen=True)
class FeatureStatistics:
    mean: float
    std: float
    minimum: float | None = None
    maximum: float | None = None

    @classmethod
    def from_mapping(cls, name: str, value: Mapping[str, Any]) -> "FeatureStatistics":
        try:
            mean = float(value["mean"])
            std = float(value["std"])
            minimum = None if value.get("min", value.get("minimum")) is None else float(value.get("min", value.get("minimum")))
            maximum = None if value.get("max", value.get("maximum")) is None else float(value.get("max", value.get("maximum")))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"normalization_stats.json中{name}必须包含数值mean/std") from exc
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"normalization_stats.json中{name}的mean必须有限且std必须>0")
        if minimum is not None and not math.isfinite(minimum):
            raise ValueError(f"normalization_stats.json中{name}.min必须为有限数")
        if maximum is not None and not math.isfinite(maximum):
            raise ValueError(f"normalization_stats.json中{name}.max必须为有限数")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"normalization_stats.json中{name}的min不能大于max")
        return cls(mean, std, minimum, maximum)


class NormalizationStats:
    """Immutable feature statistics loaded once and reused for every split.

    The loader never fits statistics on validation/test data.  If the JSON has a
    provenance field (``computed_from_split`` or ``split``), it must say TRAIN.
    """

    def __init__(self, features: Mapping[str, FeatureStatistics], source: Path) -> None:
        self._features = {name.upper(): stats for name, stats in features.items()}
        self.source = source

    @classmethod
    def load(
        cls,
        path: str | Path,
        required: tuple[str, ...] = (),
        *,
        require_train_provenance: bool = False,
    ) -> "NormalizationStats":
        source = Path(path)
        try:
            raw = json.loads(source.read_text(encoding="utf-8-sig"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"缺少训练集标准化参数文件: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"标准化参数JSON无效: {source}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"标准化参数根节点必须是JSON对象: {source}")
        provenance = raw.get("computed_from_split", raw.get("split"))
        if require_train_provenance and provenance is None:
            raise ValueError(
                "正式normalization_stats.json必须声明"
                f"computed_from_split=TRAIN，无法验证统计量无泄漏: {source}"
            )
        if provenance is not None and str(provenance).strip().upper() not in {"TRAIN", "TRAINING"}:
            raise ValueError(f"标准化参数必须只由TRAIN计算，文件声明为{provenance!r}: {source}")

        # Support the suggested flat JSON as well as {"features": {...}}.
        container = raw.get("features", raw)
        if not isinstance(container, dict):
            raise ValueError(f"normalization_stats.json的features必须是对象: {source}")
        features: dict[str, FeatureStatistics] = {}
        for name, value in container.items():
            if str(name).lower() in {"computed_from_split", "split", "node_static", "edge_static", "metadata"}:
                continue
            if isinstance(value, dict) and "mean" in value and "std" in value:
                features[str(name).upper()] = FeatureStatistics.from_mapping(str(name), value)
        missing = [name for name in required if name.upper() not in features]
        if missing:
            raise ValueError(f"normalization_stats.json缺少必需训练统计量: {missing}")
        return cls(features, source)

    def __contains__(self, name: str) -> bool:
        return name.upper() in self._features

    def __getitem__(self, name: str) -> FeatureStatistics:
        key = name.upper()
        if key not in self._features:
            raise KeyError(f"标准化参数缺少特征{key}: {self.source}")
        return self._features[key]

    def transform(self, name: str, value: torch.Tensor, mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        stats = self[name]
        valid = torch.isfinite(value) if mask is None else mask.bool() & torch.isfinite(value)
        transformed = torch.zeros_like(value, dtype=torch.float32)
        transformed[valid] = (value[valid].float() - stats.mean) / stats.std
        ood = torch.zeros_like(valid)
        if stats.minimum is not None:
            ood |= valid & value.lt(stats.minimum)
        if stats.maximum is not None:
            ood |= valid & value.gt(stats.maximum)
        return transformed, ood


@dataclass
class HunanScaler:
    """Backward-compatible vector scaler used by older synthetic experiments."""

    mean: list[float]
    std: list[float]
    minimum: list[float]
    maximum: list[float]

    @classmethod
    def fit(cls, x: torch.Tensor, mask: torch.Tensor | None = None) -> "HunanScaler":
        if x.ndim == 0:
            raise ValueError("HunanScaler.fit至少需要一个特征维")
        flat = x.reshape(-1, x.shape[-1]).float()
        valid = torch.isfinite(flat) if mask is None else mask.reshape_as(flat).bool() & torch.isfinite(flat)
        if not valid.any(0).all():
            missing = (~valid.any(0)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"以下特征没有任何有效训练值，无法拟合标准化参数: {missing}")
        means, stds, minima, maxima = [], [], [], []
        for feature in range(flat.shape[1]):
            values = flat[:, feature][valid[:, feature]]
            means.append(values.mean().item())
            stds.append(values.std(unbiased=False).clamp_min(1e-6).item())
            minima.append(values.min().item())
            maxima.append(values.max().item())
        return cls(means, stds, minima, maxima)

    def transform(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = torch.as_tensor(self.mean, device=x.device, dtype=x.dtype)
        std = torch.as_tensor(self.std, device=x.device, dtype=x.dtype)
        lower = torch.as_tensor(self.minimum, device=x.device, dtype=x.dtype)
        upper = torch.as_tensor(self.maximum, device=x.device, dtype=x.dtype)
        finite = torch.isfinite(x)
        transformed = torch.where(finite, (x - mean) / std, torch.zeros_like(x))
        return transformed, finite & (x.lt(lower) | x.gt(upper))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HunanScaler":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig")))
