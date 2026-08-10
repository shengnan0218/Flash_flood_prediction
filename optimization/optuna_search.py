"""Optuna TPE + median-pruning search restricted to shared E4 parameters."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import optuna
import yaml

from config import load_config, validate_config
from scripts.common import setup_training_from_config
from trainers import Trainer


ALLOWED_SEARCH_PARAMETERS = frozenset(
    {
        "learning_rate",
        "weight_decay",
        "hidden_dim",
        "q_peak_weight",
        "q_volume_weight",
        "z_slope_weight",
    }
)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def trial_output_directory(output_root: str | Path, trial_number: int) -> Path:
    if isinstance(trial_number, bool) or not isinstance(trial_number, int) or trial_number < 0:
        raise ValueError("trial_number必须是非负整数")
    return Path(output_root) / f"trial_{trial_number:04d}"


def build_sampler(config: Mapping[str, Any]) -> optuna.samplers.TPESampler:
    sampler = config["hyperparameter_optimization"]["sampler"]
    if sampler["name"] != "tpe":
        raise ValueError("HPO sampler必须是tpe")
    return optuna.samplers.TPESampler(seed=int(sampler["seed"]))


def build_pruner(config: Mapping[str, Any]) -> optuna.pruners.MedianPruner:
    pruner = config["hyperparameter_optimization"]["pruner"]
    if pruner["name"] != "median":
        raise ValueError("HPO pruner必须是median")
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(pruner["n_startup_trials"]),
        n_warmup_steps=int(pruner["n_warmup_steps"]),
        interval_steps=int(pruner["interval_steps"]),
    )


def sample_shared_parameters(
    trial: optuna.trial.Trial, search_space: Mapping[str, Any]
) -> dict[str, float | int]:
    """Sample exactly the six authorised shared parameters and nothing else."""

    if set(search_space) != ALLOWED_SEARCH_PARAMETERS:
        raise ValueError(
            "HPO search_space必须且只能包含共享参数，实际="
            f"{sorted(search_space)}"
        )
    sampled: dict[str, float | int] = {}
    for name in sorted(ALLOWED_SEARCH_PARAMETERS):
        specification = search_space[name]
        if specification["type"] == "log_float":
            sampled[name] = trial.suggest_float(
                name,
                float(specification["low"]),
                float(specification["high"]),
                log=True,
            )
        elif specification["type"] == "categorical":
            sampled[name] = int(
                trial.suggest_categorical(name, list(specification["choices"]))
            )
        else:
            raise ValueError(f"不支持的search type: {specification['type']!r}")
    return sampled


def apply_trial_parameters(
    base_config: Mapping[str, Any], parameters: Mapping[str, float | int]
) -> dict[str, Any]:
    if set(parameters) != ALLOWED_SEARCH_PARAMETERS:
        raise ValueError(f"trial参数越权或缺失: {sorted(parameters)}")
    config = deepcopy(dict(base_config))
    config["optimizer"]["lr"] = float(parameters["learning_rate"])
    config["optimizer"]["weight_decay"] = float(parameters["weight_decay"])
    config["hidden_dim"] = int(parameters["hidden_dim"])
    for name in ("q_peak_weight", "q_volume_weight", "z_slope_weight"):
        config["loss"][name] = float(parameters[name])
    return validate_config(config)


def report_selection_for_pruning(
    trial: optuna.trial.Trial,
    epoch: int,
    row: Mapping[str, float | int],
) -> float:
    """Report and prune exclusively on validation_selection_score."""

    score = float(row["validation_selection_score"])
    if not math.isfinite(score):
        raise FloatingPointError("HPO selection score出现NaN/Inf")
    trial.report(score, step=epoch)
    if trial.should_prune():
        raise optuna.TrialPruned(
            f"trial={trial.number}, epoch={epoch}, score={score}"
        )
    return score


def _assert_e4_hpo_contract(config: Mapping[str, Any]) -> None:
    expected = {
        "runoff_mode": "water_balance_lstm",
        "routing_mode": "kinematic_wave_gnn",
    }
    mismatches = {
        key: config.get(key)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"HPO只允许E4完整模型: {mismatches}")
    if config["loss"]["mode"] != "multitask":
        raise ValueError("HPO要求multitask loss")
    if config["validation_selection"]["mode"] != "composite":
        raise ValueError("HPO要求composite validation selection")
    if config["batch_size"] != 16 or config["history_length"] != 24 or config["forecast_horizon"] != 6:
        raise ValueError("HPO固定要求batch/history/forecast=16/24/6")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _trial_summary(
    trial: optuna.trial.Trial,
    config: Mapping[str, Any],
    trial_dir: Path,
    reports: list[tuple[int, float]],
    status: str,
) -> dict[str, Any]:
    best_epoch, best_score = (
        max(reports, key=lambda item: item[1])
        if reports
        else (None, None)
    )
    return {
        "trial_number": trial.number,
        "status": status,
        "sampled_params": dict(trial.params),
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "checkpoint_path": str(Path(config["training"]["checkpoint"]).resolve()),
        "train_log_path": str(Path(config["training"]["log_csv"]).resolve()),
        "random_seed": int(config["seed"]),
        "selection_metric": "validation_selection_score",
        "selection_direction": "maximize",
        "trial_output_dir": str(trial_dir.resolve()),
    }


def run_optimization(
    config_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    n_trials: int | None = None,
    explicitly_enabled: bool = False,
) -> optuna.study.Study:
    """Run an explicitly authorised E4 study without ever constructing TEST."""

    base_config = load_config(config_path)
    hpo = base_config["hyperparameter_optimization"]
    if not bool(hpo["enabled"]) and not explicitly_enabled:
        raise RuntimeError(
            "hyperparameter optimization默认关闭；必须显式传入--enable才会启动"
        )
    _assert_e4_hpo_contract(base_config)
    trial_count = int(hpo["n_trials"] if n_trials is None else n_trials)
    if trial_count <= 0:
        raise ValueError("n_trials必须大于0")
    output_root = Path(hpo["output_dir"]).expanduser()
    if not output_root.is_absolute():
        output_root = (_PROJECT_ROOT / output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    storage_path = output_root / "optuna_study.db"
    study = optuna.create_study(
        study_name="hunan_e4_multitask_v1",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=build_sampler(base_config),
        pruner=build_pruner(base_config),
    )

    def objective(trial: optuna.trial.Trial) -> float:
        parameters = sample_shared_parameters(trial, hpo["search_space"])
        trial_config = apply_trial_parameters(base_config, parameters)
        trial_dir = trial_output_directory(output_root, trial.number)
        trial_dir.mkdir(parents=True, exist_ok=False)
        trial_config["training"]["checkpoint"] = str(trial_dir / "best.pt")
        trial_config["training"]["log_csv"] = str(trial_dir / "train.csv")
        (trial_dir / "trial_config.yaml").write_text(
            yaml.safe_dump(trial_config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        runtime_config, model, train_loader, validation_loader, device = (
            setup_training_from_config(
                trial_config,
                dataset_root=dataset_root,
            )
        )
        trainer = Trainer(model, runtime_config, device)
        reports: list[tuple[int, float]] = []

        def report_epoch(epoch: int, row: dict[str, float | int]) -> None:
            score = float(row["validation_selection_score"])
            reports.append((epoch, score))
            report_selection_for_pruning(trial, epoch, row)

        try:
            trainer.fit(
                train_loader,
                validation_loader,
                epoch_callback=report_epoch,
            )
        except optuna.TrialPruned:
            summary = _trial_summary(
                trial, runtime_config, trial_dir, reports, "PRUNED"
            )
            _write_json(trial_dir / "trial_summary.json", summary)
            for checkpoint in trial_dir.glob("*.pt"):
                checkpoint.unlink()
            raise
        if not reports:
            raise RuntimeError("trial没有产生任何VALIDATION selection score")
        summary = _trial_summary(
            trial, runtime_config, trial_dir, reports, "COMPLETE"
        )
        _write_json(trial_dir / "trial_summary.json", summary)
        trial.set_user_attr("best_epoch", summary["best_epoch"])
        trial.set_user_attr(
            "best_validation_selection_score",
            summary["best_validation_selection_score"],
        )
        trial.set_user_attr("checkpoint_path", summary["checkpoint_path"])
        return float(summary["best_validation_selection_score"])

    study.optimize(objective, n_trials=trial_count)
    study.trials_dataframe().to_csv(output_root / "study_trials.csv", index=False)
    complete = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not complete:
        raise RuntimeError("HPO结束但没有完成的trial；未生成虚假的best_params")
    best = study.best_trial
    _write_json(output_root / "best_params.json", dict(best.params))
    _write_json(
        output_root / "best_trial_summary.json",
        {
            "trial_number": best.number,
            "best_validation_selection_score": best.value,
            "best_epoch": best.user_attrs.get("best_epoch"),
            "checkpoint_path": best.user_attrs.get("checkpoint_path"),
            "sampled_params": dict(best.params),
        },
    )
    snippet = {
        "hidden_dim": int(best.params["hidden_dim"]),
        "optimizer": {
            "lr": float(best.params["learning_rate"]),
            "weight_decay": float(best.params["weight_decay"]),
        },
        "loss": {
            "q_peak_weight": float(best.params["q_peak_weight"]),
            "q_volume_weight": float(best.params["q_volume_weight"]),
            "z_slope_weight": float(best.params["z_slope_weight"]),
        },
    }
    (output_root / "shared_hyperparameters.yaml").write_text(
        yaml.safe_dump(snippet, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return study
