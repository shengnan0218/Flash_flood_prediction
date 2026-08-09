from __future__ import annotations

import csv
import math
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from metrics import horizon_metrics, masked_huber
from trainers import Trainer
from trainers.trainer import _append_csv_row


@dataclass
class TinyBatch:
    x: torch.Tensor
    q_target: torch.Tensor
    z_target: torch.Tensor
    q_target_mask: torch.Tensor
    z_target_mask: torch.Tensor

    def to(self, device: torch.device) -> "TinyBatch":
        return TinyBatch(
            self.x.to(device),
            self.q_target.to(device),
            self.z_target.to(device),
            self.q_target_mask.to(device),
            self.z_target_mask.to(device),
        )


class TinyModel(torch.nn.Module):
    def __init__(self, initial: float = 0.0) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(initial))

    def forward(self, batch: TinyBatch) -> dict:
        prediction = self.weight * batch.x
        return {
            "q": prediction,
            "z": prediction,
            "diagnostics": {
                "explicit_equivalent_substeps": torch.ones(1)
            },
        }


class RecordingBatchSampler:
    def __init__(self) -> None:
        self.generator = torch.Generator().manual_seed(81)
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)
        self.generator.manual_seed(81 + epoch)


class EpochAwareLoader:
    def __init__(self, item: TinyBatch) -> None:
        self.item = item
        self.batch_sampler = RecordingBatchSampler()
        self.sampler = None

    def __iter__(self):
        yield self.item


def batch(
    targets: list[list[list[float]]],
    masks: list[list[list[bool]]],
) -> TinyBatch:
    target = torch.tensor(targets, dtype=torch.float32)
    mask = torch.tensor(masks, dtype=torch.bool)
    return TinyBatch(torch.ones_like(target), target, target.clone(), mask, mask.clone())


def concatenate(*batches: TinyBatch) -> TinyBatch:
    return TinyBatch(
        *(torch.cat([getattr(item, field) for item in batches], dim=0) for field in (
            "x",
            "q_target",
            "z_target",
            "q_target_mask",
            "z_target_mask",
        ))
    )


def config(directory: Path, accumulation: int = 1) -> dict:
    return {
        "optimizer": {"lr": 0.01, "weight_decay": 0.0},
        "amp": False,
        "loss_weights": {"discharge": 1.0, "water_level": 1.0},
        "gradient_accumulation_steps": accumulation,
        "batch_size": 1,
        "debug_mode": False,
        "training": {
            "epochs": 2,
            "patience": 2,
            "gradient_clip": 100.0,
            "checkpoint": str(directory / "best.pt"),
            "log_csv": str(directory / "train.csv"),
        },
    }


class TestTrainerMetrics(unittest.TestCase):
    def test_training_csv_appends_new_diagnostic_columns_on_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.csv"
            _append_csv_row(path, {"epoch": 0, "loss": 1.0})
            _append_csv_row(
                path,
                {"epoch": 1, "loss": 0.8, "val_delta_z_mae": 0.2},
            )
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                list(rows[0]), ["epoch", "loss", "val_delta_z_mae"]
            )
            self.assertEqual(rows[0]["val_delta_z_mae"], "")
            self.assertEqual(rows[1]["val_delta_z_mae"], "0.2")

    def test_formal_loss_is_dimensionless_by_train_standard_deviation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cfg = config(Path(directory))
            cfg["_runtime"] = {
                "loss_scales": {"discharge": 2.0, "water_level": 4.0}
            }
            trainer = Trainer(TinyModel(), cfg, torch.device("cpu"))
            item = batch([[[2.0]]], [[[True]]])
            zero = torch.zeros_like(item.q_target, requires_grad=True)
            loss, parts = trainer._loss({"q": zero, "z": zero}, item)
            self.assertAlmostEqual(loss.item(), 0.625, places=6)
            self.assertAlmostEqual(float(parts["q_loss"]), 0.5, places=6)
            self.assertAlmostEqual(float(parts["z_loss"]), 0.125, places=6)

    def test_accumulation_flushes_tail_with_full_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = TinyModel()
            trainer = Trainer(
                model, config(Path(directory), accumulation=2), torch.device("cpu")
            )
            observed: list[tuple[float, float]] = []
            original_step = trainer.optimizer.step

            def capture_step(*args, **kwargs):
                observed.append((model.weight.item(), model.weight.grad.item()))
                return original_step(*args, **kwargs)

            trainer.optimizer.step = capture_step  # type: ignore[method-assign]
            result = trainer.train_epoch(
                [
                    batch([[[0.2]]], [[[True]]]),
                    batch([[[0.4]]], [[[True]]]),
                    batch([[[0.6]]], [[[True]]]),
                ]
            )

            self.assertEqual(len(observed), 2)
            self.assertAlmostEqual(observed[0][1], -0.6, places=5)
            # Q and Z both contribute; the one-batch tail must not be divided by 2.
            expected_tail_gradient = 2.0 * (observed[1][0] - 0.6)
            self.assertAlmostEqual(observed[1][1], expected_tail_gradient, places=5)
            self.assertEqual(result["q_valid_count"], 3)
            self.assertEqual(result["z_valid_count"], 3)

    def test_evaluation_is_invariant_to_batch_partition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = batch(
                [[[1.0, 2.0], [3.0, 4.0]]],
                [[[True, False], [False, False]]],
            )
            second = batch(
                [[[5.0, 6.0], [7.0, 8.0]]],
                [[[True, True], [True, True]]],
            )
            trainer = Trainer(
                TinyModel(), config(Path(directory)), torch.device("cpu")
            )
            split = trainer.evaluate([first, second])
            joined = trainer.evaluate([concatenate(first, second)])

            self.assertEqual(split.keys(), joined.keys())
            for key in split:
                if isinstance(split[key], float) and math.isnan(split[key]):
                    self.assertTrue(math.isnan(joined[key]))
                else:
                    self.assertAlmostEqual(float(split[key]), float(joined[key]), places=7)
            self.assertAlmostEqual(float(split["q_h1_mae"]), 4.0)
            self.assertAlmostEqual(float(split["q_h2_mae"]), 7.5)

    def test_nonfinite_and_empty_mask_semantics(self) -> None:
        prediction = torch.tensor([[[float("inf")]]], requires_grad=True)
        target = torch.tensor([[[float("nan")]]])
        off = torch.zeros_like(target, dtype=torch.bool)
        loss = masked_huber(prediction, target, off)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(prediction.grad.item(), 0.0)
        self.assertTrue(math.isnan(horizon_metrics(prediction.detach(), target, off)["h1_mae"]))

        on = torch.ones_like(target, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "target包含NaN/Inf"):
            masked_huber(torch.zeros_like(target), target, on)
        with self.assertRaisesRegex(FloatingPointError, "预测包含NaN/Inf"):
            masked_huber(
                torch.full_like(target, float("inf")), torch.zeros_like(target), on
            )

    def test_checkpoint_separates_weights_from_strict_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = Trainer(TinyModel(0.25), config(root), torch.device("cpu"))
            source.best = 0.75
            source.stale = 2
            torch.manual_seed(1234)
            checkpoint = root / "state.pt"
            source.save_checkpoint(checkpoint, 4, {"loss": 0.8})
            expected_random = torch.rand(4)

            restored = Trainer(TinyModel(9.0), config(root), torch.device("cpu"))
            restored.load_weights(checkpoint)
            self.assertAlmostEqual(restored.model.weight.item(), 0.25)
            self.assertEqual(restored.start_epoch, 0)
            self.assertEqual(restored.best, float("inf"))

            restored.model.weight.data.fill_(10.0)
            restored.resume_checkpoint(checkpoint)
            self.assertAlmostEqual(restored.model.weight.item(), 0.25)
            self.assertEqual(restored.start_epoch, 5)
            self.assertEqual(restored.last_epoch, 4)
            self.assertEqual(restored.best, 0.75)
            self.assertEqual(restored.stale, 2)
            self.assertTrue(torch.equal(torch.rand(4), expected_random))

            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            for key in (
                "optimizer",
                "scaler",
                "rng_state",
                "best",
                "stale",
                "last_epoch",
                "last_metrics",
                "train_loader_rng_state",
            ):
                self.assertIn(key, payload)
            self.assertEqual(
                Trainer.last_checkpoint_path(root / "best.pt").name, "best.last.pt"
            )

            legacy = root / "legacy.pt"
            torch.save(
                {
                    "model": source.model.state_dict(),
                    "optimizer": source.optimizer.state_dict(),
                    "epoch": 2,
                    "metrics": {"loss": 0.4},
                    "config": source.cfg,
                },
                legacy,
            )
            legacy_restored = Trainer(TinyModel(), config(root), torch.device("cpu"))
            with self.assertWarns(RuntimeWarning):
                legacy_restored.resume_checkpoint(legacy)
            self.assertEqual(legacy_restored.start_epoch, 3)
            self.assertEqual(legacy_restored.best, 0.4)
            self.assertEqual(legacy_restored.stale, 0)

    def test_checkpoint_restores_dataloader_shuffle_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = Trainer(TinyModel(), config(root), torch.device("cpu"))
            loader = DataLoader(
                torch.arange(12),
                batch_size=3,
                shuffle=True,
                generator=torch.Generator().manual_seed(77),
            )
            list(loader)  # Advance through one epoch.
            trainer._capture_loader_rng_state(loader)
            checkpoint = root / "loader.pt"
            trainer.save_checkpoint(checkpoint, 0, {"loss": 1.0})
            expected = torch.cat(list(loader))

            resumed = Trainer(TinyModel(), config(root), torch.device("cpu"))
            resumed.resume_checkpoint(checkpoint)
            new_loader = DataLoader(
                torch.arange(12),
                batch_size=3,
                shuffle=True,
                generator=torch.Generator().manual_seed(999),
            )
            resumed._restore_loader_rng_state(new_loader)
            actual = torch.cat(list(new_loader))
            self.assertTrue(torch.equal(actual, expected))

    def test_fit_sets_epoch_and_finds_batch_sampler_generator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer = Trainer(TinyModel(), config(root), torch.device("cpu"))
            loader = EpochAwareLoader(batch([[[0.4]]], [[[True]]]))
            self.assertIs(
                Trainer._loader_generator(loader), loader.batch_sampler.generator
            )

            history = trainer.fit(loader)

            self.assertEqual(len(history), 2)
            self.assertEqual(loader.batch_sampler.epochs, [0, 1])


if __name__ == "__main__":
    unittest.main()
