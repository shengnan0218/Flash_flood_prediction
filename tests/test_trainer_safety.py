from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from trainers import Trainer


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


def config(directory: Path) -> dict:
    return {
        "optimizer": {"lr": 0.01, "weight_decay": 0.0},
        "amp": False,
        "batch_size": 1,
        "loss_weights": {"discharge": 1.0, "water_level": 1.0},
        "gradient_accumulation_steps": 1,
        "training": {
            "epochs": 1,
            "patience": 1,
            "gradient_clip": 1.0,
            "checkpoint": str(directory / "best.pt"),
            "log_csv": str(directory / "train.csv"),
        },
    }


class TestTrainerSafety(unittest.TestCase):
    def test_patience_100_runs_all_epochs_zero_through_99(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = config(root)
            cfg["training"]["epochs"] = 100
            cfg["training"]["patience"] = 100
            cfg["_runtime"] = {
                "q_scale_audit": {
                    "computed_from_split": "TRAIN",
                    "graphs": {
                        "G1": {
                            "valid_unique_point_count": 2,
                            "mean_m3s": 1.0,
                            "std_m3s": 0.5,
                            "q_loss_scale_m3s": 1.0,
                            "floor_applied": True,
                        }
                    },
                }
            }
            trainer = Trainer(TinyModel(), cfg, torch.device("cpu"))
            epochs: list[int] = []

            def constant_train_epoch(_loader, epoch: int):
                epochs.append(epoch)
                return {"loss": 1.0}

            with (
                mock.patch.object(
                    trainer, "train_epoch", side_effect=constant_train_epoch
                ),
                mock.patch.object(trainer, "save_checkpoint"),
                mock.patch("trainers.trainer._append_csv_row"),
            ):
                history = trainer.fit([object()], val_loader=None)

            self.assertEqual(epochs, list(range(100)))
            self.assertEqual(len(history), 100)
            audit_path = root / "best_q_scales.json"
            self.assertTrue(audit_path.is_file())

    def test_gradient_clipping_defers_nonfinite_amp_gradients_to_scaler(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(
                TinyModel(), config(Path(directory)), torch.device("cpu")
            )
            trainer.model.weight.grad = torch.tensor(float("inf"))

            trainer.amp = False
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                trainer._clip_gradients()

            trainer.model.weight.grad = torch.tensor(float("inf"))
            trainer.amp = True
            total_norm = trainer._clip_gradients()
            self.assertTrue(torch.isinf(total_norm))

    def test_atomic_checkpoint_replaces_only_after_successful_save(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state.pt"
            checkpoint.write_bytes(b"previous-checkpoint")
            trainer = Trainer(TinyModel(), config(root), torch.device("cpu"))
            original_replace = os.replace
            observed_temporary_paths: list[Path] = []

            def recording_replace(source, destination) -> None:
                temporary_path = Path(source)
                observed_temporary_paths.append(temporary_path)
                self.assertTrue(temporary_path.is_absolute())
                self.assertEqual(temporary_path.parent, checkpoint.parent)
                self.assertNotEqual(temporary_path, checkpoint)
                self.assertEqual(Path(destination), checkpoint)
                self.assertEqual(checkpoint.read_bytes(), b"previous-checkpoint")
                temporary_payload = torch.load(
                    temporary_path, map_location="cpu", weights_only=False
                )
                self.assertEqual(temporary_payload["epoch"], 3)
                original_replace(source, destination)

            with mock.patch(
                "trainers.trainer.os.replace", side_effect=recording_replace
            ):
                trainer.save_checkpoint(checkpoint, 3, {"loss": 0.5})

            self.assertEqual(len(observed_temporary_paths), 1)
            self.assertFalse(observed_temporary_paths[0].exists())
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.assertEqual(payload["epoch"], 3)
            self.assertEqual(payload["checkpoint_kind"], "manual")

    def test_failed_checkpoint_save_preserves_target_and_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state.pt"
            checkpoint.write_bytes(b"known-good")
            trainer = Trainer(TinyModel(), config(root), torch.device("cpu"))
            observed_temporary_paths: list[Path] = []

            def failing_save(_payload, destination) -> None:
                observed_temporary_paths.extend(root.glob(".state.pt.*.tmp"))
                destination.write(b"partial")
                raise OSError("simulated disk failure")

            with (
                mock.patch("trainers.trainer.torch.save", side_effect=failing_save),
                mock.patch("trainers.trainer.os.replace") as replace,
            ):
                with self.assertRaisesRegex(OSError, "simulated disk failure"):
                    trainer.save_checkpoint(checkpoint, 4, {"loss": 0.4})
                replace.assert_not_called()

            self.assertEqual(checkpoint.read_bytes(), b"known-good")
            self.assertEqual(len(observed_temporary_paths), 1)
            self.assertFalse(observed_temporary_paths[0].exists())
            self.assertEqual(list(root.glob(".state.pt.*.tmp")), [])

    def test_failed_atomic_replace_preserves_target_and_cleans_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "state.pt"
            checkpoint.write_bytes(b"known-good")
            trainer = Trainer(TinyModel(), config(root), torch.device("cpu"))
            observed_temporary_paths: list[Path] = []

            def failing_replace(source, destination) -> None:
                observed_temporary_paths.append(Path(source))
                self.assertEqual(Path(destination), checkpoint)
                raise PermissionError("simulated replace failure")

            with mock.patch("trainers.trainer.os.replace", side_effect=failing_replace):
                with self.assertRaisesRegex(PermissionError, "replace failure"):
                    trainer.save_checkpoint(checkpoint, 5, {"loss": 0.3})

            self.assertEqual(checkpoint.read_bytes(), b"known-good")
            self.assertEqual(len(observed_temporary_paths), 1)
            self.assertFalse(observed_temporary_paths[0].exists())
            self.assertEqual(list(root.glob(".state.pt.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
