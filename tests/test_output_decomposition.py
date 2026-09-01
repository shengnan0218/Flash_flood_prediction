from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from metrics.output_decomposition import evaluate_output_decomposition


class _Batch:
    def __init__(self) -> None:
        self.q_history = torch.tensor(
            [
                [[1.0], [2.0]],
                [[10.0], [20.0]],
            ]
        )
        self.q_mask = torch.ones((2, 2, 1), dtype=torch.bool)
        self.q_target = torch.tensor(
            [
                [[3.0], [4.0], [5.0]],
                [[21.0], [22.0], [23.0]],
            ]
        )
        self.q_target_mask = torch.ones((2, 3, 1), dtype=torch.bool)
        self.obs_station_ids = ("S1",)

    def to(self, device: torch.device):
        for name in ("q_history", "q_mask", "q_target", "q_target_mask"):
            setattr(self, name, getattr(self, name).to(device))
        return self


class _Model(nn.Module):
    def forward(self, batch: _Batch):
        base = batch.q_target.clone()
        return {
            "q": base + 1.0,
            "diagnostics": {
                "q_residual_base_m3s": base,
            },
        }


def test_decomposition_separates_routed_base_and_final(tmp_path: Path) -> None:
    trainer = SimpleNamespace(model=_Model(), device=torch.device("cpu"))
    result = evaluate_output_decomposition(
        trainer,
        [_Batch()],
        tmp_path,
        split="VALIDATION",
        checkpoint="outputs/example.pt",
    )
    methods = result["methods"]
    assert methods["persistence"]["Q0_VALID_COUNT"] == 6
    assert methods["routed_base"]["Q_NSE"] == 1.0
    assert methods["routed_base"]["SKILL_OVER_PERSISTENCE"] == 1.0
    assert methods["final"]["Q_NSE"] < methods["routed_base"]["Q_NSE"]
    assert Path(result["files"]["summary"]).exists()
    assert Path(result["files"]["by_lead"]).exists()
    assert Path(result["files"]["by_station"]).exists()
