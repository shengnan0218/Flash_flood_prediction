from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from metrics.output_decomposition import evaluate_output_decomposition
from trainers.trainer import _is_cuda_out_of_memory


class _Batch:
    def __init__(self) -> None:
        self.q_history = torch.tensor([[[1.0], [2.0]], [[10.0], [20.0]]])
        self.q_mask = torch.ones((2, 2, 1), dtype=torch.bool)
        self.q_target = torch.tensor([[[3.0], [4.0], [5.0]], [[21.0], [22.0], [23.0]]])
        self.q_target_mask = torch.ones((2, 3, 1), dtype=torch.bool)
        self.obs_station_ids = ("S1",)

    def to(self, device: torch.device):
        for name in ("q_history", "q_mask", "q_target", "q_target_mask"):
            setattr(self, name, getattr(self, name).to(device))
        return self


class _Model(nn.Module):
    def forward(self, batch: _Batch):
        q0 = batch.q_history[:, -1]
        route_delta = batch.q_target - q0.unsqueeze(1)
        return {
            "q": batch.q_target + 1.0,
            "q0_analysis": q0,
            "diagnostics": {
                "q_route_delta_m3s": route_delta,
                "q_origin_observed_available": batch.q_mask[:, -1],
            },
        }


def test_decomposition_separates_full_route_and_gated_route(tmp_path: Path) -> None:
    trainer = SimpleNamespace(model=_Model(), device=torch.device("cpu"))
    result = evaluate_output_decomposition(
        trainer, [_Batch()], tmp_path, split="VALIDATION", checkpoint="outputs/example.pt"
    )
    methods = result["methods"]
    assert methods["persistence"]["Q0_VALID_COUNT"] == 6
    assert methods["full_route"]["Q_NSE"] == 1.0
    assert methods["full_route"]["SKILL_OVER_PERSISTENCE"] == 1.0
    assert methods["gated_route"]["Q_NSE"] < methods["full_route"]["Q_NSE"]
    assert Path(result["files"]["summary"]).exists()
    assert Path(result["files"]["by_lead"]).exists()
    assert Path(result["files"]["by_station"]).exists()


def test_oom_detection_does_not_depend_on_torch_root_alias() -> None:
    assert _is_cuda_out_of_memory(RuntimeError("CUDA out of memory."))
    assert not _is_cuda_out_of_memory(RuntimeError("unrelated routing failure"))
