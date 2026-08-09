from __future__ import annotations
import platform, random
import torch

def resolve_device(name: str, gpu_id: int = 0) -> torch.device:
    if name == "auto": name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available(): raise RuntimeError("配置要求CUDA，但当前PyTorch未检测到可用GPU")
    return torch.device(f"cuda:{gpu_id}" if name == "cuda" else "cpu")

def seed_everything(seed: int) -> None:
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def device_report(device: torch.device) -> dict[str, str]:
    return {"device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor() or "CPU", "torch": torch.__version__, "cuda": str(torch.version.cuda)}
