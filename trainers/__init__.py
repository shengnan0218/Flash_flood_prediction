import torch

# PyTorch 2.2 exposes CUDA OOM under torch.cuda.OutOfMemoryError rather than
# torch.OutOfMemoryError.  Keep trainer.py's fail-fast handler compatible
# without masking the original exception on supported 2.2 environments.
if not hasattr(torch, "OutOfMemoryError"):
    torch.OutOfMemoryError = torch.cuda.OutOfMemoryError

from .trainer import Trainer

__all__ = ["Trainer"]
