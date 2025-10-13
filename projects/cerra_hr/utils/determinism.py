# aurora-europe/projects/cerra_hr/utils/determinism.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42, *, deterministic: bool = True) -> None:
    """
    Best-effort reproducibility across Python, NumPy, and PyTorch.

    Notes:
      - Full determinism can be slower and may disable some fast kernels.
      - Aurora fine-tuning uses AMP/ bf16; numerics may still vary slightly.
    """
    # Python
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (CPU + CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuBLAS determinism (needed for matmul ops in newer PyTorch)
        # Use one of ":4096:8" or ":16:8" per PyTorch docs.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # cuDNN determinism and no autotuner
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithm selection (may raise on non-deterministic ops)
        torch.use_deterministic_algorithms(True)
    else:
        # Allow fastest kernels (non-deterministic possible)
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
