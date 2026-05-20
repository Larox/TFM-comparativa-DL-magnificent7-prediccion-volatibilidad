"""Global seed setter used by every script at startup.

Torch and lightning are imported lazily so that the function works in
environments where they are not installed (e.g. pure-data tests).
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> int:
    """Seed Python's `random`, NumPy, and (if importable) PyTorch + Lightning.

    Returns the seed used so callers can log it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass

    try:
        import pytorch_lightning as pl

        pl.seed_everything(seed, workers=True)
    except ImportError:
        pass

    return seed
