"""
Funcție pentru reproducibility — setează seed-uri pentru toate librăriile.
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Setează seed pentru reproducibilitate.

    Args:
        seed: Seed value (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"✓ Seed setat: {seed}")