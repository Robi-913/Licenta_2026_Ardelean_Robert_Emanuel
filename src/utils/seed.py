import random
import numpy as np
import torch

# ═════════════════════════════════════════════════════════════════════════════
# GLOBAL SEED — schimbă DOAR aici, se propagă peste tot
# ═════════════════════════════════════════════════════════════════════════════
SEED = 42


def set_seed(seed=None):
    """
    :param seed: valoarea pentru seed; daca e none, folosim constanta globala
    :return: nimic (setam doar seed-urile intern)

    cuda foloseste gpu-ul pentru calcul paralel masiv; setam modul determinist pentru a evita
    variatiile minuscule intre rulari si a avea rezultate replicabile

    """
    s = seed if seed is not None else SEED

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed set to {s} (deterministic mode)")
    return s