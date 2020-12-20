import os
import random
import shutil

import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to np.ndarray.
    """

    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def mkdir(path: str):
    """
    Make directory if not exists.
    """

    if not os.path.exists(path):
        os.makedirs(path)


def rmdir(path: str):
    """
    Remove directory if exists.
    """

    if os.path.exists(path):
        shutil.rmtree(path)
