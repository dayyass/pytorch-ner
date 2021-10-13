import importlib
import logging
import os
import random
import shutil
import sys
from argparse import ArgumentParser
from typing import Optional

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


def str_to_class(module_name, class_name):
    """
    Convert string to Python class object.
    https://stackoverflow.com/questions/1176136/convert-string-to-python-class-object
    """

    # load the module, will raise ImportError if module cannot be loaded
    module = importlib.import_module(module_name)
    # get the class, will raise AttributeError if class cannot be found
    cls = getattr(module, class_name)
    return cls


def get_argparse() -> ArgumentParser:
    """Get argument parser.

    Returns:
        ArgumentParser: Argument parser.
    """

    parser = ArgumentParser()
    parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        help="Path to config",
    )

    return parser


def get_logger(path_to_logfile: Optional[str] = None) -> logging.Logger:
    """Get logger.

    Args:
        path_to_logfile (Optional[str], optional): Path to logfile. Defaults to None.

    Returns:
        logging.Logger: Logger.
    """

    logger = logging.getLogger("pytorch-ner-train")
    logger.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_format)
    logger.addHandler(stream_handler)

    if path_to_logfile:
        file_handler = logging.FileHandler(path_to_logfile)
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close logger.
    Source: https://stackoverflow.com/questions/15435652/python-does-not-release-filehandles-to-logfile

    Args:
        logger (logging.Logger): Logger.
    """

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
