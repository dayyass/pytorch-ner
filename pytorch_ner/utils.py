import random
import numpy as np
import torch
from typing import List, Dict


def process_tokens(tokens: List[str], token2idx: Dict[str, int], unk: str = '<UNK>') -> List[int]:
    """
    Transform list of tokens into list of tokens' indices.
    """

    processed_tokens = [token2idx.get(token, token2idx[unk]) for token in tokens]
    return processed_tokens


def process_labels(labels: List[str], label2idx: Dict[str, int]) -> List[int]:
    """
    Transform list of labels into list of labels' indices.
    """

    processed_labels = [label2idx[label] for label in labels]
    return processed_labels


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def masking(lengths: torch.Tensor) -> torch.BoolTensor:
    """
    Convert lengths tensor to binary mask
    implement: https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    """

    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    return torch.arange(end=max_len).expand(size=(lengths_shape, max_len)) < lengths.unsqueeze(1)


# TODO: add bio/biluo converters
