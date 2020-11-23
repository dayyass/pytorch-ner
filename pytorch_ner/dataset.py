import numpy as np
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset

from .utils import process_tokens, process_labels


class NERDataset(Dataset):

    """
    PyTorch Dataset for NER data format.
    Dataset might be preprocessed for more efficiency.
    """

    def __init__(
            self,
            token_seq: List[List[str]],
            label_seq: List[List[str]],
            token2idx: Dict[str, int],
            label2idx: Dict[str, int],
            preprocess: bool = True,
    ):
        self.token2idx = token2idx
        self.label2idx = label2idx
        self.preprocess = preprocess

        if preprocess:
            self.token_seq = [process_tokens(tokens, token2idx) for tokens in token_seq]
            self.label_seq = [process_labels(labels, label2idx) for labels in label_seq]
        else:
            self.token_seq = token_seq
            self.label_seq = label_seq

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocess:
            tokens = self.token_seq[idx]
            labels = self.label_seq[idx]
        else:
            tokens = process_tokens(self.token_seq[idx], self.token2idx)
            labels = process_labels(self.label_seq[idx], self.label2idx)

        lengths = len(tokens)

        return np.array(tokens), np.array(lengths), np.array(labels)
