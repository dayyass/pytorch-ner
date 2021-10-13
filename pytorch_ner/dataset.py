from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from .prepare_data import process_labels, process_tokens


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
            self.token_seq = token_seq  # type: ignore
            self.label_seq = label_seq  # type: ignore

    def __len__(self):
        return len(self.token_seq)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.preprocess:
            tokens = self.token_seq[idx]
            labels = self.label_seq[idx]
        else:
            tokens = process_tokens(self.token_seq[idx], self.token2idx)  # type: ignore
            labels = process_labels(self.label_seq[idx], self.label2idx)  # type: ignore

        lengths = [len(tokens)]

        return np.array(tokens), np.array(labels), np.array(lengths)


class NERCollator:
    """
    Collator that handles variable-size sentences.
    """

    def __init__(
        self,
        token_padding_value: int,
        label_padding_value: int,
        percentile: Union[int, float] = 100,
    ):
        self.token_padding_value = token_padding_value
        self.label_padding_value = label_padding_value
        self.percentile = percentile

    def __call__(
        self,
        batch: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        tokens, labels, lengths = zip(*batch)

        tokens = [list(i) for i in tokens]
        labels = [list(i) for i in labels]

        max_len = int(np.percentile(lengths, self.percentile))

        lengths = torch.tensor(
            np.clip(lengths, a_min=0, a_max=max_len),
            dtype=torch.long,
        ).squeeze(-1)

        for i in range(len(batch)):
            tokens[i] = torch.tensor(tokens[i][:max_len], dtype=torch.long)
            labels[i] = torch.tensor(labels[i][:max_len], dtype=torch.long)

        sorted_idx = torch.argsort(lengths, descending=True)

        tokens = pad_sequence(
            tokens, padding_value=self.token_padding_value, batch_first=True
        )[sorted_idx]
        labels = pad_sequence(
            labels, padding_value=self.label_padding_value, batch_first=True
        )[sorted_idx]
        lengths = lengths[sorted_idx]

        return tokens, labels, lengths
