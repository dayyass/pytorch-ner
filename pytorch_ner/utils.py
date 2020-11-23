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


# TODO: add bio/biluo converters
