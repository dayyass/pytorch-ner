from typing import Tuple, List, Dict
from tqdm import tqdm


# TODO: check conll
def prepare_conll_data_format(
    path: str,
    sep: str = '\t',
    lower: bool = True,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Prepare data in CoNNL like format.
    Sentences are separated by empty line.
    Tokens and labels are tab-separated.
    """

    token_seq = []
    label_seq = []
    with open(path, mode='r') as fp:
        tokens = []
        labels = []
        for line in tqdm(fp):
            if line != '\n':
                token, label = line.strip().split(sep)
                if lower:
                    token = token.lower()
                tokens.append(token)
                labels.append(label)
            else:
                if len(tokens) > 0:
                    token_seq.append(tokens)
                    label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq


def get_token2idx(
    token2cnt: Dict[str, int],
    min_count: int = 1,
    add_pad: bool = True,
    add_unk: bool = True,
) -> Dict[str, int]:
    """
    Get mapping from tokens to indices to use with Embedding layer.
    """

    token2idx = {}

    if add_pad:
        token2idx['<PAD>'] = len(token2idx)
    if add_unk:
        token2idx['<UNK>'] = len(token2idx)

    for token, cnt in token2cnt.items():
        if cnt >= min_count:
            token2idx[token] = len(token2idx)

    return token2idx


def get_label2idx(label_set: List[str]) -> Dict[str, int]:
    """
    Get mapping from labels to indices.
    """

    label2idx = {}

    for label in label_set:
        label2idx[label] = len(label2idx)

    return label2idx


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
