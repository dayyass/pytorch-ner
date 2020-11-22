from typing import Tuple, List
from tqdm import tqdm


# TODO: check conll
def prepare_conll_data_format(path: str, sep: str = '\t') -> Tuple[List[List[str]], List[List[str]]]:

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
                tokens.append(token)
                labels.append(label)
            else:
                token_seq.append(tokens)
                label_seq.append(labels)
                tokens = []
                labels = []

    return token_seq, label_seq
