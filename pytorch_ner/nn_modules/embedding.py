import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List, Dict
from gensim.models import KeyedVectors, FastText


def load_glove(
        path: str,
        add_pad: bool = True,
        add_unk: bool = True,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load glove embeddings.
    """

    token2idx = {}
    token_embeddings = []

    if add_pad:
        token2idx['<PAD>'] = len(token2idx)
    if add_unk:
        token2idx['<UNK>'] = len(token2idx)

    with open(path, mode='r') as fp:
        for line in fp:
            token, embedding = line.split(maxsplit=1)
            embedding = np.array([float(i) for i in embedding.split()])

            token2idx[token] = len(token2idx)
            token_embeddings.append(embedding)

    token_embeddings = np.array(token_embeddings)

    if add_unk:
        unk_embedding = token_embeddings.mean(axis=0)  # TODO: make better unk embedding initialization
        token_embeddings = np.vstack([unk_embedding, token_embeddings])
    if add_pad:
        pad_embedding = np.zeros(shape=token_embeddings.shape[-1])
        token_embeddings = np.vstack([pad_embedding, token_embeddings])

    return token2idx, token_embeddings


def load_word2vec(
        path: str,
        add_pad: bool = True,
        add_unk: bool = True,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load word2vec embeddings.
    """

    token2idx = {}

    if add_pad:
        token2idx['<PAD>'] = len(token2idx)
    if add_unk:
        token2idx['<UNK>'] = len(token2idx)

    model = KeyedVectors.load_word2vec_format(path)
    for token in model.index2word:
        token2idx[token] = len(token2idx)

    token_embeddings = model.vectors

    if add_unk:
        unk_embedding = token_embeddings.mean(axis=0)  # TODO: make better unk embedding initialization
        token_embeddings = np.vstack([unk_embedding, token_embeddings])
    if add_pad:
        pad_embedding = np.zeros(shape=token_embeddings.shape[-1])
        token_embeddings = np.vstack([pad_embedding, token_embeddings])

    return token2idx, token_embeddings


def fasttext2word2vec(
        path: str,
        tokens: List[str],
        add_pad: bool = True,
) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Represent train/val/test tokens as fasttext word embeddings (like word2vec) to use with nn.Embeddings.
    There is no <UNK> token, since all train/val/test tokens have embedding representation.
    Not suitable for inference (use gensim fasttext model).
    """

    token2idx = {}
    token_embeddings = []

    if add_pad:
        token2idx['<PAD>'] = len(token2idx)

    model = FastText.load(path)
    for token in tokens:
        embedding = model.wv[token]

        token2idx[token] = len(token2idx)
        token_embeddings.append(embedding)

    token_embeddings = np.array(token_embeddings)

    if add_pad:
        pad_embedding = np.zeros(shape=token_embeddings.shape[-1])
        token_embeddings = np.vstack([pad_embedding, token_embeddings])

    return token2idx, token_embeddings


class EmbeddingPreTrained(nn.Module):
    """
    Init embeddings layer from word2vec/glove.
    """

    def __init__(self, embedding_matrix: np.ndarray, freeze: bool = True):
        super(EmbeddingPreTrained, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=freeze,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
