import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict
from gensim.models import KeyedVectors


def load_glove(
        path: str,
        add_pad: bool = True,
        add_unk: bool = True,
) -> Tuple[Dict[str, int], np.ndarray]:

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


# class EmbeddingWord2Vec(nn.Module):
#     """
#     Init embeddings from gensim word2vec KeyedVectors.
#     """
#
#     def __init__(self, path: str, freeze: bool = True):
#         super(EmbeddingWord2Vec, self).__init__()
#         model = KeyedVectors.load_word2vec_format(path)
#
#         word_embeddings = model.vectors
#         pad_embedding = np.zeros(shape=model.vector_size)
#         unk_embedding = word_embeddings.mean(axis=0)  # TODO: make better unk embedding initialization
#         embedding_matrix = np.vstack([pad_embedding, unk_embedding, word_embeddings])
#
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.tensor(embedding_matrix),
#             freeze=freeze,
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.embedding(x)
#
#
# class EmbeddingGloVe(nn.Module):
#     """
#     Init embeddings from glove.
#     """
#
#     def __init__(self, path: str, freeze: bool = True):
#         super(EmbeddingGloVe, self).__init__()
#         model = KeyedVectors.load_word2vec_format(path)
#
#         word_embeddings = model.vectors
#         pad_embedding = np.zeros(shape=model.vector_size)
#         unk_embedding = word_embeddings.mean(axis=0)  # TODO: make better unk embedding initialization
#         embedding_matrix = np.vstack([pad_embedding, unk_embedding, word_embeddings])
#
#         self.embedding = nn.Embedding.from_pretrained(
#             torch.tensor(embedding_matrix),
#             freeze=freeze,
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.embedding(x)

# TODO: test add_pad, add_unk
