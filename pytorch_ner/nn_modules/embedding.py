import numpy as np
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


class EmbeddingWord2Vec(nn.Module):
    """
    Init embeddings from gensim word2vec KeyedVectors.
    """

    def __init__(self, path: str, freeze: bool = True):
        super(EmbeddingWord2Vec, self).__init__()
        model = KeyedVectors.load_word2vec_format(path)

        word_embeddings = model.vectors
        pad_embedding = np.zeros(shape=model.vector_size)
        unk_embedding = word_embeddings.mean(axis=0)  # TODO: make better unk embedding initialization
        embedding_matrix = np.vstack([pad_embedding, unk_embedding, word_embeddings])

        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix),
            freeze=freeze,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
