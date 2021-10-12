import unittest

import numpy as np
import torch

from pytorch_ner.nn_modules.dropout import SpatialDropout1d
from pytorch_ner.nn_modules.embedding import (
    Embedding,
    EmbeddingPreTrained,
    EmbeddingWithDropout,
    load_glove,
    load_word2vec,
)
from pytorch_ner.prepare_data import prepare_conll_data_format

token_seq, _ = prepare_conll_data_format(
    path="tests/data/conll.txt", sep=" ", verbose=False
)
tokens = list(set(token for sentence in token_seq for token in sentence))

_, word2vec_embeddings = load_word2vec(path="tests/data/word2vec.wv")
_, glove_embeddings = load_glove(path="tests/data/glove.txt")


embedding_w2v_freeze = EmbeddingPreTrained(word2vec_embeddings)
embedding_w2v_fine_tune = EmbeddingPreTrained(word2vec_embeddings, freeze=False)

embedding_glove_freeze = EmbeddingPreTrained(glove_embeddings)
embedding_glove_fine_tune = EmbeddingPreTrained(glove_embeddings, freeze=False)


random_embedding_with_spatial_dropout = EmbeddingWithDropout(
    embedding_layer=Embedding(num_embeddings=2000, embedding_dim=128),
    dropout=SpatialDropout1d(p=0.5),
)

emb = random_embedding_with_spatial_dropout(
    torch.randint(low=0, high=2000, size=(10, 20)),
)


class TestLoadEmbedding(unittest.TestCase):
    def test_load_glove(self):
        token2idx, embedding_matrix = load_glove(path="tests/data/glove.txt")

        self.assertEqual(len(token2idx), 10)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertEqual(token2idx["<PAD>"], 0)
        self.assertEqual(token2idx["<UNK>"], 1)
        self.assertTrue(
            np.allclose(embedding_matrix[0], np.zeros_like(embedding_matrix[0])),
        )
        self.assertTrue(
            np.allclose(embedding_matrix[1], embedding_matrix[2:].mean(axis=0)),
        )

    def test_load_glove_without_pad(self):
        token2idx, embedding_matrix = load_glove(
            path="tests/data/glove.txt",
            add_pad=False,
        )

        self.assertEqual(len(token2idx), 9)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertTrue("<PAD>" not in token2idx)
        self.assertEqual(token2idx["<UNK>"], 0)
        self.assertTrue(
            np.allclose(embedding_matrix[0], embedding_matrix[1:].mean(axis=0)),
        )

    def test_load_glove_without_unk(self):
        token2idx, embedding_matrix = load_glove(
            path="tests/data/glove.txt",
            add_unk=False,
        )

        self.assertEqual(len(token2idx), 9)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertEqual(token2idx["<PAD>"], 0)
        self.assertTrue("<UNK>" not in token2idx)
        self.assertTrue(
            np.allclose(embedding_matrix[0], np.zeros_like(embedding_matrix[0])),
        )

    def test_load_glove_without_pad_unk(self):
        token2idx, embedding_matrix = load_glove(
            path="tests/data/glove.txt",
            add_pad=False,
            add_unk=False,
        )

        self.assertEqual(len(token2idx), 8)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertTrue("<PAD>" not in token2idx)
        self.assertTrue("<UNK>" not in token2idx)

    def test_load_word2vec(self):
        token2idx, embedding_matrix = load_word2vec(path="tests/data/word2vec.wv")

        self.assertEqual(len(token2idx), 10)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertEqual(token2idx["<PAD>"], 0)
        self.assertEqual(token2idx["<UNK>"], 1)
        self.assertTrue(
            np.allclose(embedding_matrix[0], np.zeros_like(embedding_matrix[0])),
        )
        self.assertTrue(
            np.allclose(embedding_matrix[1], embedding_matrix[2:].mean(axis=0)),
        )

    def test_load_word2vec_without_pad(self):
        token2idx, embedding_matrix = load_word2vec(
            path="tests/data/word2vec.wv",
            add_pad=False,
        )

        self.assertEqual(len(token2idx), 9)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertTrue("<PAD>" not in token2idx)
        self.assertEqual(token2idx["<UNK>"], 0)
        self.assertTrue(
            np.allclose(embedding_matrix[0], embedding_matrix[1:].mean(axis=0)),
        )

    def test_load_word2vec_without_unk(self):
        token2idx, embedding_matrix = load_word2vec(
            path="tests/data/word2vec.wv",
            add_unk=False,
        )

        self.assertEqual(len(token2idx), 9)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertEqual(token2idx["<PAD>"], 0)
        self.assertTrue("<UNK>" not in token2idx)
        self.assertTrue(
            np.allclose(embedding_matrix[0], np.zeros_like(embedding_matrix[0])),
        )

    def test_load_word2vec_without_pad_unk(self):
        token2idx, embedding_matrix = load_word2vec(
            path="tests/data/word2vec.wv",
            add_pad=False,
            add_unk=False,
        )

        self.assertEqual(len(token2idx), 8)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertTrue("<PAD>" not in token2idx)
        self.assertTrue("<UNK>" not in token2idx)

    def test_compare_word2vec_glove(self):
        token2idx_word2vec, embedding_matrix_word2vec = load_word2vec(
            path="tests/data/word2vec.wv"
        )
        token2idx_glove, embedding_matrix_glove = load_glove(
            path="tests/data/glove.txt"
        )

        self.assertDictEqual(token2idx_word2vec, token2idx_glove)
        self.assertTrue(embedding_matrix_word2vec.shape == embedding_matrix_glove.shape)


class TestEmbeddingPreTrained(unittest.TestCase):
    def test_embedding_shape(self):
        # word2vec
        self.assertTrue(
            embedding_w2v_freeze.embedding.weight.shape == torch.Size([10, 100])
        )
        self.assertTrue(
            embedding_w2v_fine_tune.embedding.weight.shape == torch.Size([10, 100])
        )

        # glove
        self.assertTrue(
            embedding_glove_freeze.embedding.weight.shape == torch.Size([10, 100])
        )
        self.assertTrue(
            embedding_glove_fine_tune.embedding.weight.shape == torch.Size([10, 100])
        )

    def test_embedding_requires_grad(self):
        # word2vec
        self.assertFalse(embedding_w2v_freeze.embedding.weight.requires_grad)
        self.assertTrue(embedding_w2v_fine_tune.embedding.weight.requires_grad)

        # glove
        self.assertFalse(embedding_glove_freeze.embedding.weight.requires_grad)
        self.assertTrue(embedding_glove_fine_tune.embedding.weight.requires_grad)

    def test_embedding_pad(self):
        # word2vec
        pad_embedding = embedding_w2v_freeze(torch.tensor([0], dtype=torch.long))
        self.assertTrue(torch.equal(pad_embedding, torch.zeros_like(pad_embedding)))

        pad_embedding = embedding_w2v_fine_tune(torch.tensor([0], dtype=torch.long))
        self.assertTrue(torch.equal(pad_embedding, torch.zeros_like(pad_embedding)))

        # glove
        pad_embedding = embedding_glove_freeze(torch.tensor([0], dtype=torch.long))
        self.assertTrue(torch.equal(pad_embedding, torch.zeros_like(pad_embedding)))

        pad_embedding = embedding_glove_fine_tune(torch.tensor([0], dtype=torch.long))
        self.assertTrue(torch.equal(pad_embedding, torch.zeros_like(pad_embedding)))

    def test_embedding_unk(self):
        # word2vec
        unk_embedding = embedding_w2v_freeze(torch.tensor([1], dtype=torch.long))
        self.assertTrue(
            torch.allclose(
                unk_embedding, embedding_w2v_freeze.embedding.weight[2:].mean(dim=0)
            )
        )

        unk_embedding = embedding_w2v_fine_tune(torch.tensor([1], dtype=torch.long))
        self.assertTrue(
            torch.allclose(
                unk_embedding, embedding_w2v_fine_tune.embedding.weight[2:].mean(dim=0)
            )
        )

        # glove
        unk_embedding = embedding_glove_freeze(torch.tensor([1], dtype=torch.long))
        self.assertTrue(
            torch.allclose(
                unk_embedding, embedding_glove_freeze.embedding.weight[2:].mean(dim=0)
            )
        )

        unk_embedding = embedding_glove_fine_tune(torch.tensor([1], dtype=torch.long))
        self.assertTrue(
            torch.allclose(
                unk_embedding,
                embedding_glove_fine_tune.embedding.weight[2:].mean(dim=0),
            )
        )


class TestEmbeddingWithDropout(unittest.TestCase):
    def test_embedding_shape(self):
        self.assertTrue(emb.shape == torch.Size([10, 20, 128]))

    def test_spatial_dropout(self):
        if 0 in emb:
            emb_dim_sum = emb.sum(dim=0).sum(dim=0)
            self.assertTrue(0 in emb_dim_sum)


if __name__ == "__main__":
    unittest.main()
