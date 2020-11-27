import unittest
import numpy as np
import torch
from pytorch_ner.nn_modules.embedding import load_glove  # EmbeddingWord2Vec, EmbeddingGloVe


class TestEmbedding(unittest.TestCase):

    def test_load_glove(self):
        token2idx, embedding_matrix = load_glove(path='tests/data/glove.txt')

        self.assertEqual(len(token2idx), 10)
        self.assertEqual(len(token2idx), embedding_matrix.shape[0])
        self.assertEqual(embedding_matrix.shape[-1], 100)
        self.assertEqual(token2idx['<PAD>'], 0)
        self.assertEqual(token2idx['<UNK>'], 1)
        self.assertTrue(
            np.allclose(embedding_matrix[0], np.zeros_like(embedding_matrix[0])),
        )
        self.assertTrue(
            np.allclose(embedding_matrix[1], embedding_matrix[2:].mean(axis=0)),
        )

    # def test_embedding_shape(self):
    #     self.assertTrue(embedding_w2v_freeze.embedding.weight.shape == torch.Size([10, 100]))
    #     self.assertTrue(embedding_w2v_fine_tune.embedding.weight.shape == torch.Size([10, 100]))
    #
    # def test_embedding_requires_grad(self):
    #     self.assertFalse(embedding_w2v_freeze.embedding.weight.requires_grad)
    #     self.assertTrue(embedding_w2v_fine_tune.embedding.weight.requires_grad)
    #
    # def test_embedding_pad(self):
    #     pad_embedding = embedding_w2v_freeze(torch.tensor([0]))
    #     self.assertTrue(
    #         torch.equal(pad_embedding, torch.zeros_like(pad_embedding))
    #     )
    #
    #     pad_embedding = embedding_w2v_fine_tune(torch.tensor([0]))
    #     self.assertTrue(
    #         torch.equal(pad_embedding, torch.zeros_like(pad_embedding))
    #     )
    #
    # def test_embedding_unk(self):
    #     unk_embedding = embedding_w2v_freeze(torch.tensor([1]))
    #     self.assertTrue(
    #         torch.allclose(unk_embedding, embedding_w2v_freeze.embedding.weight[2:].mean(dim=0))
    #     )
    #
    #     unk_embedding = embedding_w2v_fine_tune(torch.tensor([1]))
    #     self.assertTrue(
    #         torch.allclose(unk_embedding, embedding_w2v_fine_tune.embedding.weight[2:].mean(dim=0))
    #     )


if __name__ == '__main__':
    unittest.main()
