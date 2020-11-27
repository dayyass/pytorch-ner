import unittest
import torch
from pytorch_ner.nn_modules.embedding import EmbeddingWord2Vec


embeddings = torch.randn(10, 20, 128)  # [batch_size, seq_len, emb_dim]

embedding_w2v_freeze = EmbeddingWord2Vec(path='tests/data/word2vec.wv')
embedding_w2v_fine_tune = EmbeddingWord2Vec(path='tests/data/word2vec.wv', freeze=False)


class TestEmbedding(unittest.TestCase):

    def test_embedding_shape(self):
        self.assertTrue(embedding_w2v_freeze.embedding.weight.shape == torch.Size([10, 100]))
        self.assertTrue(embedding_w2v_fine_tune.embedding.weight.shape == torch.Size([10, 100]))

    def test_embedding_requires_grad(self):
        self.assertFalse(embedding_w2v_freeze.embedding.weight.requires_grad)
        self.assertTrue(embedding_w2v_fine_tune.embedding.weight.requires_grad)

    def test_embedding_pad(self):
        pad_embedding = embedding_w2v_freeze(torch.tensor([0]))
        self.assertTrue(
            torch.equal(pad_embedding, torch.zeros_like(pad_embedding))
        )

        pad_embedding = embedding_w2v_fine_tune(torch.tensor([0]))
        self.assertTrue(
            torch.equal(pad_embedding, torch.zeros_like(pad_embedding))
        )

    def test_embedding_unk(self):
        unk_embedding = embedding_w2v_freeze(torch.tensor([1]))
        self.assertTrue(
            torch.allclose(unk_embedding, embedding_w2v_freeze.embedding.weight[2:].mean(dim=0))
        )

        unk_embedding = embedding_w2v_fine_tune(torch.tensor([1]))
        self.assertTrue(
            torch.allclose(unk_embedding, embedding_w2v_fine_tune.embedding.weight[2:].mean(dim=0))
        )


if __name__ == '__main__':
    unittest.main()
