import unittest

import torch

from pytorch_ner.nn_modules.dropout import SpatialDropout1d, WordEmbeddingsDropout

embeddings = torch.randn(2, 50, 128)  # [batch_size, seq_len, emb_dim]

spatial_dropout = SpatialDropout1d(p=0.5)
word_embeddings_dropout = WordEmbeddingsDropout(p=0.5)


class TestDropout(unittest.TestCase):
    def test_spatial_dropout(self):

        emb = spatial_dropout(embeddings)
        if 0 in emb:
            emb_dim_sum = emb.sum(dim=0).sum(dim=0)
            self.assertTrue(0 in emb_dim_sum)

    def test_word_embeddings_dropout(self):

        emb = word_embeddings_dropout(embeddings)
        if 0 in emb:
            emb_dim_sum = emb.sum(dim=0).sum(dim=-1)
            self.assertTrue(0 in emb_dim_sum)


if __name__ == "__main__":
    unittest.main()
