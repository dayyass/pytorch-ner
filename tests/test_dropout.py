import unittest
import torch
from pytorch_ner.nn_modules.dropout import SpatialDropout1d, WordEmbeddingsDropout
from pytorch_ner.utils import set_global_seed

set_global_seed(42)  # reproducibility


embeddings = torch.randn(2, 5, 10)  # [batch_size, seq_len, emb_dim]

spatial_dropout = SpatialDropout1d(p=0.5)
word_embeddings_dropout = WordEmbeddingsDropout(p=0.5)


class TestDropout(unittest.TestCase):

    def test_spatial_dropout(self):

        emb = spatial_dropout(embeddings)
        emb_dim_sum = emb.sum(dim=0).sum(dim=0)

        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb_dim_sum[1],  # second embedding dim
            ),
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb_dim_sum[4],  # fifth embedding dim
            ),
        )

    def test_word_embeddings_dropout(self):

        emb = word_embeddings_dropout(embeddings)
        emb_dim_sum = emb.sum(dim=0).sum(dim=-1)

        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb_dim_sum[0],  # first embedding dim
            ),
        )


if __name__ == '__main__':
    unittest.main()
