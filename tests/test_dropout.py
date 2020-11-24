import unittest
import torch
from pytorch_ner.nn_modules.dropout import SpatialDropout1d
from pytorch_ner.utils import set_seed

set_seed(42)  # reproducibility


class TestSpatialDropout(unittest.TestCase):

    spatial_dropout = SpatialDropout1d(p=0.5)
    embeddings = torch.randn(10, 2, 5)  # [batch_size, seq_len, emb_dim]

    def test_spatial_dropout(self):

        emb = self.spatial_dropout(self.embeddings)
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


if __name__ == '__main__':
    unittest.main()
