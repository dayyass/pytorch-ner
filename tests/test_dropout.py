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
        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb[:, 0, :].sum(dim=0)[2],  # third embedding dim
            ),
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb[:, 1, :].sum(dim=0)[0],  # zeros embedding dim
            ),
        )
        self.assertTrue(
            torch.equal(
                torch.tensor(0.), emb[:, 1, :].sum(dim=0)[4],  # fifth embedding dim
            ),
        )


if __name__ == '__main__':
    unittest.main()
