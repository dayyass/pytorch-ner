import unittest

import torch

from pytorch_ner.nn_modules.normalization import LayerNorm

embeddings = torch.randn(10, 20, 128)  # [batch_size, seq_len, emb_dim]

layer_norm = LayerNorm(128)


class TestNormalization(unittest.TestCase):
    def test_layer_norm_shape(self):
        self.assertTrue(
            layer_norm(embeddings).size() == embeddings.size(),
        )


if __name__ == "__main__":
    unittest.main()
