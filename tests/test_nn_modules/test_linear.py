import unittest

import torch
import torch.nn as nn

from pytorch_ner.nn_modules.linear import LinearHead

embeddings = torch.randn(10, 20, 128)  # [batch_size, seq_len, emb_dim]

# BILUO head
linear_head = LinearHead(
    nn.Sequential(
        nn.Linear(128, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 5),
    ),
)


class TestLinearHead(unittest.TestCase):
    def test_linear_shape(self):
        self.assertTrue(
            linear_head(embeddings).size() == torch.Size([10, 20, 5]),
        )


if __name__ == "__main__":
    unittest.main()
