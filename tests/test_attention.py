import unittest
import torch
from pytorch_ner.nn_modules.attention import MultiheadSelfAttention


embeddings = torch.randn(10, 20, 128)  # [batch_size, seq_len, emb_dim]

attention = MultiheadSelfAttention(
    embed_dim=128,
    num_heads=8,
    dropout=0.2,
)


class TestAttention(unittest.TestCase):

    def test_attention_shape(self):
        self.assertTrue(
            attention(embeddings).size() == embeddings.size(),
        )


if __name__ == '__main__':
    unittest.main()
