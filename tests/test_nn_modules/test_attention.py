import unittest

import torch

from pytorch_ner.nn_modules.attention import (
    AttentionWithSkipConnectionLayerNorm,
    MultiheadSelfAttention,
)
from pytorch_ner.nn_modules.normalization import LayerNorm

embeddings = torch.randn(10, 20, 128)  # [batch_size, seq_len, emb_dim]

attention = MultiheadSelfAttention(
    embed_dim=128,
    num_heads=8,
    dropout=0.2,
)

layer_norm = LayerNorm(128)

attention_with_layer_norm = AttentionWithSkipConnectionLayerNorm(
    attention_layer=attention,
    layer_norm=layer_norm,
    use_skip_connection=False,
)

attention_with_skip_connection_layer_norm = AttentionWithSkipConnectionLayerNorm(
    attention_layer=attention,
    layer_norm=layer_norm,
    use_skip_connection=True,
)


class TestAttention(unittest.TestCase):
    def test_attention_shape(self):
        self.assertEqual(attention(embeddings).size(), embeddings.size())

    def test_attention_with_layer_norm_shape(self):
        self.assertEqual(
            attention_with_layer_norm(embeddings).size(), embeddings.size()
        )

    def test_attention_with_skip_connection_layer_norm_shape(self):
        self.assertEqual(
            attention_with_skip_connection_layer_norm(embeddings).size(),
            embeddings.size(),
        )


if __name__ == "__main__":
    unittest.main()
