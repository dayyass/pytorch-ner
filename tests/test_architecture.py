import unittest
import torch
import torch.nn as nn
from pytorch_ner.nn_modules.dropout import SpatialDropout1d
from pytorch_ner.nn_modules.normalization import LayerNorm
from pytorch_ner.nn_modules.embedding import Embedding, EmbeddingWithDropout
from pytorch_ner.nn_modules.rnn import DynamicRNN
from pytorch_ner.nn_modules.attention import MultiheadSelfAttention, AttentionWithSkipConnectionLayerNorm
from pytorch_ner.nn_modules.linear import LinearHead
from pytorch_ner.nn_modules.architecture import BiLSTM


tokens = torch.randint(low=0, high=2000, size=(10, 20))
lengths = torch.arange(start=20, end=10, step=-1)


embedding_layer = EmbeddingWithDropout(
    embedding_layer=Embedding(num_embeddings=2000, embedding_dim=128),
    dropout=SpatialDropout1d(p=0.5)
)
rnn_layer = DynamicRNN(rnn_unit=nn.LSTM, input_size=128, hidden_size=256, num_layers=1, dropout=0., bidirectional=True)
attention_layer = AttentionWithSkipConnectionLayerNorm(
    attention_layer=MultiheadSelfAttention(embed_dim=512, num_heads=8, dropout=0.),
    layer_norm=LayerNorm(normalized_shape=512),
    use_skip_connection=True,
)
linear_head = LinearHead(linear_head=nn.Linear(in_features=512, out_features=5))

model = BiLSTM(
    embedding_layer=embedding_layer,
    rnn_layer=rnn_layer,
    attention_layer=attention_layer,
    linear_head=linear_head,
)


class TestBiLSTM(unittest.TestCase):

    def test_inference_shape(self):
        self.assertTrue(model(tokens, lengths).shape == torch.Size([10, 20, 5]))


if __name__ == '__main__':
    unittest.main()
