import unittest
import torch
import torch.nn as nn
from pytorch_ner.nn_modules.rnn import DynamicRNN


embeddings = torch.randn(10, 20, 100)  # [batch_size, seq_len, emb_dim]
length = torch.arange(start=20, end=10, step=-1)


rnn = DynamicRNN(
    rnn_unit=nn.RNN,
    input_size=100,
    hidden_size=200,
    num_layers=3,
    dropout=0,
    bidirectional=False,
)

gru = DynamicRNN(
    rnn_unit=nn.GRU,
    input_size=100,
    hidden_size=50,
    num_layers=2,
    dropout=0.2,
    bidirectional=True,
)

lstm = DynamicRNN(
    rnn_unit=nn.LSTM,
    input_size=100,
    hidden_size=150,
    num_layers=1,
    dropout=0,
    bidirectional=True,
)


class TestRNN(unittest.TestCase):

    def test_rnn_shape(self):
        self.assertTrue(
            rnn(embeddings, length).size() == torch.Size([10, 20, 200]),
        )

    def test_gru_shape(self):
        self.assertTrue(
            gru(embeddings, length).size() == torch.Size([10, 20, 100]),
        )

    def test_lstm_shape(self):
        self.assertTrue(
            lstm(embeddings, length).size() == torch.Size([10, 20, 300]),
        )


if __name__ == '__main__':
    unittest.main()
