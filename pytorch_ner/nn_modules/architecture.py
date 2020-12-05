import torch
import torch.nn as nn


# TODO: maybe use embedding_layer out of model architecture (another class)
class BiLSTM(nn.Module):
    """
    Bidirectional LSTM architecture.
    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        rnn_layer: nn.Module,
        linear_head: nn.Module,
    ):
        super(BiLSTM, self).__init__()
        self.embedding = embedding_layer  # EMBEDDINGS
        self.rnn = rnn_layer  # RNN
        self.linear_head = linear_head  # LINEAR HEAD

    def forward(self, x: torch.Tensor, x_length: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x)  # EMBEDDINGS
        rnn_out = self.rnn(embed, x_length)  # RNN
        logits = self.linear_head(rnn_out)  # LINEAR HEAD
        return logits


class BiLSTMAttn(nn.Module):
    """
    Bidirectional LSTM with attention architecture.
    """

    def __init__(
        self,
        embedding_layer: nn.Module,
        rnn_layer: nn.Module,
        attention_layer: nn.Module,
        linear_head: nn.Module,
    ):
        super(BiLSTMAttn, self).__init__()
        self.embedding = embedding_layer  # EMBEDDINGS
        self.rnn = rnn_layer  # RNN
        self.attention = attention_layer  # ATTENTION
        self.linear_head = linear_head  # LINEAR HEAD

    def forward(self, x: torch.Tensor, x_length: torch.Tensor) -> torch.Tensor:
        embed = self.embedding(x)  # EMBEDDINGS
        rnn_out = self.rnn(embed, x_length)  # RNN
        attn = self.attention(rnn_out)  # ATTENTION
        logits = self.linear_head(attn)  # LINEAR HEAD
        return logits
