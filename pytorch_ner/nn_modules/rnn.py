import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    """
    RNN layer wrapper to handle variable-size input.
    """

    def __init__(
            self,
            rnn_unit: nn.Module,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            dropout: float,
            bidirectional: bool,
            batch_first: bool = True,
    ):
        super(DynamicRNN, self).__init__()
        self.rnn = rnn_unit(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(
            self,
            x: torch.Tensor,
            x_length: torch.Tensor,
    ) -> torch.Tensor:
        packed_x = pack_padded_sequence(x, x_length, batch_first=self.batch_first)
        packed_rnn_out, _ = self.rnn(packed_x)
        rnn_out, _ = pad_packed_sequence(packed_rnn_out, batch_first=self.batch_first)
        return rnn_out
