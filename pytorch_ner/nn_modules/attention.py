import torch
import torch.nn as nn


class MultiheadSelfAttention(nn.Module):
    """
    nn.MultiheadAttention wrapper.
    paper: https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float,
    ):
        super(MultiheadSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 1)  # convert to [seq_len, batch_size, emb_dim]
        attn, _ = self.attention(query=x, key=x, value=x)
        attn = attn.transpose(0, 1)  # back to [batch_size, seq_len, emb_dim]
        return attn
