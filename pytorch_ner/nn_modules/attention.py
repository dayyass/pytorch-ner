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


class AttentionWithSkipConnectionLayerNorm(nn.Module):
    """
    Any kind of Self-Attention with skip-connection and LayerNorm.
    Transformer-like.
    """

    def __init__(
        self,
        attention_layer: nn.Module,
        layer_norm: nn.Module,
        use_skip_connection: bool = True,
    ):
        super(AttentionWithSkipConnectionLayerNorm, self).__init__()
        self.attention = attention_layer
        self.layer_norm = layer_norm
        self.use_skip_connection = use_skip_connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(x)

        if self.use_skip_connection:
            layer_norm_input = attn_out + x
        else:
            layer_norm_input = attn_out

        layer_norm_out = self.layer_norm(layer_norm_input)
        return layer_norm_out
