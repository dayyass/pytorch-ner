import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization
    paper: https://arxiv.org/abs/1607.06450
    """

    def __init__(self, normalized_shape: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)
