import torch
import torch.nn as nn


# TODO: parametrized linear with number of layers and features
class LinearHead(nn.Module):
    """
    Linear layer wrapper.
    """

    def __init__(self, linear_head: nn.Module):
        super(LinearHead, self).__init__()
        self.linear_head = linear_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_head(x)
