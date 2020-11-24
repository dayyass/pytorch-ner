import torch
import torch.nn as nn


class SpatialDropout(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: int):
        super(SpatialDropout, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # convert to [batch, channels, time]
        x = self.spatial_dropout(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, channels]
        return x
