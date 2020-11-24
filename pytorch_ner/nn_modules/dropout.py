import torch
import torch.nn as nn


class SpatialDropout1d(nn.Module):

    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883
    """

    def __init__(self, p: float):
        super(SpatialDropout1d, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(0, 2)  # convert to [emb_dim, seq_len, batch_size]
        x = self.spatial_dropout(x)
        x = x.transpose(0, 2)  # back to [batch_size, seq_len, emb_dim]
        return x
