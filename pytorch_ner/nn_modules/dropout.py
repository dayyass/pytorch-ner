import torch
import torch.nn as nn


class SpatialDropout1d(nn.Module):
    """
    Spatial Dropout drops a certain percentage of dimensions from each word vector in the training sample.
    implementation: https://discuss.pytorch.org/t/spatial-dropout-in-pytorch/21400;
    explanation: https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/76883.
    """

    def __init__(self, p: float):
        super(SpatialDropout1d, self).__init__()
        self.spatial_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 1, 0).unsqueeze(
            0
        )  # convert to [fake_dim, emb_dim, seq_len, batch_size]
        x = self.spatial_dropout(x)
        x = x.squeeze(0).permute(2, 1, 0)  # back to [batch_size, seq_len, emb_dim]
        return x


class WordEmbeddingsDropout(nn.Module):
    """
    Word Embeddings Dropout drops a certain percentage of entire words in the training sample.
    explanation: https://arxiv.org/abs/1512.05287v5.
    """

    def __init__(self, p: float):
        super(WordEmbeddingsDropout, self).__init__()
        self.word_embeddings_dropout = nn.Dropout2d(p=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings_dropout(x)
        return x
