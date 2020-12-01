import torch


def masking(lengths: torch.Tensor) -> torch.BoolTensor:
    """
    Convert lengths tensor to binary mask
    implement: https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    """

    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    return torch.arange(end=max_len).expand(size=(lengths_shape, max_len)) < lengths.unsqueeze(1)
