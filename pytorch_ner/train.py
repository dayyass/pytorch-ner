from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from pytorch_ner.metrics import calculate_metrics
from pytorch_ner.utils import to_numpy


def masking(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths tensor to binary mask
    implement: https://stackoverflow.com/questions/53403306/how-to-batch-convert-sentence-lengths-to-masks-in-pytorch
    """

    device = lengths.device
    lengths_shape = lengths.shape[0]
    max_len = lengths.max()
    return torch.arange(end=max_len, device=device).expand(
        size=(lengths_shape, max_len)
    ) < lengths.unsqueeze(1)


# TODO: clip_grad_norm
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    verbose: bool = True,
) -> DefaultDict[str, List[float]]:
    """
    Training loop on one epoch.
    """

    metrics: DefaultDict[str, List[float]] = defaultdict(list)
    idx2label = {v: k for k, v in dataloader.dataset.label2idx.items()}

    if verbose:
        dataloader = tqdm(dataloader)

    model.train()

    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(device),
            labels.to(device),
            lengths.to(device),
        )

        mask = masking(lengths)

        # forward pass
        logits = model(tokens, lengths)
        loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
        loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # make predictions
        y_true = to_numpy(labels[mask])
        y_pred = to_numpy(logits.argmax(dim=-1)[mask])

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
            idx2label=idx2label,
        )

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    verbose: bool = True,
) -> DefaultDict[str, List[float]]:
    """
    Validate loop on one epoch.
    """

    metrics: DefaultDict[str, List[float]] = defaultdict(list)
    idx2label = {v: k for k, v in dataloader.dataset.label2idx.items()}

    if verbose:
        dataloader = tqdm(dataloader)

    model.eval()

    for tokens, labels, lengths in dataloader:
        tokens, labels, lengths = (
            tokens.to(device),
            labels.to(device),
            lengths.to(device),
        )

        mask = masking(lengths)

        # forward pass
        with torch.no_grad():
            logits = model(tokens, lengths)
            loss_without_reduction = criterion(logits.transpose(-1, -2), labels)
            loss = torch.sum(loss_without_reduction * mask) / torch.sum(mask)

        # make predictions
        y_true = to_numpy(labels[mask])
        y_pred = to_numpy(logits.argmax(dim=-1)[mask])

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
            idx2label=idx2label,
        )

    return metrics


# TODO: add metrics as callback
# TODO: add TensorBoard support
# TODO: add EarlyStopping support
# TODO: add ModelCheckpoint support
def train(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epoch: int,
    testloader: Optional[DataLoader] = None,
    verbose: bool = True,
):
    """
    Training / validation loop for n_epoch with final testing.
    """

    for epoch in range(n_epoch):

        if verbose:
            print(f"epoch [{epoch+1}/{n_epoch}]\n")

        train_metrics = train_epoch(
            model=model,
            dataloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in train_metrics.items():
                print(f"train {metric_name}: {np.mean(metric_list)}")
            print()

        val_metrics = validate_epoch(
            model=model,
            dataloader=valloader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in val_metrics.items():
                print(f"val {metric_name}: {np.mean(metric_list)}")
            print()

    if testloader is not None:

        test_metrics = validate_epoch(
            model=model,
            dataloader=testloader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in test_metrics.items():
                print(f"test {metric_name}: {np.mean(metric_list)}")
            print()
