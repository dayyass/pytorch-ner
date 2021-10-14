import logging
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
    clip_grad_norm: float,
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

        # gradient clipping
        nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=clip_grad_norm,
            norm_type=2,
        )

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
def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad_norm: float,
    n_epoch: int,
    logger: logging.Logger,
    verbose: bool = True,
    test_loader: Optional[DataLoader] = None,
):
    """
    Training / validation loop for n_epoch with final testing.
    """

    # sanity check
    logger.info("Sanity Check starting...")
    tokens, _, lengths = next(iter(valid_loader))
    tokens, lengths = tokens.to(device), lengths.to(device)
    with torch.no_grad():
        _ = model(tokens, lengths)
    logger.info("Sanity Check passed!\n")

    for epoch in range(n_epoch):

        if verbose:
            logger.info(f"epoch [{epoch+1}/{n_epoch}]\n")

        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            clip_grad_norm=clip_grad_norm,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in train_metrics.items():
                logger.info(f"train {metric_name}: {np.mean(metric_list)}")
            logger.info("\n")

        valid_metrics = validate_epoch(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in valid_metrics.items():
                logger.info(f"valid {metric_name}: {np.mean(metric_list)}")
            logger.info("\n")

    if test_loader is not None:

        test_metrics = validate_epoch(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in test_metrics.items():
                logger.info(f"test {metric_name}: {np.mean(metric_list)}")
            logger.info("\n")
