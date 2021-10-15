from typing import DefaultDict, Dict, List

import numpy as np
from sklearn.metrics import f1_score


def calculate_metrics(
    metrics: DefaultDict[str, List[float]],
    loss: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    idx2label: Dict[int, str],
) -> DefaultDict[str, List[float]]:
    """
    Calculate metrics on epoch.
    """

    metrics["loss"].append(loss)

    f1_per_class = f1_score(
        y_true=y_true, y_pred=y_pred, labels=range(len(idx2label)), average=None
    )
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average="weighted")
    for cls, f1 in enumerate(f1_per_class):
        metrics[f"f1 {idx2label[cls]}"].append(f1)
    metrics["f1-weighted"].append(f1_weighted)

    return metrics
