import json
import os
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from pytorch_ner.onnx import onnx_export_and_check
from pytorch_ner.utils import mkdir, rmdir


def model_checkpoint(
    model: nn.Module,
    epoch: int,
    save_best_weights: bool,
    val_metrics,
    val_losses,
    path_to_folder: str,
    export_onnx: bool,
    save_frequency: int,
):

    """
    This function creates check point based on either one of the two scenarios:
        1. Save best weights regarding the val_loss
        2. Save weights frequently with save_frequency int

    """
    if save_best_weights:
        if np.mean(val_metrics["loss"]) < min(val_losses):
            # This iteration has lower val_loss, let's save it
            val_losses.append(np.mean(val_metrics["loss"]))
            pth_file_name = "best_model.pth"
            onnx_file_name = "best_model.onnx"
        else:
            # No need to save weights
            return
    else:
        if epoch % save_frequency == 0:
            # We're at multiple of save_frequency, let's save weights
            pth_file_name = "model_epoch_" + str(epoch) + ".pth"
            onnx_file_name = "model_epoch_" + str(epoch) + ".onnx"
        else:
            # No need to save weights
            return

    torch.save(model.state_dict(), os.path.join(path_to_folder, pth_file_name))
    if export_onnx:
        onnx_export_and_check(
            model=model, path_to_save=os.path.join(path_to_folder, onnx_file_name)
        )
