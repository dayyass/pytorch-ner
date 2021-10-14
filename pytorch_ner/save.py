import json
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from pytorch_ner.onnx import onnx_export_and_check


def save_model(
    path_to_folder: str,
    model: nn.Module,
    token2idx: Dict[str, int],
    label2idx: Dict[str, int],
    config: Dict,
    export_onnx: bool = False,
):

    path_to_save = Path(path_to_folder)

    # save torch model
    model.cpu()
    model.eval()
    torch.save(model.state_dict(), path_to_save / "model.pth")

    # save token2idx
    with open(file=path_to_save / "token2idx.json", mode="w") as fp:
        json.dump(token2idx, fp)

    # save label2idx
    with open(file=path_to_save / "label2idx.json", mode="w") as fp:
        json.dump(label2idx, fp)

    # save config
    shutil.copy2(config["save"]["path_to_config"], path_to_save / "config.yaml")

    # save onnx model
    if export_onnx:
        onnx_export_and_check(
            model=model,
            path_to_save=str(path_to_save / "model.onnx"),
        )  # type: ignore
