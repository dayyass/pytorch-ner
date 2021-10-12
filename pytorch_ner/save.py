import json
import os
from typing import Dict

import torch
import torch.nn as nn
import yaml

from pytorch_ner.onnx import onnx_export_and_check
from pytorch_ner.utils import mkdir, rmdir


def save_model(
    path_to_folder: str,
    model: nn.Module,
    token2idx: Dict[str, int],
    label2idx: Dict[str, int],
    config: Dict,
    export_onnx: bool = False,
):

    if not os.path.exists(path_to_folder):
        # make empty dir
        mkdir(path_to_folder)

    model.cpu()
    model.eval()

    # save torch model
    torch.save(model.state_dict(), os.path.join(path_to_folder, "model.pth"))

    # save token2idx
    with open(file=os.path.join(path_to_folder, "token2idx.json"), mode="w") as fp:
        json.dump(token2idx, fp)

    # save label2idx
    with open(file=os.path.join(path_to_folder, "label2idx.json"), mode="w") as fp:
        json.dump(label2idx, fp)

    # save config
    with open(file=os.path.join(path_to_folder, "config.yaml"), mode="w") as fp:
        yaml.dump(config, fp)

    # save onnx model
    if export_onnx:
        onnx_export_and_check(
            model=model, path_to_save=os.path.join(path_to_folder, "model.onnx")
        )
