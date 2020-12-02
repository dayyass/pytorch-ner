import numpy as np

import torch
import torch.nn as nn

import onnx
import onnxruntime

from pytorch_ner.utils import to_numpy


def onnx_export(
        model: nn.Module,
        path_to_save: str,
):

    model.eval()

    # hardcoded [batch_size, seq_len] = [1, 1] export
    tokens = torch.randint(low=0, high=2000, size=(1, 1))
    lengths = torch.tensor([1], dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=(tokens, lengths),
            f=path_to_save,
            export_params=True,
            opset_version=12,  # hardcoded
            do_constant_folding=True,  # hardcoded
            input_names=['tokens', 'lengths'],
            output_names=['output'],
            dynamic_axes={
                'tokens': {0: 'batch_size', 1: 'seq_len'},
                'lengths': {0: 'batch_size'},
                'output': {0: 'batch_size', 1: 'seq_len'},
            },
        )


def onnx_check_model(path_to_load: str):
    """
    Check that the IR is well formed.
    """

    onnx_model = onnx.load(path_to_load)
    onnx.checker.check_model(onnx_model)


def onnx_check_inference(model: nn.Module, path_to_load: str, tokens: torch.Tensor, lengths: torch.Tensor):
    """
    Compute ONNX Runtime output prediction and compare with PyTorch results.
    """

    # pytorch inference
    model.eval()
    torch_out = model(tokens, lengths)

    # onnx inference
    ort_session = onnxruntime.InferenceSession(path_to_load)
    ort_inputs = {'tokens': to_numpy(tokens), 'lengths': to_numpy(lengths)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
