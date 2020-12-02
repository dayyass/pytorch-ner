import os
import torch
from warnings import filterwarnings

from tests.test_architecture import model_bilstm as model
from pytorch_ner.onnx import onnx_export, onnx_check_model, onnx_check_inference

filterwarnings(action='ignore', category=UserWarning)


path_to_save = 'models/model.onnx'

if not os.path.exists('models'):
    os.makedirs('models')


tokens = torch.randint(low=0, high=2000, size=(1, 1))
lengths = torch.tensor([1], dtype=torch.long)

tokens_dynamic = torch.randint(low=0, high=2000, size=(10, 20))
lengths_dynamic = torch.tensor(10 * [20], dtype=torch.long)


onnx_export(model=model, path_to_save=path_to_save)
onnx_check_model(path_to_load=path_to_save)
onnx_check_inference(model=model, path_to_load=path_to_save, tokens=tokens, lengths=lengths)
onnx_check_inference(model=model, path_to_load=path_to_save, tokens=tokens_dynamic, lengths=lengths_dynamic)
