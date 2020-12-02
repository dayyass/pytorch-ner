import os
import torch
from warnings import filterwarnings

from tests.test_nn_modules.test_architecture import model_bilstm as model
from pytorch_ner.onnx import onnx_export_and_check

# TODO: fix it
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
filterwarnings(action='ignore', category=UserWarning)


path_to_save = 'models/model.onnx'

if not os.path.exists('models'):
    os.makedirs('models')


tokens = torch.randint(low=0, high=2000, size=(1, 1))
lengths = torch.tensor([1], dtype=torch.long)

tokens_dynamic = torch.randint(low=0, high=2000, size=(10, 20))
lengths_dynamic = torch.tensor(10 * [20], dtype=torch.long)


onnx_export_and_check(model=model, path_to_save=path_to_save)
