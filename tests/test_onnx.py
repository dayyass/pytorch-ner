import os
import torch
import unittest
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


class TestONNX(unittest.TestCase):

    def test_onnx_check_model(self):
        onnx_check_model(path_to_load=path_to_save)

    def test_onnx_check_inference(self):
        onnx_check_inference(model=model, path_to_load=path_to_save, tokens=tokens, lengths=lengths)

    def test_onnx_check_dynamic_inference(self):
        onnx_check_inference(model=model, path_to_load=path_to_save, tokens=tokens_dynamic, lengths=lengths_dynamic)


if __name__ == '__main__':
    unittest.main()
