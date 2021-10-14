import os

from pytorch_ner.onnx import onnx_export_and_check
from tests.test_nn_modules.test_architecture import model_bilstm as model

path_to_save = "models/model.onnx"

onnx_export_and_check(model=model, path_to_save=path_to_save)

# tear down
os.remove(path_to_save)
