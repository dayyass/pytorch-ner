from test_nn_modules.test_architecture import model_bilstm as model

from pytorch_ner.onnx import onnx_export_and_check
from pytorch_ner.utils import mkdir

path_to_save = "models/model.onnx"
mkdir("models")

onnx_export_and_check(model=model, path_to_save=path_to_save)
