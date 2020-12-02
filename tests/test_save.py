import os
import yaml
import unittest

from pytorch_ner.save import save_model
from tests.test_train import model, token2idx, label2idx


path_to_folder = 'models/test_save/'
path_to_onnx_folder = 'models/test_onnx_save/'


with open('config.yaml', 'r') as fp:
    config = yaml.safe_load(fp)


# without onnx
save_model(
    path_to_folder=path_to_folder,
    model=model,
    token2idx=token2idx,
    label2idx=label2idx,
    config=config,
    export_onnx=False,
)

# with onnx
save_model(
    path_to_folder=path_to_onnx_folder,
    model=model,
    token2idx=token2idx,
    label2idx=label2idx,
    config=config,
    export_onnx=True,
)


class TestSave(unittest.TestCase):

    def test_mkdir(self):
        self.assertTrue(os.path.exists(path_to_folder))

    def test_num_files(self):
        self.assertTrue(len(os.listdir(path_to_folder)) == 4)

    def test_num_files_with_onnx(self):
        self.assertTrue(len(os.listdir(path_to_onnx_folder)) == 5)


if __name__ == '__main__':
    unittest.main()
