import os
import unittest
from pathlib import Path

import yaml

from pytorch_ner.save import save_model
from pytorch_ner.utils import rmdir
from tests.test_train import label2idx, model, token2idx

path_to_save_folder = Path("models/test")
path_to_no_onnx_folder = path_to_save_folder / "no_onnx"
path_to_onnx_folder = path_to_save_folder / "onnx"

path_to_no_onnx_folder.absolute().mkdir(parents=True, exist_ok=True)
path_to_onnx_folder.absolute().mkdir(parents=True, exist_ok=True)


path_to_config = "config.yaml"

with open(path_to_config, mode="r") as fp:
    config = yaml.safe_load(fp)
    config["save"]["path_to_config"] = path_to_config


# without onnx
save_model(
    path_to_folder=str(path_to_no_onnx_folder),
    model=model,
    token2idx=token2idx,
    label2idx=label2idx,
    config=config,
    export_onnx=False,
)

# with onnx
save_model(
    path_to_folder=str(path_to_onnx_folder),
    model=model,
    token2idx=token2idx,
    label2idx=label2idx,
    config=config,
    export_onnx=True,
)


class TestSave(unittest.TestCase):
    def test_num_files(self):
        """Check 4 files"""
        self.assertTrue(len(os.listdir(path_to_no_onnx_folder)) == 4)

    def test_num_files_with_onnx(self):
        """Check 5 files"""
        self.assertTrue(len(os.listdir(path_to_onnx_folder)) == 5)

    @classmethod
    def tearDownClass(cls):
        rmdir(path_to_save_folder)


if __name__ == "__main__":
    unittest.main()
