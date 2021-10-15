[![tests](https://github.com/dayyass/pytorch-ner/actions/workflows/tests.yml/badge.svg)](https://github.com/dayyass/pytorch-ner/actions/workflows/tests.yml)
[![linter](https://github.com/dayyass/pytorch-ner/actions/workflows/linter.yml/badge.svg)](https://github.com/dayyass/pytorch-ner/actions/workflows/linter.yml)
[![codecov](https://codecov.io/gh/dayyass/pytorch-ner/branch/main/graph/badge.svg?token=WSB83O6GVV)](https://codecov.io/gh/dayyass/pytorch-ner)

[![python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://github.com/dayyass/pytorch-ner#requirements)
[![release (latest by date)](https://img.shields.io/github/v/release/dayyass/pytorch-ner)](https://github.com/dayyass/pytorch-ner/releases/latest)
[![license](https://img.shields.io/github/license/dayyass/pytorch-ner?color=blue)](https://github.com/dayyass/pytorch-ner/blob/main/LICENSE)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-black)](https://github.com/dayyass/pytorch-ner/blob/main/.pre-commit-config.yaml)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![pypi version](https://img.shields.io/pypi/v/pytorch-ner)](https://pypi.org/project/pytorch-ner)
[![pypi downloads](https://img.shields.io/pypi/dm/pytorch-ner)](https://pypi.org/project/pytorch-ner)

### Named Entity Recognition (NER) with PyTorch
Pipeline for training **NER** models using **PyTorch**.

**ONNX** export supported.

### Usage
Instead of writing custom code for specific NER task, you just need:
1. install pipeline:
```shell script
pip install pytorch-ner
```
2. run pipeline:
- either in **terminal**:
```shell script
pytorch-ner-train --path_to_config config.yaml
```
- or in **python**:
```python3
import pytorch_ner

pytorch_ner.train(path_to_config="config.yaml")
```

#### Config
The user interface consists of only one file [**config.yaml**](https://github.com/dayyass/pytorch-ner/blob/main/config.yaml).<br/>
Change it to create the desired configuration.

Default **config.yaml**:
```yaml
torch:
  device: 'cpu'
  seed: 42

data:
  train_data:
    path: 'data/conll2003/train.txt'
    sep: ' '
    lower: true
    verbose: true
  valid_data:
    path: 'data/conll2003/valid.txt'
    sep: ' '
    lower: true
    verbose: true
  test_data:
    path: 'data/conll2003/test.txt'
    sep: ' '
    lower: true
    verbose: true
  token2idx:
    min_count: 1
    add_pad: true
    add_unk: true

dataloader:
  preprocess: true
  token_padding: '<PAD>'
  label_padding: 'O'
  percentile: 100
  batch_size: 256

model:
  embedding:
    embedding_dim: 128
  rnn:
    rnn_unit: LSTM  # GRU, RNN
    hidden_size: 256
    num_layers: 1
    dropout: 0
    bidirectional: true

optimizer:
  optimizer_type: Adam  # torch.optim
  clip_grad_norm: 0.1
  params:
    lr: 0.001
    weight_decay: 0
    amsgrad: false

train:
  n_epoch: 10
  verbose: true

save:
  path_to_folder: 'models'
  export_onnx: true
```

**NOTE**: to export trained model to **ONNX** use the following config parameter:
```
save:
  export_onnx: true
```

#### Data Format
Pipeline works with text file containing separated tokens and labels on each line. Sentences are separated by empty line.
Labels should already be in necessary format, e.g. IO, BIO, BILUO, ...

Example:
```
token_11    label_11
token_12    label_12

token_21    label_21
token_22    label_22
token_23    label_23

...
```

#### Output
After training the model, the pipeline will return the following files:
- `model.pth` - pytorch NER model
- `model.onnx` - onnx NER model (optional)
- `token2idx.json` - mapping from token to its index
- `label2idx.json` - mapping from label to its index
- `config.yaml` - config that was used to train the model
- `logging.txt` - logging file

### Models
List of implemented models:
- [x] BiLTSM
- [ ] BiLTSMCRF
- [ ] BiLTSMAttn
- [ ] BiLTSMAttnCRF
- [ ] BiLTSMCNN
- [ ] BiLTSMCNNCRF
- [ ] BiLTSMCNNAttn
- [ ] BiLTSMCNNAttnCRF

### Evaluation
All results are obtained on CoNLL-2003 [dataset](https://github.com/dayyass/pytorch-ner/tree/develop/data/conll2003). We didn't search the best parameters.

| Model  | Train F1-weighted | Validation F1-weighted | Test F1-weighted |
| ------ | ----------------- | ---------------------- | ---------------- |
| BiLSTM | 0.968             | 0.928                  | 0.876            |

### Requirements
Python >= 3.6

### Citation
If you use **pytorch_ner** in a scientific publication, we would appreciate references to the following BibTex entry:
```bibtex
@misc{dayyass2020ner,
    author       = {El-Ayyass, Dani},
    title        = {Pipeline for training NER models using PyTorch},
    howpublished = {\url{https://github.com/dayyass/pytorch_ner}},
    year         = {2020}
}
```

# TODO: docker cuda
