# Named Entity Recognition (NER) with PyTorch

![test Status](https://github.com/dayyass/pytorch_ner/workflows/test/badge.svg)
![lint Status](https://github.com/dayyass/pytorch_ner/workflows/lint/badge.svg)
![License](https://img.shields.io/github/license/dayyass/pytorch_ner)
![release (latest by date)](https://img.shields.io/github/v/release/dayyass/pytorch_ner)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

### About
Pipeline for training NER models using PyTorch.<br/>
ONNX export supported.<br/>

### Usage
The user interface consists of only one file *config.yaml*.<br/>
Change *config.yaml* to create the desired configuration and start the pipeline with the following command:
```
python main.py --config config.yaml
```
If *--config* argument is not specified, then used config.yaml.

To export trained model to ONNX use config.yaml:
```
save:
  export_onnx: True
```

### Data Format:
Text file containing separated tokens and labels on each line. Sentences are separated by empty line.
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

### Docker
To simplify installation, you can deploy a container with all dependencies pre-installed.

Build container<br/>
`$ docker build -t pytorch_ner .`


Run it (add `--gpus all` to use GPUs)<br/>
`$ docker container run --rm -it -v ${PWD}:/workspace -p 6006:6006 pytorch_ner`


### Requirements
Python 3.6+

<!--
# TODO: add model results
-->
