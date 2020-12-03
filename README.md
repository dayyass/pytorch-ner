### About
Pipeline for training NER models using PyTorch.

### Usage
The user interface consists of only one file *config.yaml*.<br/>
Change *config.yaml* to create the desired configuration and start the pipeline with the following command:
```
python main.py --config config.yaml
```
If *--config* argument is not specified, then used config.yaml.

### Models
List of implemented models:
- [x] BiLTSM
- [ ] BiLTSMCRF
- [x] BiLTSMAttn
- [ ] BiLTSMAttnCRF
- [ ] BiLTSMCNN
- [ ] BiLTSMCNNCRF
- [ ] BiLTSMCNNAttn
- [ ] BiLTSMCNNAttnCRF

<!--
# TODO: add model results
-->
