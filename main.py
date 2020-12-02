import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import Counter

from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx
from pytorch_ner.dataset import NERDataset, NERCollator

from pytorch_ner.nn_modules.embedding import Embedding
from pytorch_ner.nn_modules.rnn import DynamicRNN


with open('config.yaml', 'r') as fp:
    config = yaml.safe_load(fp)

device = torch.device(config['torch']['device'])

train_token_seq, train_label_seq = prepare_conll_data_format(
    path=config['prepare_data']['train_data']['path'],
    sep=config['prepare_data']['train_data']['sep'],
    lower=config['prepare_data']['train_data']['lower'],
    verbose=config['prepare_data']['train_data']['verbose'],
)

val_token_seq, val_label_seq = prepare_conll_data_format(
    path=config['prepare_data']['val_data']['path'],
    sep=config['prepare_data']['val_data']['sep'],
    lower=config['prepare_data']['val_data']['lower'],
    verbose=config['prepare_data']['val_data']['verbose'],
)

test_token_seq, test_label_seq = prepare_conll_data_format(
    path=config['prepare_data']['test_data']['path'],
    sep=config['prepare_data']['test_data']['sep'],
    lower=config['prepare_data']['test_data']['lower'],
    verbose=config['prepare_data']['test_data']['verbose'],
)

# print(train_token_seq)
# print(val_token_seq)
# print(test_token_seq)

token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

token2idx = get_token2idx(
    token2cnt=token2cnt,
    min_count=config['prepare_data']['token2idx']['min_count'],
    add_pad=config['prepare_data']['token2idx']['add_pad'],
    add_unk=config['prepare_data']['token2idx']['add_unk'],
)

label2idx = get_label2idx(label_set=label_set)

# print(token2idx)
# print(label2idx)

trainset = NERDataset(
    token_seq=train_token_seq,
    label_seq=train_label_seq,
    token2idx=token2idx,
    label2idx=label2idx,
    preprocess=config['dataloader']['preprocess'],
)

valset = NERDataset(
    token_seq=val_token_seq,
    label_seq=val_label_seq,
    token2idx=token2idx,
    label2idx=label2idx,
    preprocess=config['dataloader']['preprocess'],
)

testset = NERDataset(
    token_seq=test_token_seq,
    label_seq=test_label_seq,
    token2idx=token2idx,
    label2idx=label2idx,
    preprocess=config['dataloader']['preprocess'],
)

# print(trainset)
# print(valset)
# print(testset)

train_collator = NERCollator(
    token_padding_value=token2idx[config['dataloader']['token_padding']],
    label_padding_value=label2idx[config['dataloader']['label_padding']],
    percentile=config['dataloader']['percentile'],
)

val_collator = NERCollator(
    token_padding_value=token2idx[config['dataloader']['token_padding']],
    label_padding_value=label2idx[config['dataloader']['label_padding']],
    percentile=100,  # hardcoded
)

test_collator = NERCollator(
    token_padding_value=token2idx[config['dataloader']['token_padding']],
    label_padding_value=label2idx[config['dataloader']['label_padding']],
    percentile=100,  # hardcoded
)

# TODO: add more params to config.yaml
trainloader = DataLoader(
    dataset=trainset,
    batch_size=config['dataloader']['batch_size'],
    shuffle=True,  # hardcoded
    collate_fn=train_collator,
)

valloader = DataLoader(
    dataset=valset,
    batch_size=1,  # hardcoded
    shuffle=False,  # hardcoded
    collate_fn=val_collator,
)

testloader = DataLoader(
    dataset=testset,
    batch_size=1,  # hardcoded
    shuffle=False,  # hardcoded
    collate_fn=test_collator,
)

# print(len(trainloader))
# print(len(valloader))
# print(len(testloader))

# TODO: add more params to config.yaml
embedding_layer = Embedding(
    num_embeddings=len(token2idx),
    embedding_dim=config['model']['embedding']['embedding_dim'],
)

# print(embedding_layer)

rnn_layer = DynamicRNN(
    rnn_unit=eval(config['model']['rnn']['rnn_unit']),  # TODO: fix eval
    input_size=config['model']['embedding']['embedding_dim'],  # reference to embedding_dim
    hidden_size=config['model']['rnn']['hidden_size'],
    num_layers=config['model']['rnn']['num_layers'],
    dropout=config['model']['rnn']['dropout'],
    bidirectional=config['model']['rnn']['bidirectional'],
)

# print(rnn_layer)
