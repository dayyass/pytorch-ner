import unittest
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tests.test_nn_modules.test_architecture import model_bilstm as model

from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx
from pytorch_ner.dataset import NERDataset, NERCollator
from pytorch_ner.train import train, validate_loop


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# LOAD DATA

token_seq, label_seq = prepare_conll_data_format('tests/data/conll.txt', verbose=False)

token2cnt = Counter([token for sentence in token_seq for token in sentence])
label_set = sorted(set(label for sentence in label_seq for label in sentence))

token2idx = get_token2idx(token2cnt)
label2idx = get_label2idx(label_set)

dataset = NERDataset(token_seq=token_seq, label_seq=label_seq, token2idx=token2idx, label2idx=label2idx)
collator = NERCollator(token_padding_value=token2idx['<PAD>'], label_padding_value=label2idx['O'], percentile=95)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)


# INIT MODEL

model.to(device)


# CRITERION AND OPTIMIZER

criterion = nn.CrossEntropyLoss(reduction='none')
optimizer = optim.Adam(model.parameters())


# TRAIN MODEL

train(
    model=model,
    trainloader=dataloader,
    valloader=dataloader,
    testloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    n_epoch=10,
    verbose=False,
)


class TestTrain(unittest.TestCase):

    # TODO: fix it - not always True
    def test_val_metrics(self):

        val_metrics = validate_loop(
            model=model.to(device),
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            verbose=False,
        )

        for metric_name, metric_list in val_metrics.items():
            if metric_name.startswith('f1'):
                self.assertTrue(np.mean(metric_list) == 1.0)


if __name__ == '__main__':
    unittest.main()
