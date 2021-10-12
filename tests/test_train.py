import unittest
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_ner.dataset import NERCollator, NERDataset
from pytorch_ner.prepare_data import (
    get_label2idx,
    get_token2idx,
    prepare_conll_data_format,
)
from pytorch_ner.train import train, validate_loop
from tests.test_nn_modules.test_architecture import model_bilstm as model

device = torch.device("cpu")


# LOAD DATA

token_seq, label_seq = prepare_conll_data_format(
    path="tests/data/conll.txt", verbose=False
)

token2cnt = Counter([token for sentence in token_seq for token in sentence])
label_set = sorted(set(label for sentence in label_seq for label in sentence))

token2idx = get_token2idx(token2cnt)
label2idx = get_label2idx(label_set)

dataset = NERDataset(
    token_seq=token_seq, label_seq=label_seq, token2idx=token2idx, label2idx=label2idx
)
collator = NERCollator(
    token_padding_value=token2idx["<PAD>"],
    label_padding_value=label2idx["O"],
    percentile=95,
)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collator)


# INIT MODEL

model.to(device)


# CRITERION AND OPTIMIZER

criterion = nn.CrossEntropyLoss(reduction="none")
optimizer = optim.Adam(model.parameters())


# VALIDATE

metrics_before = validate_loop(
    model=model.to(device),
    dataloader=dataloader,
    criterion=criterion,
    device=device,
    verbose=False,
)


# TRAIN MODEL

train(
    model=model,
    trainloader=dataloader,
    valloader=dataloader,
    testloader=dataloader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    n_epoch=5,
    export_onnx=True,
    path_to_folder="models/test_main/",
    save_frequency=1,
    save_best_weights=True,
    verbose=False,
)


class TestTrain(unittest.TestCase):
    def test_val_metrics(self):

        metrics_after = validate_loop(
            model=model.to(device),
            dataloader=dataloader,
            criterion=criterion,
            device=device,
            verbose=False,
        )

        for metric_name in metrics_after.keys():
            if not metric_name.startswith("loss"):
                self.assertLessEqual(
                    np.mean(metrics_before[metric_name]),
                    np.mean(metrics_after[metric_name]),
                )


if __name__ == "__main__":
    unittest.main()
