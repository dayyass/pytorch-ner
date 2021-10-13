import os
from collections import Counter

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from pytorch_ner.dataset import NERCollator, NERDataset
from pytorch_ner.nn_modules.architecture import BiLSTM
from pytorch_ner.nn_modules.embedding import Embedding
from pytorch_ner.nn_modules.linear import LinearHead
from pytorch_ner.nn_modules.rnn import DynamicRNN
from pytorch_ner.prepare_data import (
    get_label2idx,
    get_token2idx,
    prepare_conll_data_format,
)
from pytorch_ner.save import save_model
from pytorch_ner.train import train_loop
from pytorch_ner.utils import set_global_seed, str_to_class


def _train(path_to_config: str):
    """Main function to train NER model."""

    with open(path_to_config, mode="r") as fp:
        config = yaml.safe_load(fp)

    # check existence of save path_to_folder
    if os.path.exists(config["save"]["path_to_folder"]):
        raise FileExistsError("save directory already exists")

    device = torch.device(config["torch"]["device"])
    set_global_seed(config["torch"]["seed"])

    # LOAD DATA

    # tokens / labels sequences

    train_token_seq, train_label_seq = prepare_conll_data_format(
        path=config["prepare_data"]["train_data"]["path"],
        sep=config["prepare_data"]["train_data"]["sep"],
        lower=config["prepare_data"]["train_data"]["lower"],
        verbose=config["prepare_data"]["train_data"]["verbose"],
    )

    val_token_seq, val_label_seq = prepare_conll_data_format(
        path=config["prepare_data"]["val_data"]["path"],
        sep=config["prepare_data"]["val_data"]["sep"],
        lower=config["prepare_data"]["val_data"]["lower"],
        verbose=config["prepare_data"]["val_data"]["verbose"],
    )

    if "test_data" in config["prepare_data"]:
        test_token_seq, test_label_seq = prepare_conll_data_format(
            path=config["prepare_data"]["test_data"]["path"],
            sep=config["prepare_data"]["test_data"]["sep"],
            lower=config["prepare_data"]["test_data"]["lower"],
            verbose=config["prepare_data"]["test_data"]["verbose"],
        )

    # token2idx / label2idx

    token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
    label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

    token2idx = get_token2idx(
        token2cnt=token2cnt,
        min_count=config["prepare_data"]["token2idx"]["min_count"],
        add_pad=config["prepare_data"]["token2idx"]["add_pad"],
        add_unk=config["prepare_data"]["token2idx"]["add_unk"],
    )

    label2idx = get_label2idx(label_set=label_set)

    # datasets

    trainset = NERDataset(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    valset = NERDataset(
        token_seq=val_token_seq,
        label_seq=val_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    if "test_data" in config["prepare_data"]:
        testset = NERDataset(
            token_seq=test_token_seq,
            label_seq=test_label_seq,
            token2idx=token2idx,
            label2idx=label2idx,
            preprocess=config["dataloader"]["preprocess"],
        )

    # collators

    train_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=config["dataloader"]["percentile"],
    )

    val_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=100,  # hardcoded
    )

    if "test_data" in config["prepare_data"]:
        test_collator = NERCollator(
            token_padding_value=token2idx[config["dataloader"]["token_padding"]],
            label_padding_value=label2idx[config["dataloader"]["label_padding"]],
            percentile=100,  # hardcoded
        )

    # dataloaders

    # TODO: add more params to config.yaml
    trainloader = DataLoader(
        dataset=trainset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,  # hardcoded
        collate_fn=train_collator,
    )

    valloader = DataLoader(
        dataset=valset,
        batch_size=1,  # hardcoded
        shuffle=False,  # hardcoded
        collate_fn=val_collator,
    )

    if "test_data" in config["prepare_data"]:
        testloader = DataLoader(
            dataset=testset,
            batch_size=1,  # hardcoded
            shuffle=False,  # hardcoded
            collate_fn=test_collator,
        )

    # INIT MODEL

    # TODO: add more params to config.yaml
    # TODO: add pretrained embeddings
    # TODO: add dropout
    embedding_layer = Embedding(
        num_embeddings=len(token2idx),
        embedding_dim=config["model"]["embedding"]["embedding_dim"],
    )

    rnn_layer = DynamicRNN(
        rnn_unit=eval(config["model"]["rnn"]["rnn_unit"]),  # TODO: fix eval
        input_size=config["model"]["embedding"][
            "embedding_dim"
        ],  # reference to embedding_dim
        hidden_size=config["model"]["rnn"]["hidden_size"],
        num_layers=config["model"]["rnn"]["num_layers"],
        dropout=config["model"]["rnn"]["dropout"],
        bidirectional=config["model"]["rnn"]["bidirectional"],
    )

    # TODO: add attention if needed in config
    linear_head = LinearHead(
        linear_head=nn.Linear(
            in_features=(
                (2 if config["model"]["rnn"]["bidirectional"] else 1)
                * config["model"]["rnn"]["hidden_size"]
            ),
            out_features=len(label2idx),
        ),
    )

    # TODO: add model architecture in config
    # TODO: add attention if needed
    model = BiLSTM(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        linear_head=linear_head,
    ).to(device)

    # CRITERION AND OPTIMIZER

    criterion = nn.CrossEntropyLoss(reduction="none")  # hardcoded

    optimizer_type = str_to_class(
        module_name="torch.optim",
        class_name=config["optimizer"]["optimizer_type"],
    )
    optimizer = optimizer_type(
        params=model.parameters(),
        **config["optimizer"]["params"],
    )

    # TRAIN MODEL

    train_loop(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader if "test_data" in config["prepare_data"] else None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=config["optimizer"]["clip_grad_norm"],
        n_epoch=config["train"]["n_epoch"],
        verbose=config["train"]["verbose"],
    )

    # SAVE MODEL

    save_model(
        path_to_folder=config["save"]["path_to_folder"],
        model=model,
        token2idx=token2idx,
        label2idx=label2idx,
        config=config,
        export_onnx=config["save"]["export_onnx"],
    )