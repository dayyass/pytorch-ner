import logging
from collections import Counter
from typing import Any, Dict

import torch
import torch.nn as nn
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


def _train(
    config: Dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Main function to train NER model."""

    # log config
    with open(config["save"]["path_to_config"], mode="r") as fp:
        logger.info(f"Config:\n\n{fp.read()}")

    device = torch.device(config["torch"]["device"])
    set_global_seed(config["torch"]["seed"])

    # LOAD DATA

    # tokens / labels sequences

    train_token_seq, train_label_seq = prepare_conll_data_format(
        path=config["data"]["train_data"]["path"],
        sep=config["data"]["train_data"]["sep"],
        lower=config["data"]["train_data"]["lower"],
        verbose=config["data"]["train_data"]["verbose"],
    )

    valid_token_seq, valid_label_seq = prepare_conll_data_format(
        path=config["data"]["valid_data"]["path"],
        sep=config["data"]["valid_data"]["sep"],
        lower=config["data"]["valid_data"]["lower"],
        verbose=config["data"]["valid_data"]["verbose"],
    )

    if "test_data" in config["data"]:
        test_token_seq, test_label_seq = prepare_conll_data_format(
            path=config["data"]["test_data"]["path"],
            sep=config["data"]["test_data"]["sep"],
            lower=config["data"]["test_data"]["lower"],
            verbose=config["data"]["test_data"]["verbose"],
        )

    # token2idx / label2idx

    token2cnt = Counter([token for sentence in train_token_seq for token in sentence])
    label_set = sorted(set(label for sentence in train_label_seq for label in sentence))

    token2idx = get_token2idx(
        token2cnt=token2cnt,
        min_count=config["data"]["token2idx"]["min_count"],
        add_pad=config["data"]["token2idx"]["add_pad"],
        add_unk=config["data"]["token2idx"]["add_unk"],
    )

    label2idx = get_label2idx(label_set=label_set)

    # datasets

    train_set = NERDataset(
        token_seq=train_token_seq,
        label_seq=train_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    valid_set = NERDataset(
        token_seq=valid_token_seq,
        label_seq=valid_label_seq,
        token2idx=token2idx,
        label2idx=label2idx,
        preprocess=config["dataloader"]["preprocess"],
    )

    if "test_data" in config["data"]:
        test_set = NERDataset(
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

    valid_collator = NERCollator(
        token_padding_value=token2idx[config["dataloader"]["token_padding"]],
        label_padding_value=label2idx[config["dataloader"]["label_padding"]],
        percentile=100,  # hardcoded
    )

    if "test_data" in config["data"]:
        test_collator = NERCollator(
            token_padding_value=token2idx[config["dataloader"]["token_padding"]],
            label_padding_value=label2idx[config["dataloader"]["label_padding"]],
            percentile=100,  # hardcoded
        )

    # dataloaders

    # TODO: add more params to config.yaml
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=True,  # hardcoded
        collate_fn=train_collator,
    )

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=1,  # hardcoded
        shuffle=False,  # hardcoded
        collate_fn=valid_collator,
    )

    if "test_data" in config["data"]:
        test_loader = DataLoader(
            dataset=test_set,
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
        rnn_unit=str_to_class(
            module_name="torch.nn",
            class_name=config["model"]["rnn"]["rnn_unit"],
        ),
        input_size=config["model"]["embedding"]["embedding_dim"],  # ref to emb_dim
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
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader if "test_data" in config["data"] else None,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        clip_grad_norm=config["optimizer"]["clip_grad_norm"],
        n_epoch=config["train"]["n_epoch"],
        verbose=config["train"]["verbose"],
        logger=logger,
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
