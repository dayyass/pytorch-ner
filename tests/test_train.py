import unittest
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx
from pytorch_ner.dataset import NERDataset, NERCollator
from pytorch_ner.train import train, validate_loop

from pytorch_ner.nn_modules.dropout import SpatialDropout1d
from pytorch_ner.nn_modules.normalization import LayerNorm
from pytorch_ner.nn_modules.embedding import Embedding, EmbeddingWithDropout
from pytorch_ner.nn_modules.rnn import DynamicRNN
from pytorch_ner.nn_modules.attention import MultiheadSelfAttention, AttentionWithSkipConnectionLayerNorm
from pytorch_ner.nn_modules.linear import LinearHead
from pytorch_ner.nn_modules.architecture import BiLSTM


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

embedding_layer = EmbeddingWithDropout(
    embedding_layer=Embedding(num_embeddings=len(token2idx), embedding_dim=128),
    dropout=SpatialDropout1d(p=0.)
)
rnn_layer = DynamicRNN(rnn_unit=nn.LSTM, input_size=128, hidden_size=256, num_layers=1, dropout=0., bidirectional=True)
attention_layer = AttentionWithSkipConnectionLayerNorm(
    attention_layer=MultiheadSelfAttention(embed_dim=512, num_heads=8, dropout=0.),
    layer_norm=LayerNorm(normalized_shape=512),
    use_skip_connection=True,
)
linear_head = LinearHead(linear_head=nn.Linear(in_features=512, out_features=2))

model = BiLSTM(
    embedding_layer=embedding_layer,
    rnn_layer=rnn_layer,
    linear_head=linear_head,
).to(device)


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

    def test_val_metrics(self):

        val_metrics = validate_loop(
            model=model,
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
