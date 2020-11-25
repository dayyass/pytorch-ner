import unittest
import torch
import numpy as np
from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx
from pytorch_ner.dataset import NERDataset, NERCollator


token_seq, label_seq = prepare_conll_data_format('conll.txt')
token2idx = get_token2idx(token_seq)
label2idx = get_label2idx(label_seq)

dataset_preprocessed = NERDataset(
    token_seq=token_seq, label_seq=label_seq, token2idx=token2idx, label2idx=label2idx,
)
dataset = NERDataset(
    token_seq=token_seq, label_seq=label_seq, token2idx=token2idx, label2idx=label2idx, preprocess=False,
)

ref_tokens_0 = np.array([2, 3, 4, 5, 6, 7])
ref_labels_0 = np.array([0, 0, 0, 0, 0, 1])
ref_lengths_0 = np.array(6)

ref_tokens_1 = np.array([8, 3, 4, 5, 9, 7])
ref_labels_1 = np.array([0, 0, 0, 0, 0, 1])
ref_lengths_1 = np.array(6)


collator_1 = NERCollator(token_padding_value=0, label_padding_value=0, percentile=100)
collator_2 = NERCollator(token_padding_value=1, label_padding_value=1, percentile=50)

batch = [
    (np.array([1, 2, 3]), np.array([0, 0, 0]), np.array([3])),
    (np.array([1, 2, 3, 4, 5]), np.array([0, 0, 0, 0, 0]), np.array([5])),
]


class TestDataset(unittest.TestCase):

    def test_len(self):
        self.assertEqual(len(dataset_preprocessed), 2)
        self.assertEqual(len(dataset), 2)

    def test_preprocessed_get_item_0(self):
        tokens, labels, lengths = dataset_preprocessed[0]
        self.assertTrue(np.all(tokens == ref_tokens_0))
        self.assertTrue(np.all(labels == ref_labels_0))
        self.assertTrue(np.all(lengths == ref_lengths_0))

    def test_get_item_0(self):
        tokens, labels, lengths = dataset[0]
        self.assertTrue(np.all(tokens == ref_tokens_0))
        self.assertTrue(np.all(labels == ref_labels_0))
        self.assertTrue(np.all(lengths == ref_lengths_0))

    def test_preprocessed_get_item_1(self):
        tokens, labels, lengths = dataset_preprocessed[1]
        self.assertTrue(np.all(tokens == ref_tokens_1))
        self.assertTrue(np.all(labels == ref_labels_1))
        self.assertTrue(np.all(lengths == ref_lengths_1))

    def test_get_item_1(self):
        tokens, labels, lengths = dataset[1]
        self.assertTrue(np.all(tokens == ref_tokens_1))
        self.assertTrue(np.all(labels == ref_labels_1))
        self.assertTrue(np.all(lengths == ref_lengths_1))


class TestCollator(unittest.TestCase):

    def test_collator_1(self):
        tokens, labels, lengths = collator_1(batch)
        self.assertTrue(torch.equal(tokens, torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 5]])))
        self.assertTrue(torch.equal(labels, torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])))
        self.assertTrue(torch.equal(lengths, torch.tensor([3, 5])))

    def test_collator_2(self):
        tokens, labels, lengths = collator_2(batch)
        self.assertTrue(torch.equal(tokens, torch.tensor([[1, 2, 3, 1], [1, 2, 3, 4]])))
        self.assertTrue(torch.equal(labels, torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0]])))
        self.assertTrue(torch.equal(lengths, torch.tensor([3, 4])))


if __name__ == '__main__':
    unittest.main()