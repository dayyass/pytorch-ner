import unittest
from collections import Counter

from pytorch_ner.prepare_data import (
    get_label2idx,
    get_token2idx,
    prepare_conll_data_format,
    process_labels,
    process_tokens,
)

token_seq, label_seq = prepare_conll_data_format(
    path="tests/data/conll.txt", sep=" ", verbose=False
)
token_seq_cased, label_seq_cased = prepare_conll_data_format(
    path="tests/data/conll.txt", sep=" ", lower=False, verbose=False
)

token2cnt = Counter([token for sentence in token_seq for token in sentence])
label_set = sorted(set(label for sentence in label_seq for label in sentence))

token2idx = get_token2idx(token2cnt)
label2idx = get_label2idx(label_set)

tokens = ["simple", "is", "better", "than", "complex", "."]
labels = ["O", "O", "O", "O", "O", "B-punctuation"]


class TestPrepareConllDataFormat(unittest.TestCase):
    def test_token_seq(self):
        self.assertEqual(
            token_seq,
            [
                ["beautiful", "is", "better", "than", "ugly", "."],
                ["explicit", "is", "better", "than", "implicit", "."],
            ],
        )

    def test_token_seq_cased(self):
        self.assertEqual(
            token_seq_cased,
            [
                ["Beautiful", "is", "better", "than", "ugly", "."],
                ["Explicit", "is", "better", "than", "implicit", "."],
            ],
        )

    def test_label_seq(self):
        self.assertEqual(
            label_seq,
            [
                ["O", "O", "O", "O", "O", "B-punctuation"],
                ["O", "O", "O", "O", "O", "B-punctuation"],
            ],
        )


class TestToken2idx(unittest.TestCase):
    def test_get_token2idx(self):
        token2idx = get_token2idx(token2cnt)
        self.assertEqual(
            token2idx,
            {
                "<PAD>": 0,
                "<UNK>": 1,
                "beautiful": 2,
                "is": 3,
                "better": 4,
                "than": 5,
                "ugly": 6,
                ".": 7,
                "explicit": 8,
                "implicit": 9,
            },
        )

    def test_get_token2idx_special_tokens(self):
        token2idx = get_token2idx(token2cnt, add_pad=False, add_unk=False)
        self.assertEqual(
            token2idx,
            {
                "beautiful": 0,
                "is": 1,
                "better": 2,
                "than": 3,
                "ugly": 4,
                ".": 5,
                "explicit": 6,
                "implicit": 7,
            },
        )

    def test_get_token2idx_min_count(self):
        token2idx = get_token2idx(token2cnt, min_count=2)
        self.assertEqual(
            token2idx,
            {
                "<PAD>": 0,
                "<UNK>": 1,
                "is": 2,
                "better": 3,
                "than": 4,
                ".": 5,
            },
        )


class TestLabel2idx(unittest.TestCase):
    def test_label2idx_type(self):
        self.assertTrue(isinstance(label2idx, dict))

    def test_label2idx_set(self):
        self.assertTrue("O" in label2idx)
        self.assertTrue("B-punctuation" in label2idx)

    def test_get_label2idx(self):
        self.assertEqual(label2idx, {"B-punctuation": 0, "O": 1})


class TestProcess(unittest.TestCase):
    def test_process_tokens(self):
        self.assertEqual(
            process_tokens(tokens, token2idx),
            [1, 3, 4, 5, 1, 7],
        )

    def test_process_labels(self):
        self.assertEqual(
            process_labels(labels, label2idx),
            [1, 1, 1, 1, 1, 0],
        )


if __name__ == "__main__":
    unittest.main()
