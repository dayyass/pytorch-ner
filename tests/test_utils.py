import unittest
from collections import Counter
from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx
from pytorch_ner.utils import process_tokens, process_labels


token_seq, label_seq = prepare_conll_data_format('tests/data/conll.txt')

token2cnt = Counter([token for sentence in token_seq for token in sentence])
label_set = sorted(set(label for sentence in label_seq for label in sentence))

token2idx = get_token2idx(token2cnt)
label2idx = get_label2idx(label_set)

tokens = ['simple', 'is', 'better', 'than', 'complex', '.']
labels = ['O', 'O', 'O', 'O', 'O', 'B-punctuation']


class TestUtils(unittest.TestCase):

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


if __name__ == '__main__':
    unittest.main()
