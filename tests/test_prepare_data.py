import unittest

# TODO: fix it
import sys
sys.path.append('../pytorch_ner')
from prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx


token_seq, label_seq = prepare_conll_data_format('conll.txt')


class TestPrepareConllDataFormat(unittest.TestCase):

    def test_token_seq(self):
        self.assertEqual(
            token_seq,
            [
                ['Beautiful', 'is', 'better', 'than', 'ugly', '.'],
                ['Explicit', 'is', 'better', 'than', 'implicit', '.'],
            ],
        )

    def test_label_seq(self):
        self.assertEqual(
            label_seq,
            [
                ['O', 'O', 'O', 'O', 'O', 'B-Puctuation'],
                ['O', 'O', 'O', 'O', 'O', 'B-Puctuation'],
            ]
        )


class TestToken2idx(unittest.TestCase):

    def test_get_token2idx(self):
        token2idx = get_token2idx(token_seq)
        self.assertEqual(
            token2idx,
            {
                '<PAD>': 0,
                '<UNK>': 1,
                'Beautiful': 2,
                'is': 3,
                'better': 4,
                'than': 5,
                'ugly': 6,
                '.': 7,
                'Explicit': 8,
                'implicit': 9,
            }
        )

    def test_get_token2idx_special_tokens(self):
        token2idx = get_token2idx(token_seq, add_pad=False, add_unk=False)
        self.assertEqual(
            token2idx,
            {
                'Beautiful': 0,
                'is': 1,
                'better': 2,
                'than': 3,
                'ugly': 4,
                '.': 5,
                'Explicit': 6,
                'implicit': 7,
            }
        )

    def test_get_token2idx_min_count(self):
        token2idx = get_token2idx(token_seq, min_count=2)
        self.assertEqual(
            token2idx,
            {
                '<PAD>': 0,
                '<UNK>': 1,
                'is': 2,
                'better': 3,
                'than': 4,
                '.': 5,
            }
        )


class TestLabel2idx(unittest.TestCase):

    def test_get_label2idx(self):
        label2idx = get_label2idx(label_seq)
        self.assertEqual(
            label2idx,
            {
                'O': 0,
                'B-Puctuation': 1,
            }
        )


if __name__ == '__main__':
    unittest.main()
