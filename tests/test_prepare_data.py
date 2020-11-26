import unittest
from collections import Counter
from pytorch_ner.prepare_data import prepare_conll_data_format, get_token2idx, get_label2idx


token_seq, label_seq = prepare_conll_data_format('tests/data/conll.txt')

token2cnt = Counter([token for sentence in token_seq for token in sentence])
label_set = sorted(set(label for sentence in label_seq for label in sentence))

label2idx = get_label2idx(label_set)


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
                ['O', 'O', 'O', 'O', 'O', 'B-punctuation'],
                ['O', 'O', 'O', 'O', 'O', 'B-punctuation'],
            ]
        )


class TestToken2idx(unittest.TestCase):

    def test_get_token2idx(self):
        token2idx = get_token2idx(token2cnt)
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
        token2idx = get_token2idx(token2cnt, add_pad=False, add_unk=False)
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
        token2idx = get_token2idx(token2cnt, min_count=2)
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

    def test_label2idx_type(self):
        self.assertTrue(isinstance(label2idx, dict))

    def test_label2idx_set(self):
        self.assertTrue('O' in label2idx)
        self.assertTrue('B-punctuation' in label2idx)

    def test_get_label2idx(self):
        self.assertEqual(label2idx, {'B-punctuation': 0, 'O': 1})


if __name__ == '__main__':
    unittest.main()
