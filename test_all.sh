#!/usr/bin/env bash
#python3 -m unittest tests/test_prepare_data.py &&
#python3 -m unittest tests/test_dataset.py &&
#python3 -m unittest tests/test_utils.py &&
python3 -m unittest tests/test_dropout.py &&
python3 -m unittest tests/test_normalization.py &&
python3 -m unittest tests/test_rnn.py &&
python3 -m unittest tests/test_attention.py &&
python3 -m unittest tests/test_linear.py &&