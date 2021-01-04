import numpy as np

from attention import merge


def test_unbalanced():
    tokens = ["straw", "##berries"]
    words = ["strawberries"]
    word_ends = ["##berries"]
    attention = np.array([[0.2, 0.8], [0.2, 0.8]])
    merged = merge(attention, tokens, words, word_ends)
    expected = np.array([[1.0]])
    np.testing.assert_allclose(merged, expected)


def test_simple():
    tokens = ["A", "B"]
    words = ["AB"]
    word_ends = ["B"]
    attention = np.array([[1, 0], [0, 1]], dtype=np.float32)
    merged = merge(attention, tokens, words, word_ends)
    expected = np.array([[1.0]])
    np.testing.assert_allclose(merged, expected)
