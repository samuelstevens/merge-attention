import numpy as np

from merge_attention.reference import merge


def test_simple():
    tokens = ["AB"]
    words = ["A", "B"]
    word_ends = ["AB", "AB"]
    attention = np.array([[1]], dtype=np.float32)
    merged = merge(attention, tokens, words, word_ends)
    expected = np.array([[1, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(merged, expected)
