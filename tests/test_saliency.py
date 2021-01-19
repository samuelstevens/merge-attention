import numpy as np
from hypothesis import given

from merge_attention import merge
import util


def test_simple():
    tokens = ['A', 'B', 'C']
    words = ['AB', 'C']
    word_ends = ['B', 'C']

    saliency = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = merge(attention, tokens, words, word_ends)

    expected = np.array([[0.3, 0.3], [0.3, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(merged, expected)


def test_not_same():
    tokens = ['A', 'B', 'C']
    words = ['AB', 'C']
    word_ends = ['B', 'C']

    saliency = np.array([0.0, 0.2, 0.3], dtype=np.float32)
    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = merge(attention, tokens, words, word_ends)

    expected = np.array([[0.2, 0.3], [0.2, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(merged, expected)


@given(util.saliency_token_word_ends())
def test_saliency_properties(args):
    saliency, tokens, word_ends = args
    merged = merge(saliency, tokens, word_ends, word_ends),

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)
