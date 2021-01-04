import numpy as np

from attention import merge


def test_paren_comma() -> None:
    tokens = ["),"]
    words = [")", ","]
    word_ends = ["),", "),"]
    attention = np.array([[1.0]])
    merged = merge(attention, tokens, words, word_ends)
    assert merged.shape == (2, 2)
    np.testing.assert_allclose(merged[0, 0], 1.0, atol=1e-10)
    np.testing.assert_allclose(merged[0, 1], 0.0, atol=1e-10)
    np.testing.assert_allclose(merged[1, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(merged[1, 1], 0.0, atol=1e-10)
