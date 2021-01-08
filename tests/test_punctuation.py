import numpy as np

from merge_attention import merge


def test_paren_comma() -> None:
    tokens = ["),"]
    words = [")", ","]
    word_ends = ["),", "),"]
    attention = np.array([[1.0]], dtype=np.float32)
    merged = merge(attention, tokens, words, word_ends)
    expected = np.array([[1, 0], [0, 0]], dtype=np.float32)
    np.testing.assert_allclose(merged, expected, atol=1e-10)


def test_paren_comma_with_other_words() -> None:
    tokens = ["at", "),", "Eq"]
    words = ["at", ")", ",", "Eq"]
    word_ends = ["at", "),", "),", "Eq"]
    attention = np.ones((len(tokens), len(tokens)), dtype=np.float32)
    merged = merge(attention, tokens, words, word_ends)
    assert merged.shape == (4, 4)
    expected = np.array(
        [[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0], [1, 1, 0, 1]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(merged, expected, atol=1e-10)


def test_paren_comma_with_many_words() -> None:
    tokens = ["at", "),", "E", "q"]
    words = ["at", ")", ",", "Eq"]
    word_ends = ["at", "),", "),", "q"]
    attention = np.ones((len(tokens), len(tokens)), dtype=np.float32)
    merged = merge(attention, tokens, words, word_ends)
    assert merged.shape == (4, 4)
    expected = np.array(
        [[1, 1, 0, 2], [1, 1, 0, 2], [0, 0, 0, 0], [1, 1, 0, 2]],
        dtype=np.float32,
    )
    np.testing.assert_allclose(merged, expected, atol=1e-10)

