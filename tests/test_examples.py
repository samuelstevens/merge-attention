import numpy as np

from merge_attention import merge, reference


def test_unbalanced():
    tokens = ["straw", "##berries"]
    words = ["strawberries"]
    word_ends = ["##berries"]
    attention = np.array([[0.2, 0.8], [0.2, 0.8]], dtype=np.float32)

    merged = merge(attention, tokens, words, word_ends)
    expected = np.array([[1.0]])
    np.testing.assert_allclose(merged, expected)

    merged = reference.merge(attention, tokens, words, word_ends)
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

    merged = reference.merge(attention, tokens, words, word_ends)
    expected = np.array([[1.0]])
    np.testing.assert_allclose(merged, expected)


def test_3x3():
    tokens = ["A", "B", "C"]
    words = ["A", "B", "C"]
    word_ends = ["A", "B", "C"]
    attention = np.ones((3, 3), dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)
    np.testing.assert_allclose(merged, attention)

    merged = merge(attention, tokens, words, word_ends)
    np.testing.assert_allclose(merged, attention)


def test_near_zero():
    tokens = ["A", "B", "C"]
    words = ["A", "B", "C"]
    word_ends = ["A", "B", "C"]
    attention = np.array(
        [[1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16]],
        dtype=np.float32,
    )

    merged = reference.merge(attention, tokens, words, word_ends)
    np.testing.assert_allclose(merged, attention)

    merged = merge(attention, tokens, words, word_ends)
    np.testing.assert_allclose(merged, attention)
