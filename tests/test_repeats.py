import numpy as np
from merge_attention import merge, reference


def test_real_with_simple_attn():
    tokens = ['[CLS]', 'for', 'the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']
    words = ['[CLS]', 'For', 'the', '_MATH_-th', 'discrepancy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']
    word_ends = ['[CLS]', 'for', 'the', 'th', 'cy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_reference_with_simple_attn():
    tokens = ['[CLS]', 'for', 'the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']
    words = ['[CLS]', 'For', 'the', '_MATH_-th', 'discrepancy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']
    word_ends = ['[CLS]', 'for', 'the', 'th', 'cy', 'we', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_', 'and', '_MATH_', '.', '[SEP]']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_minimal_reference_with_simple_attn():
    tokens = ['[CLS]', 'for', 'the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    words = ['[CLS]', 'For', 'the', '_MATH_-th', 'discrepancy', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    word_ends = ['[CLS]', 'for', 'the', 'th', 'cy', 'have', '_MATHDISP_', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_smaller_example_reference():
    tokens = ['[CLS]', 'for', 'the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    words = ['[CLS]', 'For', 'the', '_MATH_-th', 'discrepancy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    word_ends = ['[CLS]', 'for', 'the', 'th', 'cy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 1, 1, 3, 4, 1, 1, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_even_smaller_example_reference():
    tokens = ['the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    words = ['the', '_MATH_-th', 'discrepancy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']
    word_ends = ['the', 'th', 'cy', ',', 'where', '_MATH_', ',', '_MATH_', ',', '_MATH_']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 3, 4, 1, 1, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_still_smaller_example_reference():
    tokens = ['the', '_MATH_', '-', 'th', 'disc', 're', 'pan', 'cy', ',', 'where', '_MATH_', ',', '_MATH_']
    words = ['the', '_MATH_-th', 'discrepancy', ',', 'where', '_MATH_', ',', '_MATH_']
    word_ends = ['the', 'th', 'cy', ',', 'where', '_MATH_', ',', '_MATH_']

    saliency = np.ones((len(tokens)), dtype=np.float32)

    attention = [saliency for _ in range(len(tokens))]
    attention = np.array(attention, dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [1, 3, 4, 1, 1, 1, 1, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_the_smallest_so_far_reference():
    tokens = ['_MATH_', 'th', '_MATH_']
    words = ['_MATH_th', '_MATH_']
    word_ends = ['th', '_MATH_']

    attention = np.ones((len(tokens), len(tokens)), dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [2, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)

    for row1, row2 in zip(merged, merged[1:]):
        np.testing.assert_allclose(row1, row2)


def test_minimal_example():
    tokens = ['a', 'b', 'c']
    words = ['ab', 'c']
    word_ends = ['b', 'c']
    attention = np.ones((len(tokens), len(tokens)), dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends)

    expected = np.array([
        [2, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)


def test_minimal_example_with_repeat():
    tokens = ['a', 'b', 'a']
    words = ['ab', 'a']
    word_ends = ['b', 'a']
    attention = np.ones((len(tokens), len(tokens)), dtype=np.float32)

    merged = reference.merge(attention, tokens, words, word_ends, verbosity=2)

    expected = np.array([
        [2, 1] for _ in range(len(words))
    ], dtype=np.float32)

    np.testing.assert_allclose(merged, expected)
