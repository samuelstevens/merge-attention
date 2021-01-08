import logging
from typing import List, Sequence, Tuple

import numpy as np


def split_precondition(
    tokens: Sequence[str], words: Sequence[str], word_ends: Sequence[str]
) -> bool:
    """
    In order to split, there must be a token that is two words. That means there is at least one duplicated word_end that is not a word.
    """
    duplicated_word_ends = []
    for end1, end2 in zip(word_ends, word_ends[1:]):
        if end1 == end2:
            duplicated_word_ends.append(end1)

    if not duplicated_word_ends:
        return False

    duplicate_not_word = False
    for duplicate in duplicated_word_ends:
        if duplicate not in words:
            duplicate_not_word = True
            break

    if not duplicate_not_word:
        return False

    return True


def split(
    attention_in: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    word_ends: Sequence[str],
    verbosity: int = 0,
) -> Tuple[np.ndarray, List[str], List[str], List[str]]:

    assert split_precondition(tokens, words, word_ends), "Split precondition not met."

    # do nothing for now
    logging.info(
        "There is at least one token in this sentence that needs to be split. The code is too hard to write for now, so this does nothing."
    )
    return attention_in, tokens, words, word_ends


def merge(
    attention_in: np.ndarray,
    tokens: Sequence[str],
    words: Sequence[str],
    word_ends: Sequence[str],
    verbosity: int = 0,
) -> np.ndarray:
    assert attention_in.shape == (len(tokens), len(tokens))

    if split_precondition(tokens, words, word_ends):
        attention_in, tokens, words, word_ends = split(
            attention_in, tokens, words, word_ends, verbosity
        )

    if verbosity == 1:
        print(attention_in.shape)
        print()
    if verbosity == 2:
        print(attention_in)
        print()

    # step 1: merge attention *to* split words

    merged_attention = np.zeros((len(tokens), len(words)), dtype=np.float32)

    for token_i, token_from in enumerate(tokens):
        attention_sum = 0
        word_j = -1
        for token_j, token_to in enumerate(tokens):
            attention_sum += attention_in[token_i, token_j]
            if token_to in word_ends[word_j + 1 :]:
                word_j = word_ends.index(token_to, word_j + 1)
                merged_attention[token_i, word_j] = attention_sum
                attention_sum = 0

    if verbosity == 1:
        print(merged_attention.shape)
        print()
    if verbosity == 2:
        print(merged_attention)
        print()

    final_attention = np.zeros((len(words), len(words)), dtype=np.float32)

    # step 2: merge attention *from* split words

    for word_j, word in enumerate(words):
        word_i = -1
        attention_to_word = 0
        tokens_to_word_count = 0
        for token_i, token in enumerate(tokens):
            attention_to_word += merged_attention[token_i, word_j]
            tokens_to_word_count += 1

            if token in word_ends[word_i + 1 :]:
                word_i = word_ends.index(token, word_i + 1)
                attention_from_word = attention_to_word / tokens_to_word_count
                final_attention[word_i, word_j] = attention_from_word
                attention_to_word = 0
                tokens_to_word_count = 0

    if verbosity == 1:
        print(final_attention.shape)
        print()
    if verbosity == 2:
        print(final_attention)
        print()

    return final_attention


if __name__ == "__main__":
    tokens = ["AB"]
    words = ["A", "B"]
    word_ends = ["AB", "AB"]
    attention = np.array([[1]], dtype=np.float32)
    print(merge(attention, tokens, words, word_ends))

    tokens = ["[CLS]", "time", "-", "v", "ary", "ing", "[SEP]"]
    words = ["[CLS]", "time-varying", "[SEP]"]
    word_ends = ["[CLS]", "ing", "[SEP]"]
    attention = np.ones((len(tokens), len(tokens)))
    print(merge(attention, tokens, words, word_ends, verbosity=2))

    tokens = ["straw", "##berries"]
    words = ["strawberries"]
    word_ends = ["##berries"]
    attention = np.array([[0.2, 0.8], [0.8, 0.2]])
    print(merge(attention, tokens, words, word_ends, verbosity=0))

    tokens = ["straw", "##berries"]
    words = ["strawberries"]
    word_ends = ["##berries"]
    attention = np.array([[0.2, 0.8], [0.2, 0.8]])
    print(merge(attention, tokens, words, word_ends, verbosity=0))

    tokens = ["and", "and"]
    words = ["and", "and"]
    word_ends = ["and", "and"]
    attention = np.array([[0.9, 0.1], [0.1, 0.9]])
    print(merge(attention, tokens, words, word_ends, verbosity=0))

    tokens = ["A", "B", "C"]
    words = ["A", "B", "C"]
    word_ends = ["A", "B", "C"]
    attention = np.array(
        [[1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16]],
        dtype=np.float32,
    )
    print(merge(attention, tokens, words, word_ends, verbosity=0))

