import random
from typing import List

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given

from attention import optimized, reference

np.seterr("raise")


def make_word_ends_from_tokens(tokens: List[str]) -> List[str]:
    """
    pick word ends randomly, with 0.9 probability
    """
    return [tok for tok in tokens if random.random() < 0.9]


@st.composite
def array_tokens_word_ends(draw):
    size = draw(st.integers(min_value=1, max_value=12))
    arr = draw(hnp.arrays(np.float32, (size, size), elements=st.floats(0, 1, width=32)))
    tokens = draw(st.lists(st.text(), min_size=size, max_size=size, unique=True))
    word_ends = make_word_ends_from_tokens(tokens)
    return arr, tokens, word_ends


@given(array_tokens_word_ends())
def test_against_reference(args):
    attn, tokens, word_ends = args
    np.testing.assert_allclose(
        reference.merge(attn, tokens, word_ends, word_ends),
        optimized.merge(attn, tokens, word_ends, word_ends),
    )
