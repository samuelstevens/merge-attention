import random
import numpy as np
from typing import List

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st


def make_word_ends_from_tokens(tokens: List[str]) -> List[str]:
    """
    pick word ends randomly, with 0.9 probability
    """
    return [tok for i, tok in enumerate(tokens) if i % 5 != 0]


@st.composite
def array_tokens_word_ends(draw):
    size = draw(st.integers(min_value=1, max_value=16))
    arr = draw(hnp.arrays(np.float32, (size, size), elements=st.floats(0, 1, width=32)))
    tokens = draw(st.lists(st.text(), min_size=size, max_size=size, unique=True))
    word_ends = make_word_ends_from_tokens(tokens)
    return arr, tokens, word_ends


@st.composite
def saliency_token_word_ends(draw):
    size = draw(st.integers(min_value=1, max_value=16))
    arr = draw(hnp.arrays(np.float32, (size), elements=st.floats(0, 1, width=32)))
    arr = np.array([arr for _ in range(size)], dtype=np.float32)
    tokens = draw(st.lists(st.text(), min_size=size, max_size=size, unique=True))
    word_ends = make_word_ends_from_tokens(tokens)
    return arr, tokens, word_ends

