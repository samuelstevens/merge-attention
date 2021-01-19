import numpy as np
from hypothesis import given

from merge_attention import merge, reference
import util

np.seterr("raise")


@given(util.array_tokens_word_ends())
def test_against_reference(args):
    attn, tokens, word_ends = args
    np.testing.assert_allclose(
        merge(attn, tokens, word_ends, word_ends),
        reference.merge(attn, tokens, word_ends, word_ends),
        atol=1e-16
    )
