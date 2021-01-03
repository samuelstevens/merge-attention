import ctypes
import pathlib
from typing import List

import numpy as np
from .errors import LoadDllError

try:
    numpy_ctypeslib_flags = ["C_CONTIGUOUS", "ALIGNED"]

    here = pathlib.Path(__file__).parent.resolve()

    attention_dll = np.ctypeslib.load_library("attentionmodule.so", here)
    attention_dll.merge.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=numpy_ctypeslib_flags),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags=numpy_ctypeslib_flags),
        ctypes.POINTER(ctypes.c_char_p),  # tokenv
        ctypes.c_int32,  # tokenc
        ctypes.POINTER(ctypes.c_char_p),  # wordendsv
        ctypes.c_int32,  # wordendsc
        ctypes.c_int8,  # verbosity
    ]
    attention_dll.merge.restype = None
except OSError:
    raise LoadDllError("attentionmodule.so", here)


def strlist_to_char_p_p(lst: List[str]) -> ctypes.POINTER(ctypes.c_char_p):
    """
    Converts a list of strings to a char**
    """
    arr = (ctypes.c_char_p * len(lst))()
    arr[:] = [bytes(s, "utf-8") for s in lst]
    return arr


def merge(
    attention_in: np.ndarray,
    tokens: List[str],
    words: List[str],
    word_ends: List[str],
    verbosity: int = 0,
) -> np.ndarray:
    attention_in = attention_in.astype(np.float32)
    attention_out = np.zeros((len(words), len(words)), dtype=np.float32)
    assert len(words) == len(word_ends)

    attention_dll.merge(
        attention_in,
        attention_out,
        strlist_to_char_p_p(tokens),
        len(tokens),
        strlist_to_char_p_p(word_ends),
        len(word_ends),
        verbosity,
    )

    return attention_out


if __name__ == "__main__":
    tokens = ["[CLS]", "time", "-", "v", "ary", "ing", "[SEP]"]
    words = ["[CLS]", "time-varying", "[SEP]"]
    word_ends = ["[CLS]", "ing", "[SEP]"]
    attention = np.ones((len(tokens), len(tokens)))
    print(merge(attention, tokens, words, word_ends))

    tokens = ["[CLS]", "time", "-", "v", "ary", "ing", "[SEP]"]
    words = ["[CLS]", "time-varying", "[SEP]"]
    word_ends = ["[CLS]", "ing", "[SEP]"]
    attention = np.ones((len(tokens), len(tokens)))
    print(merge(attention, tokens, words, word_ends))

    tokens = ["red", "straw", "##berries"]
    words = ["red", "strawberries"]
    word_ends = ["red", "##berries"]
    attention = np.array([[1.0, 0, 0], [0, 0.2, 0.8], [0, 0.8, 0.2]])
    print(merge(attention, tokens, words, word_ends))

    tokens = ["straw", "##berries"]
    words = ["strawberries"]
    word_ends = ["##berries"]
    attention = np.array([[0.2, 0.8], [0.2, 0.8]])
    print(merge(attention, tokens, words, word_ends))
