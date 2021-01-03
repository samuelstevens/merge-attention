import logging

from .errors import LoadDllError

try:
    from .optimized import merge
except LoadDllError as err:
    # couldn't load the dll, so we will use the reference implementation.
    logging.warning(err.message)
    logging.warning("Using pure Python implementation.")
    from .reference import merge

__all__ = ["merge"]
