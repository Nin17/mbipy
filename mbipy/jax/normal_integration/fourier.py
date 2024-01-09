"""_summary_
"""

__all__ = "frankot_chellappa", "kottler"


from jax import jit
from jax import numpy as np

from ...src.normal_integration import create_antisym as _create_antisym
from ...src.normal_integration import (
    create_frankot_chellappa as _create_frankot_chellappa,
)
from ...src.normal_integration import create_kottler as _create_kottler

_antisym = _create_antisym(np)

_frankot_chellappa = _create_frankot_chellappa(np)
_kottler = _create_kottler(np)

frankot_chellappa = jit(_antisym(_frankot_chellappa))
kottler = jit(_antisym(_kottler))
