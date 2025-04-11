"""Tests for normal integration functions with Numba."""

from functools import cached_property

import numpy as np
import pytest

try:
    from numba import njit

    _have_numba = True
except ImportError:
    _have_numba = False


from typing import Callable

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)


def jit(cls) -> Callable:
    """Wrap cls._func with Numba's jit decorator."""

    @cached_property
    def _jit(self) -> Callable:
        """Wrap cls._func with Numba's jit decorator."""
        return njit(super(cls, self)._func)#, cache=True)

    cls._func = _jit
    cls._func.__set_name__(cls, "_func")

    return cls


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class Test_arnison(_Test_arnison):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class Test_kottler(_Test_kottler):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class Test_frankot(_Test_frankot):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class Test_dct_poisson(_Test_dct_poisson):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class Test_dst_poisson(_Test_dst_poisson):
    xp = np
