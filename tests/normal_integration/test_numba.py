"""Tests for normal integration functions with Numba."""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

try:
    from numba import errors, njit

    _have_numba = True
except ImportError:
    _have_numba = False


from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)

REASON = "Numba is not installed."


@property
def _func(self) -> Callable:
    return njit(super(self.__class__, self)._func)


def _test_wrong_dtype(self) -> None:
    """Test that the FFT integration functions raise an error for non-real inputs."""
    gy, gx = self.gf32
    gy = self.xp.astype(gy, self.xp.complex64)
    gx = self.xp.astype(gx, self.xp.complex64)
    msg = "Input arrays must be real-valued."
    with pytest.raises(errors.TypingError, match=msg):
        self._func(gy, gx)


@pytest.mark.skipif(not _have_numba, reason=REASON)
class Test_arnison(_Test_arnison):
    xp = np
    _func = _func
    test_wrong_dtype = _test_wrong_dtype


@pytest.mark.skipif(not _have_numba, reason=REASON)
class Test_kottler(_Test_kottler):
    xp = np
    _func = _func
    test_wrong_dtype = _test_wrong_dtype


@pytest.mark.skipif(not _have_numba, reason=REASON)
class Test_frankot(_Test_frankot):
    xp = np
    _func = _func
    test_wrong_dtype = _test_wrong_dtype


@pytest.mark.skipif(not _have_numba, reason=REASON)
class Test_dct_poisson(_Test_dct_poisson):
    xp = np
    _func = _func
    test_wrong_dtype = _test_wrong_dtype


@pytest.mark.skipif(not _have_numba, reason=REASON)
class Test_dst_poisson(_Test_dst_poisson):
    xp = np
    _func = _func
    test_wrong_dtype = _test_wrong_dtype
