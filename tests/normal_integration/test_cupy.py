"""Test normal integration functions using cupy."""

try:
    import cupy as cp

    _have_cupy = True
except ImportError:
    _have_cupy = False
    cp = None

import pytest

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class Test_arnison(_Test_arnison):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class Test_kottler(_Test_kottler):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class Test_frankot(_Test_frankot):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class Test_dct_poisson(_Test_dct_poisson):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class Test_dst_poisson(_Test_dst_poisson):
    xp = cp
