"""Tests for normal integration functions with numpy."""

import numpy as np
import pytest

try:
    import scipy

    _have_scipy = True
except ImportError:
    _have_scipy = False

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
    _Test_li,
    _Test_southwell,
)


class Test_arnison(_Test_arnison):
    xp = np


class Test_kottler(_Test_kottler):
    xp = np


class Test_frankot(_Test_frankot):
    xp = np


class Test_dct_poisson(_Test_dct_poisson):
    xp = np


class Test_dst_poisson(_Test_dst_poisson):
    xp = np


@pytest.mark.skipif(not _have_scipy, reason="Scipy is not installed")
class Test_li(_Test_li):
    xp = np

    @pytest.mark.xfail(
        reason="too many values to unpack (expected 2)",
        raises=ValueError,
    )
    def test_broadcast(self) -> None:
        super().test_broadcast()


@pytest.mark.skipif(not _have_scipy, reason="Scipy is not installed")
class Test_southwell(_Test_southwell):
    xp = np
