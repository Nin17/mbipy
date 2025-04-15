"""Tests for normal integration functions with numpy."""

import numpy as np

try:
    import scipy

    __have_scipy__ = True
except ImportError:
    __have_scipy__ = False

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


if __have_scipy__:

    class Test_li(_Test_li):
        xp = np

    class Test_southwell(_Test_southwell):
        xp = np
