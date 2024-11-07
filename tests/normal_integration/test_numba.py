import numpy as np
import pytest

try:
    from numba import njit

    _have_numba = True
except ImportError:
    _have_numba = False

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest

# def xfail(*funcs):
#     def decorator(cls):
        
#         @property
#         def _func(self):
#             return njit(
#                 super(cls, self).func,
#             )
        
#         for i in funcs:
#             _func = getattr(cls, i)
#             setattr(cls, i, pytest.mark.xfail(raises=AssertionError)(_func))
#         return cls
#     return decorator


def jit(cls):
    @property
    def _func(self):
        return njit(
            super(cls, self).func,
        )

    cls.func = _func

    return cls


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class TestArnison(ArnisonTest):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class TestKottler(KottlerTest):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class TestFrankot(FrankotTest):
    xp = np


@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class TestDctPoisson(DctPoissonTest):
    xp = np

# @xfail("test_f32")
@pytest.mark.skipif(not _have_numba, reason="Numba is not installed")
@jit
class TestDstPoisson(DstPoissonTest):
    xp = np
