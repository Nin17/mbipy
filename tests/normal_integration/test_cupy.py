"""_summary_
"""

try:
    import cupy as cp

    _have_cupy = True
except ImportError:
    _have_cupy = False
    cp = None

import pytest

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class TestArnison(ArnisonTest):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class TestKottler(KottlerTest):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class TestFrankot(FrankotTest):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class TestDctPoisson(DctPoissonTest):
    xp = cp


@pytest.mark.skipif(not _have_cupy, reason="Cupy is not installed")
class TestDstPoisson(DstPoissonTest):
    xp = cp
