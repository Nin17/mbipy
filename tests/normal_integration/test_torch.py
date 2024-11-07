"""_summary_
"""

import pytest

try:
    import torch
    _have_torch = True
except ImportError:
    _have_torch = False
    torch = None

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class TestArnison(ArnisonTest):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class TestKottler(KottlerTest):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class TestFrankot(FrankotTest):
    xp = torch


# @pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
@pytest.mark.xfail(raises=NotImplementedError)
class TestDctPoisson(DctPoissonTest):
    xp = torch


@pytest.mark.skip(reason="DST Type 1 only works with scipy")
class TestDstPoisson(DstPoissonTest):
    xp = torch
