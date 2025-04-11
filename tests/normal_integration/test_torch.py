"""Tests for normal integration using PyTorch."""

import pytest

try:
    import torch

    _have_torch = True
except ImportError:
    _have_torch = False
    torch = None

from .utils import (
    _Test_arnison,
    _Test_dct_poisson,
    _Test_dst_poisson,
    _Test_frankot,
    _Test_kottler,
)


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class Test_arnison(_Test_arnison):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class Test_kottler(_Test_kottler):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class Test_frankot(_Test_frankot):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class Test_dct_poisson(_Test_dct_poisson):
    xp = torch


@pytest.mark.skipif(not _have_torch, reason="PyTorch is not installed")
class Test_dst_poisson(_Test_dst_poisson):
    xp = torch
