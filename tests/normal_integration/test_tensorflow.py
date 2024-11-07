"""_summary_
"""

from functools import wraps

import tensorflow.experimental.numpy as tnp

tnp.experimental_enable_numpy_behavior()
import pytest

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest


@pytest.mark.xfail(raises=AttributeError)
class TestArnison(ArnisonTest):
    xp = tnp


@pytest.mark.xfail(raises=AttributeError)
class TestKottler(KottlerTest):
    xp = tnp


@pytest.mark.xfail(raises=AttributeError)
class TestFrankot(FrankotTest):
    xp = tnp


@pytest.mark.xfail(raises=AttributeError)
class TestDctPoisson(DctPoissonTest):
    xp = tnp


@pytest.mark.xfail(raises=AttributeError)
class TestDstPoisson(DstPoissonTest):
    xp = tnp
