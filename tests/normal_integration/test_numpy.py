"""_summary_
"""

import numpy as np

from .utils import ArnisonTest, DctPoissonTest, DstPoissonTest, FrankotTest, KottlerTest


class TestArnison(ArnisonTest):
    xp = np


class TestKottler(KottlerTest):
    xp = np


class TestFrankot(FrankotTest):
    xp = np


class TestDctPoisson(DctPoissonTest):
    xp = np


class TestDstPoisson(DstPoissonTest):
    xp = np
