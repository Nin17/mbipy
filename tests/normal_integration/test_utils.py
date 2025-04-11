"""_summary_"""

from numpy import random, testing
from scipy.fft import dstn, idstn

from mbipy.src.normal_integration.fourier.utils import dst2, idst2

x = random.random((2, 3, 11, 13))


def test_dstn():
    testing.assert_allclose(dst2(x), dstn(x, type=1, axes=(-2, -1)))


def test_idstn():
    testing.assert_allclose(idst2(x), idstn(x, type=1, axes=(-2, -1)))
