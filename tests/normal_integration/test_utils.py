"""_summary_"""

from numpy import random, testing
from scipy.fft import dstn, idstn

from mbipy.src.normal_integration.fourier.utils import dst1_2d, idst1_2d

x = random.random((2, 3, 11, 13))


def test_dstn() -> None:
    """Test the discrete sine transform of type 1 vs SciPy."""
    testing.assert_allclose(dst1_2d(x), dstn(x, type=1, axes=(-2, -1)))


def test_idstn() -> None:
    """Test the inverse discrete sine transform of type 1 vs SciPy."""
    testing.assert_allclose(idst1_2d(x), idstn(x, type=1, axes=(-2, -1)))

