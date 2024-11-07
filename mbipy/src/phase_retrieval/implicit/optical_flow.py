"""_summary_
"""

__all__ = ("optical_flow",)

import importlib

from numpy import floating, pi
from numpy.typing import NDArray

from ...config import __have_scipy__
from ...utils import array_namespace


def get_dfts(xp):
    # TODO(nin17): implement this once - same as in integration

    def remove_workers(func):
        return lambda *args, workers, **kwargs: func(*args, **kwargs)

    if "jax" in xp.__name__:
        return remove_workers(xp.fft.fft2), remove_workers(xp.fft.ifft2)

    if "numpy" in xp.__name__:
        if __have_scipy__:
            fft = importlib.import_module("scipy.fft")
            return fft.fft2, fft.ifft2
        return np.fft.fft2, np.fft.ifft2

    if "torch" in xp.__name__:
        return remove_workers(xp.fft.fftn), remove_workers(xp.fft.ifftn)


def _kspace(ny, nx, xp):
    dqy = 2.0 * pi / ny
    dqx = 2.0 * pi / nx
    ky = 2.0 * pi * xp.fft.fftfreq(ny)[:, None]
    kx = 2.0 * pi * xp.fft.fftfreq(nx)[None]
    return ky, kx, dqy, dqx


def _denominator(ky, kx, xp):
    denominator = ky**2 + kx**2
    # avoid division by zero warning
    denominator[..., 0, 0] = 1.0  # TODO(nin17): setitem
    return denominator


def _high_pass_filter(ny, nx, high_pass_sigma, xp):
    ky, kx, dqy, dqx = _kspace(ny, nx, xp)
    denominator = _denominator(ky, kx, xp)

    if high_pass_sigma:
        sigma_x = 2 * (dqx * high_pass_sigma) ** 2
        sigma_y = 2 * (dqy * high_pass_sigma) ** 2
        g = xp.exp(-((kx**2) / sigma_x + (ky**2) / sigma_y))
        beta = 1.0 - g
    else:
        beta = 1.0

    filt_x = beta * kx / denominator
    filt_x[0, 0] = 0.0
    filt_y = beta * ky / denominator
    filt_y[0, 0] = 0.0

    return filt_y, filt_x


def _process_sample(sample, reference, absorption_sigma, filt_y, filt_x, workers):
    xp = array_namespace(sample, reference, filt_y, filt_x)
    fft2, ifft2 = get_dfts(xp)

    if absorption_sigma:
        absorption_mask = (
            gaussian_filter(sample, absorption_sigma, axes=(-2, -1))
            / gaussian_filter(reference, absorption_sigma, axes=(-2, -1))
        ).mean(axis=-3)
    else:
        absorption_mask = xp.array([1.0])  # TODO reshape for ndim - possibly?

    numerator = fft2(sample / absorption_mask - reference, workers=workers)

    # output calculation
    dx = (-ifft2(filt_x * numerator, workers=workers).imag / reference).mean(axis=-3)
    dy = (-ifft2(filt_y * numerator, workers=workers).imag / reference).mean(axis=-3)

    return dy, dx


class OpticalFlow:
    # TODO(nin17): add all changing xp logic etc...
    def __init__(self, reference, high_pass_sigma=0.0, xp=None):
        _xp = xp or array_namespace(reference)
        reference = _xp.asarray(reference)
        filt_y, filt_x = _high_pass_filter(*reference.shape[-2:], high_pass_sigma, _xp)

        self.reference = reference
        self.filters = filt_y, filt_x

    def __call__(self, sample, absorption_sigma=0.0, workers=-1):

        return _process_sample(
            sample, self.reference, absorption_sigma, *self.filters, workers
        )


def optical_flow(
    sample: NDArray[floating],
    reference: NDArray[floating],
    high_pass_sigma: float = 0.0,
    absorption_sigma: float = 0.0,
    workers=-1,
) -> tuple[NDArray[floating], NDArray[floating]]:
    xp = array_namespace(sample, reference)
    filt_y, filt_x = _high_pass_filter(*reference.shape[-2:], high_pass_sigma, xp)

    return _process_sample(sample, reference, absorption_sigma, filt_y, filt_x, workers)
