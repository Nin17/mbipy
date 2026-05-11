"""Normal integration using the method of Frankot and Chellappa.

[Frankot, R. T. & Chellappa, R. A method for enforcing integrability in shape
from shading algorithms.
IEEE Transactions on pattern analysis and machine intelligence 10, 439-451 (1988)](https://doi.org/10.1109/34.3909)
"""

from __future__ import annotations

__all__ = ["frankot"]


from typing import TYPE_CHECKING, Literal

from mbipy.src.utils import array_namespace, astype, at, get_dtypes

from ._padding import antisymmetric
from ._utils import FFTMethod, fft_2d, ifft_2d, irfft_2d, rfft_2d

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

# TODO(nin17): dtype kwarg in np.fft.fftfreq/rfftfreq wait for numpy & rocketfft
# TODO(nin17): dtype promotion in numba


def frankot(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: Literal["antisymmetric"] | None = None,
    workers: int | None = None,
    fft_method: FFTMethod = FFTMethod.FFT,
) -> NDArray[floating]:
    """Perform normal integration using the method of Frankot and Chellappa[^1].

    ??? info "Array API Compatibility"

        {{ NormalIntegration.row("frankot") | indent(4) }}

    !!! example "[Example][frankot-example]"

    [^1]:[Frankot, R. T. & Chellappa, R. A method for enforcing integrability in shape
    from shading algorithms.
    IEEE Transactions on pattern analysis and machine intelligence 10, 439-451 (1988)](https://doi.org/10.1109/34.3909)

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).
    pad : Literal["antisymmetric"] | None, optional
        Type of padding to apply:
        ["antisymmetric"][mbipy.normal_integration.padding.antisymmetric] | None,
        by default `None`
    workers : int | None, optional
        Passed to [scipy.fft.fft2][]/[scipy.fft.rfft2][] &
        [scipy.fft.ifft2][]/[scipy.fft.irfft2][] if `gy` & `gx` are numpy
        arrays and
        [config.use_scipy_fft][mbipy.src.config.Config.use_scipy_fft] = True,
        by default `None`
    fft_method : FFTMethod, optional
        FFT method to use, by default FFTMethod.FFT

    Returns
    -------
    NDArray[floating] (..., M, N)
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.
    ValueError
        If the pad argument is not None or "antisymmetric".

    """
    xp = array_namespace(gy, gx)
    dtype, cdtype = get_dtypes(gy, gx)
    y, x = xp.broadcast_shapes(gx.shape, gy.shape)[-2:]
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x

    match pad:
        case "antisymmetric":
            gy, gx = antisymmetric(gy=gy, gx=gx)
        case None:
            ...
        case _:
            msg = f"Invalid value for pad: {pad}"
            raise ValueError(msg)

    match fft_method:
        case FFTMethod.FFT:
            fx = xp.fft.fftfreq(x2)  # FIXME: dtype
            gx_fft = fft_2d(astype(gx, cdtype, copy=True), workers=workers)
            gy_fft = fft_2d(astype(gy, cdtype, copy=True), workers=workers)
        case FFTMethod.RFFT:
            fx = xp.fft.rfftfreq(x2)  # FIXME: dtype
            s = (y2, x2)
            gx_fft = rfft_2d(gx, s=s, workers=workers)
            gy_fft = rfft_2d(gy, s=s, workers=workers)
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)

    fx = astype(fx, dtype)  # FIXME: dtype
    fy = astype(xp.fft.fftfreq(y2)[:, None], dtype)  # FIXME: dtype

    gx_fft = at(gx_fft)[:].multiply(fx)
    gy_fft = at(gy_fft)[:].multiply(fy)
    f_num = gx_fft + gy_fft

    fx = at(fx)[:].multiply(fx)
    fy = at(fy)[:].multiply(fy)
    fx = astype(fx, cdtype) # FIXME: fx = fx * (2.0j * xp.pi)
    fy = astype(fy, cdtype) # FIXME: fy = fy * (2.0j * xp.pi)
    fx = at(fx)[:].multiply(2.0j * xp.pi)
    fy = at(fy)[:].multiply(2.0j * xp.pi)
    f_den = fx + fy

    f_den = at(f_den)[..., 0, 0].set(1.0)  # avoid division by zero warning
    f_num = at(f_num)[:].divide(f_den)
    f_num = at(f_num)[..., 0, 0].set(0.0)

    match fft_method:
        case FFTMethod.FFT:
            return xp.real(ifft_2d(f_num, workers=workers)[..., :y, :x])
        case FFTMethod.RFFT:
            return irfft_2d(f_num, s=s, workers=workers)[..., :y, :x]
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)
