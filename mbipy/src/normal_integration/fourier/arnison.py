"""Normal integration using the method of Arnison et al.

Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
Linear phase imaging using differential interference contrast microscopy.
Journal of microscopy 214, 7-12 (2004).
"""

from __future__ import annotations

__all__ = ("arnison",)

from typing import TYPE_CHECKING, Literal

from mbipy.src.normal_integration.fourier.padding import antisym
from mbipy.src.normal_integration.fourier.utils import (
    fft_2d,
    ifft_2d,
    irfft_2d,
    rfft_2d,
)
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, cast_scalar, idiv, imul, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def arnison(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: Literal["antisym"] | None = None,
    workers: int | None = None,
    use_rfft: bool | None = None,
) -> NDArray[floating]:
    """Perform normal integration using the method of Arnison et al.

    Arnison, M. R., Larkin, K. G., Sheppard, C. J., Smith, N. I. & Cogswell, C. J.
    Linear phase imaging using differential interference contrast microscopy.
    Journal of microscopy 214, 7-12 (2004).

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    pad : Literal["antisym"] | None, optional
        Type of padding to apply: "antisym" | None, by default None
    workers : int | None, optional
        Passed to scipy.fft fft2 & ifft2, by default None
    use_rfft : bool | None, optional
        Use a rfftn instead of fftn, by default None


    Returns
    -------
    (..., M, N) NDArray[floating]
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.
    ValueError
        If the pad argument is not None or "antisym".
    """
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    cdtype = xp.result_type(dtype, xp.complex64)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    y, x = check_shapes(gx, gy)
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x

    if pad == "antisym":
        gy, gx = antisym(gy=gy, gx=gx)
    elif pad is None:
        pass
    else:
        msg = f"Invalid value for pad: {pad}"
        raise ValueError(msg)

    if use_rfft:
        fx = xp.astype(xp.fft.rfftfreq(x2), dtype, copy=False)
    else:
        fx = xp.astype(xp.fft.fftfreq(x2), dtype, copy=False)
    fy = xp.astype(xp.fft.fftfreq(y2), dtype, copy=False)

    # !!! Cast scalars to the same dtype as the result. Necessary for Numba.
    zero = cast_scalar(0.0, dtype)
    one = cast_scalar(1.0, dtype)
    pi = cast_scalar(xp.pi, dtype)
    one_j = cast_scalar(1j, cdtype)
    two_j = cast_scalar(2j, cdtype)

    fx = imul(fx, slice(0, None), pi)
    fy = imul(fy, slice(0, None), pi)

    sinfx = xp.sin(fx)  # TODO(nin17): inplace
    sinfy = xp.sin(fy)

    if use_rfft:
        s = (y2, x2)
        gxfft2 = rfft_2d(gx, s=s, workers=workers)
        gyfft2 = rfft_2d(gy, s=s, workers=workers)
        f_num = gxfft2 + one_j * gyfft2
    else:
        f_num = fft_2d(gx + one_j * gy, workers=workers)
    f_den = two_j * (sinfx + one_j * sinfy[:, None])

    f_den = setitem(f_den, (..., 0, 0), one)  # avoid division by zero warning
    frac = idiv(f_num, (...,), f_den)
    frac = setitem(frac, (..., 0, 0), zero)
    if use_rfft:
        return irfft_2d(frac, s=s, workers=workers)[..., :y, :x]
    return xp.real(ifft_2d(frac, workers=workers)[..., :y, :x])
