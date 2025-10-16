"""Normal integration using the method of Frankot and Chellappa.

Frankot, R. T. & Chellappa, R. A method for enforcing integrability in shape
from shading algorithms.
IEEE Transactions on pattern analysis and machine intelligence 10, 439-451 (1988)
"""

from __future__ import annotations

__all__ = ("frankot",)


from typing import TYPE_CHECKING, Literal

from mbipy.src.normal_integration.fourier.padding import antisymmetric
from mbipy.src.normal_integration.fourier.utils import (
    FFTMethod,
    fft_2d,
    ifft_2d,
    irfft_2d,
    rfft_2d,
)
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import (
    array_namespace,
    astype,
    get_dtypes,
    idiv,
    imul,
    setitem,
)

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def frankot(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: Literal["antisymmetric"] | None = None,
    workers: int | None = None,
    fft_method: FFTMethod = FFTMethod.FFT,
) -> NDArray[floating]:
    """Perform normal integration using the method of Frankot and Chellappa.

    Frankot, R. T. & Chellappa, R. A method for enforcing integrability in shape
    from shading algorithms.
    IEEE Transactions on pattern analysis and machine intelligence 10, 439-451 (1988)

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    pad : Literal["antisymmetric"] | None, optional
        Type of padding to apply: "antisymmetric" | None , by default None
    workers : int | None, optional
        Passed to scipy.fft fftn & ifftn, by default None
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
        If the pad argument is not None or "antisymmetric".

    """
    xp = array_namespace(gy, gx)
    dtype, cdtype = get_dtypes(gy, gx)
    y, x = check_shapes(gx, gy)
    y2, x2 = 2 * y if pad else y, 2 * x if pad else x

    match pad:
        case "antisymmetric":
            gy, gx = antisymmetric(gy=gy, gx=gx)
        case None:
            ...
        case _:
            msg = f"Invalid value for pad: {pad}"
            raise ValueError(msg)

    fx = astype(
        xp.fft.rfftfreq(x2) if fft_method == FFTMethod.RFFT else xp.fft.fftfreq(x2),
        dtype,
    )
    fy = astype(xp.fft.fftfreq(y2)[:, None], dtype)

    match fft_method:
        case FFTMethod.RFFT:
            s = (y2, x2)
            gx_fft = rfft_2d(gx, s=s, workers=workers)
            gy_fft = rfft_2d(gy, s=s, workers=workers)
        case FFTMethod.FFT:
            gx_fft = fft_2d(astype(gx, cdtype, copy=True), workers=workers)
            gy_fft = fft_2d(astype(gy, cdtype, copy=True), workers=workers)
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)

    gx_fft = imul(gx_fft, ..., fx)
    gy_fft = imul(gy_fft, ..., fy)
    f_num = gx_fft + gy_fft

    fx = imul(fx, ..., fx)
    fy = imul(fy, ..., fy)
    fx = astype(fx, cdtype)
    fy = astype(fy, cdtype)
    fx = imul(fx, ..., 2.0j * xp.pi)
    fy = imul(fy, ..., 2.0j * xp.pi)
    f_den = fx + fy

    f_den = setitem(f_den, (..., 0, 0), 1.0)  # avoid division by zero warning
    frac = idiv(f_num, ..., f_den)  # f_num is frac
    frac = setitem(frac, (..., 0, 0), 0.0)

    match fft_method:
        case FFTMethod.RFFT:
            return irfft_2d(frac, s=s, workers=workers)[..., :y, :x]
        case FFTMethod.FFt:
            return xp.real(ifft_2d(frac, workers=workers)[..., :y, :x])
        case _:
            msg = f"Invalid value for fft_method: {fft_method}"
            raise ValueError(msg)
