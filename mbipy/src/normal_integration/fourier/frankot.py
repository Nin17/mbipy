"""Normal integration using the method of Frankot and Chellappa.

Frankot, R. T. & Chellappa, R. A method for enforcing integrability in shape
from shading algorithms.
IEEE Transactions on pattern analysis and machine intelligence 10, 439-451 (1988)
"""

from __future__ import annotations

__all__ = ("frankot",)


from typing import TYPE_CHECKING

from mbipy.src.normal_integration.padding import antisym
from mbipy.src.normal_integration.utils import check_shapes, get_dfts
from mbipy.src.utils import array_namespace, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def frankot(
    gy: NDArray[floating],
    gx: NDArray[floating],
    pad: str | None = None,
    workers: int | None = None,
) -> NDArray[floating]:
    # TODO(nin17): docstring
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    fft2, ifft2 = get_dfts(xp)
    y, x = check_shapes(gx, gy)

    if pad == "antisym":
        gy, gx = antisym(gy=gy, gx=gx)
    elif pad is None:
        pass
    else:
        msg = f"Invalid value for pad: {pad}"
        raise ValueError(msg)

    fx = xp.astype(xp.fft.fftfreq(2 * x if pad else x), dtype, copy=False)
    fy = xp.astype(xp.fft.fftfreq(2 * y if pad else y)[:, None], dtype, copy=False)

    gx_fft = fft2(gx, axes=(-2, -1), workers=workers)
    gy_fft = fft2(gy, axes=(-2, -1), workers=workers)

    f_num = fx * gx_fft + fy * gy_fft
    f_den = 2j * xp.pi * (fx * fx + fy * fy)
    # avoid division by zero warning
    f_den = setitem(f_den, (..., 0, 0), 1.0, xp)
    f_phase = f_num / f_den
    f_phase = setitem(f_phase, (..., 0, 0), 0.0, xp)
    return ifft2(f_phase, axes=(-2, -1), workers=workers).real[..., :y, :x]
