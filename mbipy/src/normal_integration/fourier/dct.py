"""Solve the poisson equation with the DCT - homogeneous Neumann boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DCT_Poisson.m
"""

from __future__ import annotations

__all__ = ("dct_poisson",)

from typing import TYPE_CHECKING

from mbipy.src.normal_integration.fourier.utils import dct2, idct2
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, cast_scalar, isub, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def dct_poisson(
    gy: NDArray[floating],
    gx: NDArray[floating],
    workers: int | None = None,
) -> NDArray[floating]:
    """Perform normal integration using the DCT solution to the poisson equation.

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    workers : int | None, optional
        Passed to scipy.fft dctn & idctn, by default None

    Returns
    -------
    NDArray[floating]
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.

    """
    xp = array_namespace(gy, gx)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)

    sy, sx = check_shapes(gx, gy)

    # !!! Cast scalars to the same dtype as the result. Necessary for Numba.
    half = cast_scalar(0.5, dtype)
    one = cast_scalar(1.0, dtype)
    four = cast_scalar(4.0, dtype)

    arange = xp.arange(max(sy, sx), dtype=xp.int64)

    indices_y = xp.empty(sy + 2, dtype=xp.int64)
    indices_y = setitem(indices_y, 0, 0)
    indices_y = setitem(indices_y, -1, -1)
    indices_y = setitem(indices_y, slice(1, -1), arange[:sy])

    indices_x = xp.empty(sx + 2, dtype=xp.int64)
    indices_x = setitem(indices_x, 0, 0)
    indices_x = setitem(indices_x, -1, -1)
    indices_x = setitem(indices_x, slice(1, -1), arange[:sx])

    # Divergence of (gy, gx) using central differences
    qy = half * (gy[..., indices_y[2:], :] - gy[..., indices_y[:-2], :])
    px = half * (gx[..., :, indices_x[2:]] - gx[..., :, indices_x[:-2]])

    # Div(gy, gx)
    f = qy + px

    # Modification near the boundaries to enforce the non-homogeneous Neumann
    # BC (Eq. 53 in [1])
    s1_1 = slice(1, -1)
    f = isub(f, (..., 0, s1_1), gy[..., 0, s1_1])
    f = isub(f, (..., -1, s1_1), gy[..., -1, s1_1])
    f = isub(f, (..., s1_1, 0), gx[..., s1_1, 0])
    f = isub(f, (..., s1_1, -1), gx[..., s1_1, -1])
    # Equivalent to:
    # f[..., 0, 1:-1] -= gy[..., 0, 1:-1]
    # f[..., -1, 1:-1] -= gy[..., -1, 1:-1]
    # f[..., 1:-1, 0] -= gx[..., 1:-1, 0]
    # f[..., 1:-1, -1] -= gx[..., 1:-1, -1]

    # Modification near the corners (Eq. 54 in [1])
    f = isub(f, (..., 0, -1), -gy[..., 0, 0] - gx[..., 0, 0])
    f = isub(f, (..., -1, -1), gy[..., -1, -1] + gx[..., -1, -1])
    f = isub(f, (..., -1, 0), gy[..., -1, 0] - gx[..., -1, 0])
    f = isub(f, (..., 0, 0), -gy[..., 0, -1] + gx[..., 0, -1])
    # Equivalent to:
    # f[..., 0, -1] -= -gy[..., 0, 0] - gx[..., 0, 0]
    # f[..., -1, -1] -= gy[..., -1, -1] + gx[..., -1, -1]
    # f[..., -1, 0] -= gy[..., -1, 0] - gx[..., -1, 0]
    # f[..., 0, 0] -= -gy[..., 0, -1] + gx[..., 0, -1]

    fcos = dct2(f, workers=workers)

    # dtype not supported in numba
    x = xp.astype(xp.linspace(0, xp.pi / 2, sx), dtype, copy=False)
    y = xp.astype(xp.linspace(0, xp.pi / 2, sy), dtype, copy=False)[:, None]
    # Faster to do * before + : x.size + y.size vs x.size * y.size
    sinx = xp.sin(x)
    siny = xp.sin(y)

    denom = (four * sinx * sinx) + (four * siny * siny)
    denom = setitem(denom, (0, 0), one)
    z_bar_bar = -fcos / denom

    return idct2(z_bar_bar, workers=workers)
