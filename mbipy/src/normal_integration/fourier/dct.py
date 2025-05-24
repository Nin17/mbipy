"""Solve the poisson equation with the DCT - homogeneous Neumann boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DCT_Poisson.m
"""

from __future__ import annotations

__all__ = ("dct_poisson",)

from typing import TYPE_CHECKING

from mbipy.src.normal_integration.fourier.utils import dct2_2d, idct2_2d
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import (
    array_namespace,
    astype,
    get_dtypes,
    idiv,
    imul,
    isub,
    setitem,
)

if TYPE_CHECKING:  # pragma: no cover
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
    (..., M, N) NDArray[floating]
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.

    """
    xp = array_namespace(gy, gx)
    dtype, _ = get_dtypes(gy, gx)
    sy, sx = check_shapes(gx, gy)

    arange = xp.arange(max(sy, sx), dtype=xp.int64)

    indices_y = xp.empty(sy + 2, dtype=xp.int64)
    indices_y = setitem(indices_y, 0, 0)
    indices_y = setitem(indices_y, -1, -1)
    indices_y = setitem(indices_y, slice(1, -1), arange[:sy])

    indices_x = xp.empty(sx + 2, dtype=xp.int64)
    indices_x = setitem(indices_x, 0, 0)
    indices_x = setitem(indices_x, -1, -1)
    indices_x = setitem(indices_x, slice(1, -1), arange[:sx])

    # Divergence (∇) of (gy, gx) using central differences
    qy = gy[..., indices_y[2:], :] - gy[..., indices_y[:-2], :]  # ??? do inplace
    px = gx[..., :, indices_x[2:]] - gx[..., :, indices_x[:-2]]

    # ∇(gy, gx)
    f = qy + px
    f = idiv(f, ..., 2.0)

    # Modification near the boundaries to enforce the non-homogeneous Neumann
    # BC (Eq. 53 in [1])
    s1_1 = slice(1, -1)
    # Equivalent to: f[..., 0, 1:-1] -= gy[..., 0, 1:-1]
    f = isub(f, (..., 0, s1_1), gy[..., 0, s1_1])
    # Equivalent to: f[..., -1, 1:-1] -= gy[..., -1, 1:-1]
    f = isub(f, (..., -1, s1_1), gy[..., -1, s1_1])
    # Equivalent to: f[..., 1:-1, 0] -= gx[..., 1:-1, 0]
    f = isub(f, (..., s1_1, 0), gx[..., s1_1, 0])
    # Equivalent to: f[..., 1:-1, -1] -= gx[..., 1:-1, -1]
    f = isub(f, (..., s1_1, -1), gx[..., s1_1, -1])

    # Modification near the corners (Eq. 54 in [1])
    # Equivalent to: f[..., 0, -1] -= -gy[..., 0, 0] - gx[..., 0, 0]
    f = isub(f, (..., 0, -1), -gy[..., 0, 0] - gx[..., 0, 0])
    # Equivalent to: f[..., -1, -1] -= gy[..., -1, -1] + gx[..., -1, -1]
    f = isub(f, (..., -1, -1), gy[..., -1, -1] + gx[..., -1, -1])
    # Equivalent to: f[..., -1, 0] -= gy[..., -1, 0] - gx[..., -1, 0]
    f = isub(f, (..., -1, 0), gy[..., -1, 0] - gx[..., -1, 0])
    # Equivalent to: f[..., 0, 0] -= -gy[..., 0, -1] + gx[..., 0, -1]
    f = isub(f, (..., 0, 0), -gy[..., 0, -1] + gx[..., 0, -1])

    fcos = dct2_2d(f, workers=workers)
    fcos = imul(fcos, ..., -1.0)

    # dtype not supported in numba
    x = astype(xp.linspace(0.0, xp.pi / 2.0, sx), dtype)
    y = astype(xp.linspace(0.0, xp.pi / 2.0, sy), dtype)[:, None]
    # Faster to do * before + : x.size + y.size vs x.size * y.size multiplications
    sinx = xp.sin(x)  # ??? do inplace
    siny = xp.sin(y)
    sinx = imul(sinx, ..., sinx)
    siny = imul(siny, ..., siny)
    fsinx = imul(sinx, ..., 4.0)
    fsiny = imul(siny, ..., 4.0)

    denom = fsinx + fsiny
    denom = setitem(denom, (0, 0), 1.0)
    z_bar_bar = idiv(fcos, ..., denom)

    return idct2_2d(z_bar_bar, workers=workers)
