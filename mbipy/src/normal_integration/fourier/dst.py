"""Solve the poisson equation with the DST - Dirichlet boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DST_Poisson.m
"""

from __future__ import annotations

__all__ = ("dst_poisson",)


from typing import TYPE_CHECKING

from numpy import broadcast_shapes

from mbipy.src.normal_integration.fourier.utils import dst1_2d, idst1_2d
from mbipy.src.normal_integration.utils import check_shapes
from mbipy.src.utils import array_namespace, cast_scalar, isub, setitem

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray


def dst_poisson(
    gy: NDArray[floating],
    gx: NDArray[floating],
    ub: NDArray[floating] | None = None,
    workers: int | None = None,
) -> NDArray[floating]:
    """Perform normal integration using the DST solution to the poisson equation.

    Parameters
    ----------
    gy : (..., M, N) NDArray[floating]
        Vertical gradient(s).
    gx : (..., M, N) NDArray[floating]
        Horizontal gradient(s).
    ub : (..., M, N) NDArray[floating] | None, optional
        Boundary value(s), by default None
    workers : int | None, optional
        Passed to scipy.fft dstn & idstn, by default None

    Returns
    -------
    NDArray[floating]
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.

    """
    xp = array_namespace(gx, gy)
    dtype = xp.result_type(gy, gx)
    if not xp.isdtype(dtype, "real floating"):
        msg = "Input arrays must be real-valued."
        raise ValueError(msg)
    # !!! Slower algorithm if not using SciPy or Numba: doubles the array size
    sy, sx = check_shapes(gx, gy)
    if ub is not None:
        result_shape = broadcast_shapes(gy.shape, gx.shape, ub.shape)
    else:
        result_shape = broadcast_shapes(gy.shape, gx.shape)

    # !!! Cast scalars to the same dtype as the result. Necessary for Numba.
    half = cast_scalar(0.5, dtype)
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

    qy = half * (gy[..., indices_y[2:], :] - gy[..., indices_y[:-2], :])
    px = half * (gx[..., :, indices_x[2:]] - gx[..., :, indices_x[:-2]])

    f = qy + px

    if ub is not None:
        # Modification near the boundaries (Eq. 46 in [1])
        s2_2 = slice(2, -2)
        f = isub(f, (..., 1, s2_2), ub[..., 0, s2_2])
        f = isub(f, (..., -2, s2_2), ub[..., -1, s2_2])
        f = isub(f, (..., s2_2, 1), ub[..., s2_2, 0])
        f = isub(f, (..., s2_2, -2), ub[..., s2_2, -1])
        # Equivalent to:
        # f[..., 1, 2:-2] -= ub[..., 0, 2:-2]
        # f[..., -2, 2:-2] -= ub[..., -1, 2:-2]
        # f[..., 2:-2, 1] -= ub[..., 2:-2, 0]
        # f[..., 2:-2, -2] -= ub[..., 2:-2, -1]

        # # Modification near the corners (Eq. 47 in [1])
        f = isub(f, (..., 1, 1), ub[..., 1, 0] + ub[..., 0, 1])
        f = isub(f, (..., 1, -2), ub[..., 1, -1] + ub[..., 0, -2])
        f = isub(f, (..., -2, -2), ub[..., -2, -1] + ub[..., -1, -2])
        f = isub(f, (..., -2, 1), ub[..., -2, 0] + ub[..., -1, 1])
        # f[..., 1, 1] -= ub[..., 1, 0] + ub[..., 0, 1]
        # f[..., 1, -2] -= ub[..., 1, -1] + ub[..., 0, -2]
        # f[..., -2, -2] -= ub[..., -2, -1] + ub[..., -1, -2]
        # f[..., -2, 1] -= ub[..., -2, 0] + ub[..., -1, 1]

    fsin = dst1_2d(f[..., 1:-1, 1:-1], workers=workers)

    # dtype not supported in numba
    x = xp.astype(xp.linspace(0.0, xp.pi / 2.0, sx), dtype, copy=False)[1:-1]
    y = xp.astype(xp.linspace(0.0, xp.pi / 2.0, sy), dtype, copy=False)[1:-1][:, None]
    # Faster to do * before + : x.size + y.size vs x.size * y.size
    sinx = xp.sin(x)
    siny = xp.sin(y)

    denom = (-four * sinx * sinx) + (-four * siny * siny)
    z_bar = fsin / denom
    z = xp.zeros(result_shape, dtype=fsin.dtype)
    if ub is not None:
        z = setitem(z, (...,), ub)  # z[:] = ub

    s1_1 = slice(1, -1)
    return setitem(z, (..., s1_1, s1_1), idst1_2d(z_bar, workers=workers))
    # Equivalent to:
    # z[..., 1:-1, 1:-1] = idst2(z_bar, type=1, workers=workers)
    # return z
