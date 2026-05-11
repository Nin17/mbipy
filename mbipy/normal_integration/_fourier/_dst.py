"""Solve the poisson equation with the DST - Dirichlet boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DST_Poisson.m
"""

from __future__ import annotations

__all__ = ["dst_poisson"]


from typing import TYPE_CHECKING

from mbipy.src.utils import array_namespace, astype, at, get_dtypes

from ._utils import dst1_2d, idst1_2d

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

# TODO(nin17): dtype kwarg in np.linspace - waiting for numba
# TODO(nin17): try to do this with type 2/3/4 dst - faster algorithm


def dst_poisson(
    gy: NDArray[floating],
    gx: NDArray[floating],
    ub: NDArray[floating] | None = None,
    workers: int | None = None,
) -> NDArray[floating]:
    # TODO(nin17): reference + doi link
    """Perform normal integration using the DST solution to the poisson equation.

    ??? info "Array API Compatibility"

        {{ NormalIntegration.row("dst_poisson") | indent(4) }}

    !!! example "[Example][dst_poisson-example]"

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).
    ub : NDArray[floating] (..., M, N) | None, optional
        Boundary value(s), by default `None`
    workers : int | None, optional
        Passed to [scipy.fft.dstn][] & [scipy.fft.idstn][] if `gy` & `gx` are numpy
        arrays, by default `None`

    Returns
    -------
    NDArray[floating] (..., M, N)
        Normal field(s).

    Raises
    ------
    ValueError
        If the input arrays are not real-valued.

    """
    xp = array_namespace(gx, gy)
    dtype, _ = get_dtypes(gy, gx)
    sy, sx = xp.broadcast_shapes(gx.shape, gy.shape)[-2:]

    if ub is not None:
        result_shape = xp.broadcast_shapes(gy.shape, gx.shape, ub.shape)
    else:
        result_shape = xp.broadcast_shapes(gy.shape, gx.shape)

    arange = xp.arange(max(sy, sx), dtype=xp.int64)

    indices_y = xp.empty(sy + 2, dtype=xp.int64)
    indices_y = at(indices_y)[0].set(0)
    indices_y = at(indices_y)[-1].set(0)
    indices_y = at(indices_y)[1:-1].set(arange[:sy])

    indices_x = xp.empty(sx + 2, dtype=xp.int64)
    indices_x = at(indices_x)[0].set(0)
    indices_x = at(indices_x)[-1].set(-1)
    indices_x = at(indices_x)[1:-1].set(arange[:sx])

    # Divergence (∇) of (gy, gx) using central differences
    qy = gy[..., indices_y[2:], :] - gy[..., indices_y[:-2], :]
    px = gx[..., :, indices_x[2:]] - gx[..., :, indices_x[:-2]]

    # ∇(gy, gx)
    f = qy + px
    f = at(f)[:].divide(2.0)

    if ub is not None:
        # Modification near the boundaries (Eq. 46 in [1])
        f = at(f)[..., 1, 2:-2].subtract(ub[..., 0, 2:-2])
        f = at(f)[..., -2, 2:-2].subtract(ub[..., -1, 2:-2])
        f = at(f)[..., 2:-2, 1].subtract(ub[..., 2:-2, 0])
        f = at(f)[..., 2:-2, -2].subtract(ub[..., 2:-2, -1])

        # # Modification near the corners (Eq. 47 in [1])
        f = at(f)[..., 1, 1].subtract(ub[..., 1, 0])
        f = at(f)[..., 1, 1].subtract(ub[..., 0, 1])
        f = at(f)[..., 1, -2].subtract(ub[..., 1, -1])
        f = at(f)[..., 1, -2].subtract(ub[..., 0, -2])
        f = at(f)[..., -2, -2].subtract(ub[..., -2, -1])
        f = at(f)[..., -2, -2].subtract(ub[..., -1, -2])
        f = at(f)[..., -2, 1].subtract(ub[..., -2, 0])
        f = at(f)[..., -2, 1].subtract(ub[..., -1, 1])

    fsin = dst1_2d(f[..., 1:-1, 1:-1], workers=workers)

    x = astype(xp.linspace(0.0, xp.pi / 2.0, sx), dtype)[1:-1]  # FIXME: dtype
    y = astype(xp.linspace(0.0, xp.pi / 2.0, sy), dtype)[1:-1][:, None]  # FIXME: dtype
    # Faster to do * before + : x.size + y.size vs x.size * y.size multiplications
    sinx = xp.sin(x)
    siny = xp.sin(y)
    sinx = at(sinx)[:].multiply(sinx)
    siny = at(siny)[:].multiply(siny)
    sinx = at(sinx)[:].multiply(-4.0)
    siny = at(siny)[:].multiply(-4.0)

    denom = sinx + siny
    fsin = at(fsin)[:].divide(denom)
    z = xp.zeros(result_shape, dtype=fsin.dtype)
    if ub is not None:
        z = at(z)[:].set(ub)

    return at(z)[..., 1:-1, 1:-1].set(idst1_2d(fsin, workers=workers))
