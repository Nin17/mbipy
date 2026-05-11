"""Solve the poisson equation with the DCT - homogeneous Neumann boundary conditions.

Queau, Y., Durou, J.-D. & Aujol, J.-F. Normal Integration: A Survey.
http://arxiv.org/abs/1709.05940 (Sept. 2017).

Python implementation of: https://github.com/yqueau/normal_integration/blob/1f69b9f1f35bb79457f6a8af753a5d4978811b11/Toolbox/DCT_Poisson.m
"""

from __future__ import annotations

__all__ = ["dct_poisson"]

from typing import TYPE_CHECKING

from mbipy.src.utils import array_namespace, astype, at, get_dtypes

from ._utils import dct2_2d, idct2_2d

if TYPE_CHECKING:
    from numpy import floating
    from numpy.typing import NDArray

# TODO(nin17): dtype kwarg in np.linspace - waiting for numba


def dct_poisson(
    gy: NDArray[floating],
    gx: NDArray[floating],
    workers: int | None = None,
) -> NDArray[floating]:
    # TODO(nin17): reference + doi link
    """Perform normal integration using the DCT solution to the poisson equation.

    ??? info "Array API Compatibility"

        {{ NormalIntegration.row("dct_poisson") | indent(4) }}

    !!! example "[Example][dct_poisson-example]"

    Parameters
    ----------
    gy : NDArray[floating] (..., M, N)
        Vertical gradient(s).
    gx : NDArray[floating] (..., M, N)
        Horizontal gradient(s).
    workers : int | None, optional
        Passed to [scipy.fft.dctn][] & [scipy.fft.idctn][] if `gy` & `gx` are numpy
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
    xp = array_namespace(gy, gx)
    dtype, _ = get_dtypes(gy, gx)
    sy, sx = xp.broadcast_shapes(gx.shape, gy.shape)[-2:]

    arange = xp.arange(max(sy, sx), dtype=xp.int64)

    indices_y = xp.empty(sy + 2, dtype=xp.int64)
    indices_y = at(indices_y)[0].set(0)
    indices_y = at(indices_y)[-1].set(-1)
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

    # Modification near the boundaries to enforce the non-homogeneous Neumann
    # BC (Eq. 53 in [1])
    f = at(f)[..., 0, 1:-1].subtract(gy[..., 0, 1:-1])
    f = at(f)[..., -1, 1:-1].subtract(gy[..., -1, 1:-1])
    f = at(f)[..., 1:-1, 0].subtract(gx[..., 1:-1, 0])
    f = at(f)[..., 1:-1, -1].subtract(gx[..., 1:-1, -1])

    # Modification near the corners (Eq. 54 in [1])
    f = at(f)[..., 0, -1].add(gy[..., 0, 0])
    f = at(f)[..., 0, -1].add(gx[..., 0, 0])
    f = at(f)[..., -1, -1].subtract(gy[..., -1, -1])
    f = at(f)[..., -1, -1].subtract(gx[..., -1, -1])
    f = at(f)[..., -1, 0].subtract(gy[..., -1, 0])
    f = at(f)[..., -1, 0].add(gx[..., -1, 0])
    f = at(f)[..., 0, 0].add(gy[..., 0, -1])
    f = at(f)[..., 0, 0].subtract(gx[..., 0, -1])

    fcos = dct2_2d(f, workers=workers)
    fcos = at(fcos)[:].multiply(-1.0)  # TODO(nin17): avoid - negate everything else

    x = astype(xp.linspace(0.0, xp.pi / 2.0, sx), dtype)  # FIXME: dtype
    y = astype(xp.linspace(0.0, xp.pi / 2.0, sy), dtype)[:, None]  # FIXME: dtype
    # Faster to do * before + : x.size + y.size vs x.size * y.size multiplications
    sinx = xp.sin(x)
    siny = xp.sin(y)
    sinx = at(sinx)[:].multiply(sinx)
    siny = at(siny)[:].multiply(siny)
    fsinx = at(sinx)[:].multiply(4.0)
    fsiny = at(siny)[:].multiply(4.0)

    denom = fsinx + fsiny
    denom = at(denom)[0, 0].set(1.0)
    fcos = at(fcos)[:].divide(denom)

    return idct2_2d(fcos, workers=workers)
