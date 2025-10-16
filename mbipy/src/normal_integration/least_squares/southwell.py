"""Normal integration using the method of Southwell.

Southwell, W. Wave-front estimation from wave-front slope measurements.
J. Opt. Soc. Am. 70, 998-1006.
https://opg.optica.org/abstract.cfm?URI=josa-70-8-998 (Aug. 1980).
"""

from __future__ import annotations

__all__ = ("Southwell", "southwell")

import functools
from typing import TYPE_CHECKING

from numpy import broadcast_shapes

from mbipy.src.normal_integration.least_squares.utils import (
    BaseSparseNormalIntegration,
    csr_matrix,
    factorized,
)
from mbipy.src.utils import array_namespace, get_dtypes

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from numpy import floating
    from numpy.typing import DTypeLike, NDArray

    from mbipy.src.config import config as cfg

    if cfg.have_scipy:
        from scipy.sparse import spmatrix


def _southwell_vec(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
    """Compute vector from gradient fields, based on the method of Southwell.

    Parameters
    ----------
    gy : (M, N) NDArray[floating]
        Vertical gradient.
    gx : (M, N) NDArray[floating]
        Horizontal gradient.

    Returns
    -------
    (2MN - (M+N)) NDArray[floating]
        Vector of size 2MN - (M+N), solve with sparse matrix for given image shape.

    """
    xp = array_namespace(gy, gx)
    shape = broadcast_shapes(gy.shape, gx.shape)
    dtype, _ = get_dtypes(gy, gx)
    i, j = shape
    i_1 = i - 1
    j_1 = j - 1
    ij_1 = i * j_1

    out = xp.empty(2 * i * j - i - j, dtype=dtype)
    # Equivalent to: 0.5 * (gx[:, 1:] + gx[:, :-1])
    xp.add(gx[:, 1:], gx[:, :-1], out=xp.reshape(out[:ij_1], (i, j_1), copy=False))
    # Equivalent to: 0.5 * (gy[:-1, :] + gy[1:, :])
    xp.add(gy[:-1, :], gy[1:, :], out=xp.reshape(out[ij_1:], (i_1, j), copy=False))
    out /= 2.0
    return out


@functools.lru_cache
def _southwell_matrix(
    shape: tuple[int, int],
    xp: ModuleType,
    idtype: DTypeLike,
    fdtype: DTypeLike,
) -> spmatrix:
    i, j = shape
    n = i * j
    ij_1 = i * (j - 1)
    ji_1 = j * (i - 1)
    stop = ij_1 + ji_1
    array = xp.arange(stop, dtype=idtype)

    rows = xp.empty((2, stop), dtype=idtype)
    rows[:, :] = array
    rows = xp.reshape(rows, -1, copy=False)

    cols = xp.empty((2, stop), dtype=idtype)
    col_view1 = xp.reshape(cols[:, :ij_1], (2, i, j - 1), copy=False)
    col_view1[:, :, :] = xp.reshape(array[:n], (i, j), copy=False)[:, :-1]
    col_view1[1, :, :] += 1  # !!! to avoid copy in the above reshape
    del col_view1  # Deletes the view, not the data - avoid accidental reuse
    cols[0, ij_1:] = array[:ji_1]
    cols[1, ij_1:] = array[j : i * j]
    cols = xp.reshape(cols, -1, copy=False)

    data = xp.empty((2, stop), dtype=fdtype)
    data[0, :] = -1.0
    data[1, :] = 1.0
    data = xp.reshape(data, -1, copy=False)

    return csr_matrix(data, rows, cols, shape=(stop, n))


@functools.lru_cache
def _southwell_factorized_mt(
    shape: tuple[int, int],
    xp: ModuleType,
    idtype: DTypeLike,
    fdtype: DTypeLike,
) -> tuple[Callable[[NDArray], NDArray], spmatrix]:
    m = _southwell_matrix(shape, xp, idtype, fdtype)
    mt = m.T
    return factorized(mt @ m), mt


def southwell(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
    """Perform normal integration using the method of Southwell.

    Southwell, W. Wave-front estimation from wave-front slope measurements.
    J. Opt. Soc. Am. 70, 998-1006.
    https://opg.optica.org/abstract.cfm?URI=josa-70-8-998 (Aug. 1980).

    Parameters
    ----------
    gy : (M, N) NDArray[floating]
        Vertical gradient.
    gx : (M, N) NDArray[floating]
        Horizontal gradient.

    Returns
    -------
    (M, N) NDArray[floating]
        Normal field.

    """
    xp = array_namespace(gy, gx)
    shape = broadcast_shapes(gy.shape, gx.shape)
    i, j = shape
    stop = 2 * i * j - i - j
    idtype = xp.int32 if stop < xp.iinfo(xp.int32).max else xp.int64
    fdtype = xp.result_type(gy.dtype, gx.dtype)

    vector = _southwell_vec(gy, gx)

    f, mt = _southwell_factorized_mt(shape, xp, idtype, fdtype)

    mt_vector = mt @ vector

    return xp.reshape(xp.asarray(f(mt_vector)), shape)


class Southwell(BaseSparseNormalIntegration):
    """Perform normal integration using the method of Southwell.

    Southwell, W. Wave-front estimation from wave-front slope measurements.
    J. Opt. Soc. Am. 70, 998-1006.
    https://opg.optica.org/abstract.cfm?URI=josa-70-8-998 (Aug. 1980).
    """

    @staticmethod
    def _factorized_mt_func(
        shape: tuple[int, int],
        xp: ModuleType,
        idtype: DTypeLike,
        fdtype: DTypeLike,
    ) -> tuple[Callable[[NDArray], NDArray], spmatrix]:
        return _southwell_factorized_mt(shape, xp, idtype, fdtype)

    @staticmethod
    def _vec_func(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
        return _southwell_vec(gy, gx)
