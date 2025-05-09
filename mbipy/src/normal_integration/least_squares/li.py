"""Normal integration using the method of Li et al.

Li, G., Li, Y., Liu, K., Ma, X. & Wang, H. Improving wavefront reconstruction
accuracy by using integration equations with higher-order truncation errors in the
Southwell geometry. J. Opt. Soc. Am. A 30, 1448-1459.
https://opg.optica.org/josaa/abstract.cfm?URI=josaa-30-7-1448 (July 2013).
"""

from __future__ import annotations

__all__ = ("Li", "li")

import functools
from typing import TYPE_CHECKING

from numpy import broadcast_shapes

from mbipy.src.normal_integration.least_squares.utils import (
    BaseSparseNormalIntegration,
    csr_matrix,
    factorized,
)
from mbipy.src.utils import array_namespace, get_dtypes

if TYPE_CHECKING:  # pragma: no cover
    from types import ModuleType
    from typing import Callable

    from numpy import floating
    from numpy.typing import DTypeLike, NDArray

    from mbipy.src.config import _have_scipy

    if _have_scipy:
        from scipy.sparse import spmatrix


def _li_vec(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
    """Compute vector from gradient fields, based on the method of Li et al.

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
    j_3 = j - 3
    i_3 = i - 3
    ij_3 = i * j_3
    ji_3 = j * i_3
    _s = ij_3 + ji_3

    out = xp.empty((_s + 2 * i + 2 * j), dtype=dtype)

    w_size = max(ij_3, ji_3, 2 * i, 2 * j)
    w = xp.empty(w_size, dtype=dtype)  # Work array - sliced for each operation

    w1 = w[:ij_3]
    out1 = out[:ij_3]
    # Equivalent to: 13/24*(gx[:, 1:-2] + gx[:, 2:-1] - 1/13 * (gx[:, :-3] + gx[:, 3:]))
    xp.add(gx[:, 1:-2], gx[:, 2:-1], out=xp.reshape(out1, (i, j_3), copy=False))
    xp.add(gx[:, :-3], gx[:, 3:], out=xp.reshape(w1, (i, j_3), copy=False))
    w1 /= 13.0
    out1 -= w1
    # !!! Done later: out[:ij_3] *= 13 / 24
    del w1, out1  # Deletes the views, not the data - avoid accidental reuse

    w2 = w[:ji_3]
    out2 = out[ij_3:_s]
    # Equivalent to: 13/24*(gy[1:-2, :] + gy[2:-1, :] - 1/13 * (gy[:-3, :] + gy[3:, :]))
    xp.add(gy[1:-2, :], gy[2:-1, :], out=xp.reshape(out2, (i_3, j), copy=False))
    xp.add(gy[:-3, :], gy[3:, :], out=xp.reshape(w2, (i_3, j), copy=False))
    w2 /= 13.0
    out2 -= w2
    out[:_s] *= 13.0 / 24.0  # !!! Here!
    del w2, out2  # Deletes the views, not the data - avoid accidental reuse

    a0_3 = xp.asarray([0, -3], dtype=xp.int64)
    a2_1 = xp.asarray([2, -1], dtype=xp.int64)
    a1_2 = xp.asarray([1, -2], dtype=xp.int64)

    w3 = w[: i * 2]
    out3 = out[_s : _s + 2 * i]
    # Equivalent to: (gx[:, (0, -3)] + gx[:, (2, -1) + 4.0 * gx[:, (1, -2)]]) / 3.0
    xp.add(gx[:, a0_3], gx[:, a2_1], out=xp.reshape(out3, (i, 2), copy=False))
    xp.multiply(gx[:, a1_2], 4.0, out=xp.reshape(w3, (i, 2), copy=False))
    out3 += w3
    out3 /= 3.0
    del w3, out3  # Deletes the views, not the data - avoid accidental reuse

    w4 = w[: 2 * j]
    out4 = out[_s + 2 * i :]
    # Equivalent to: (gy[(0, -3), :] + gy[(2, -1), :] + 4.0 * gy[(1, -2), :]) / 3.0
    xp.add(gy[a0_3, :], gy[a2_1, :], out=xp.reshape(out4, (2, j), copy=False))
    xp.multiply(gy[a1_2, :], 4.0, out=xp.reshape(w4, (2, j), copy=False))
    out4 += w4
    out4 /= 3.0
    del w4, out4, w  # Deletes the views, not the data - avoid accidental reuse

    return out


@functools.lru_cache
def _li_matrix(
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

    ij_3 = i * (j - 3)
    ji_3 = j * (i - 3)

    _s = ij_3 + ji_3

    array = xp.arange(stop, dtype=idtype)

    rows = xp.empty((2, stop), dtype=idtype)
    rows[:] = array
    rows = xp.reshape(rows, -1, copy=False)

    cols = xp.empty((2, stop), dtype=idtype)
    col_view1 = xp.reshape(cols[:, :ij_3], (2, i, j - 3), copy=False)
    col_view1[:] = xp.reshape(array[:n], (i, j), copy=False)[:, 1:-2]
    col_view1[1] += 1  # !!! to avoid copy in the above reshape
    del col_view1  # Deletes the view, not the data - avoid accidental reuse

    cols[0, ij_3:_s] = array[j : j * (i - 2)]
    cols[1, ij_3:_s] = array[j + j : j * (i - 1)]

    cols[0, _s : _s + i] = array[: i * j : j]
    cols[1, _s : _s + i] = array[2 : i * j + 2 : j]

    cols[0, _s + i : _s + 2 * i] = array[j - 3 : i * j - 2 : j]
    cols[1, _s + i : _s + 2 * i] = array[j - 1 : i * j : j]

    cols[0, _s + 2 * i : _s + 2 * i + j] = array[:j]
    cols[1, _s + 2 * i : _s + 2 * i + j] = array[j * 2 : j * 3]

    cols[0, _s + 2 * i + j :] = array[j * (i - 3) : j * (i - 2)]
    cols[1, _s + 2 * i + j :] = array[j * (i - 1) : j * i]
    cols = xp.reshape(cols, -1, copy=False)

    data = xp.empty((2, stop), dtype=fdtype)
    data[0] = -1.0
    data[1] = 1.0
    data = xp.reshape(data, -1, copy=False)

    return csr_matrix(data, rows, cols, shape=(stop, n))


@functools.lru_cache
def _li_factorized_mt(
    shape: tuple[int, int],
    xp: ModuleType,
    idtype: DTypeLike,
    fdtype: DTypeLike,
) -> tuple[Callable[[NDArray], NDArray], spmatrix]:
    m = _li_matrix(shape, xp, idtype, fdtype)
    mt = m.T
    return factorized(mt @ m), mt


def li(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
    """Perform normal integration using the method of Li et al.

    Li, G., Li, Y., Liu, K., Ma, X. & Wang, H. Improving wavefront reconstruction
    accuracy by using integration equations with higher-order truncation errors in the
    Southwell geometry. J. Opt. Soc. Am. A 30, 1448-1459.
    https://opg.optica.org/josaa/abstract.cfm?URI=josaa-30-7-1448 (July 2013).

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

    vector = _li_vec(gy, gx)

    f, mt = _li_factorized_mt(shape, xp, idtype, fdtype)

    mt_vector = mt @ vector

    return xp.reshape(xp.asarray(f(mt_vector)), shape)


class Li(BaseSparseNormalIntegration):
    """Perform normal integration using the method of Li et al.

    Li, G., Li, Y., Liu, K., Ma, X. & Wang, H. Improving wavefront reconstruction
    accuracy by using integration equations with higher-order truncation errors in the
    Southwell geometry. J. Opt. Soc. Am. A 30, 1448-1459.
    https://opg.optica.org/josaa/abstract.cfm?URI=josaa-30-7-1448 (July 2013).
    """

    @staticmethod
    def _factorized_mt_func(
        shape: tuple[int, int],
        xp: ModuleType,
        idtype: DTypeLike,
        fdtype: DTypeLike,
    ) -> tuple[Callable[[NDArray], NDArray], spmatrix]:
        return _li_factorized_mt(shape, xp, idtype, fdtype)

    @staticmethod
    def _vec_func(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
        return _li_vec(gy, gx)
