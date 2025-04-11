"""Utilities for normal integration methods that solve sparse least squares problems."""

from __future__ import annotations

__all__ = ("BaseSparseNormalIntegration", "csr_matrix", "factorized")

import importlib
from typing import TYPE_CHECKING

from array_api_compat import array_namespace, is_cupy_namespace, is_numpy_namespace

if TYPE_CHECKING:

    from types import ModuleType
    from typing import Callable

    from numpy import floating, integer
    from numpy.typing import DTypeLike, NDArray

    from mbipy.src.config import __have_scipy__

    if __have_scipy__:
        from scipy.sparse import spmatrix


# TODO(nin17): docstring
def csr_matrix(
    data: NDArray[floating],
    rows: NDArray[integer],
    cols: NDArray[integer],
    shape: tuple[int, int],
) -> spmatrix:
    """Compressed Sparse Row Matrix.

    data, rows and cols satisfy the relationship a[rows[k], cols[k]] = data[k]

    Parameters
    ----------
    data : NDArray[floating]
        _description_
    rows : NDArray[integer]
        _description_
    cols : NDArray[integer]
        _description_
    shape : tuple[int, int]
        _description_

    Returns
    -------
    spmatrix
        _description_
    """
    xp = array_namespace(data, rows, cols)
    if is_numpy_namespace(xp):
        sparse = importlib.import_module("scipy.sparse")
    elif is_cupy_namespace(xp):
        sparse = importlib.import_module("cupyx.scipy.sparse")
    return sparse.csr_matrix((data, (rows, cols)), shape=shape)


# TODO(nin17): docstring
def factorized(A: spmatrix) -> Callable[[NDArray[floating]], NDArray[floating]]:
    """Return a function for solving a sparse linear system, with A pre-factorized.

    Parameters
    ----------
    A : spmatrix
        _description_

    Returns
    -------
    Callable[[NDArray[floating]], NDArray[floating]]
        _description_
    """
    xp = array_namespace(A.data, A.indices, A.indptr)
    if is_numpy_namespace(xp):
        splinalg = importlib.import_module("scipy.sparse.linalg")
    elif is_cupy_namespace(xp):
        splinalg = importlib.import_module("cupyx.scipy.sparse.linalg")
    return splinalg.factorized(A)


class BaseSparseNormalIntegration:

    def __init__(
        self,
        shape: tuple[int, int],
        xp: ModuleType | None = None,
        idtype: DTypeLike | None = None,
        fdtype: DTypeLike | None = None,
    ) -> None:
        """Based class for sparse least squares normal integration."""
        self.shape = shape
        xp = xp or importlib.import_module("numpy")
        self.xp = xp
        idtype = idtype or xp.int64
        fdtype = fdtype or xp.float64
        self.idtype = idtype
        self.fdtype = fdtype

        # TODO(nin17): use _factorized instead of _mat_func
        m = self._mat_func(shape, xp, idtype, fdtype)
        mt = m.T

        self._f = factorized(mt @ m)
        self._mt = mt

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.shape}, xp={self.xp.__name__}, "
            f"idtype={self.idtype.__name__}, fdtype={self.fdtype.__name__})"
        )

    def __call__(
        self,
        gy: NDArray[floating],
        gx: NDArray[floating],
    ) -> NDArray[floating]:
        """Perform normal integration.

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
        shape = self.xp.broadcast_shapes(gy.shape, gx.shape)
        if shape != self.shape:
            msg = "Input arrays must have the same shape as the object."
            raise ValueError(msg)
        vector = self._vec_func(gy, gx)

        return self._f(self._mt @ vector).reshape(shape)

    @staticmethod
    def _mat_func(
        shape: tuple[int, int],
        xp: ModuleType,
        idtype: DTypeLike,
        fdtype: DTypeLike,
    ) -> spmatrix:
        """Matrix function to be implemented by subclasses."""
        msg = "Subclasses must implement _mat_func."
        raise NotImplementedError(msg)

    @staticmethod
    def _vec_func(gy: NDArray[floating], gx: NDArray[floating]) -> NDArray[floating]:
        """Vector function to be implemented by subclasses."""
        msg = "Subclasses must implement _vec_func."
        raise NotImplementedError(msg)
